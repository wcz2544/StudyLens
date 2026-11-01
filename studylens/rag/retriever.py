"""SQLite 活动版本目录 + Milvus/BM25 双路召回 + 重排 + 父块回填。"""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
from time import perf_counter
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Plus

from .chunking import ChunkingConfig, SemanticHierarchySplitter
from .embeddings import validate_vectors
from .types import ConcurrentUpdateError, Evidence, IndexedChunk, KnowledgeError, Reranker, VectorStore

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """英文按词，中文用单字和连续双字，避免英文缩写被逐字拆散。"""
    words = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u3400-\u9fff]+", text.lower())
    result = []
    for word in words:
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            result.extend(word)
            result.extend(word[i:i + 2] for i in range(len(word) - 1))
        else:
            result.append(word)
    return result


def reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """融合名次而不是相加不在同一量纲的 BM25 和余弦分数。"""
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, (identifier, _) in enumerate(ranking, 1):
            if identifier not in seen:
                scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
                seen.add(identifier)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class KnowledgeBase:
    """有界单节点知识库；SQLite 目录是可见性依据，Milvus 不能自行发布版本。"""

    def __init__(self, root_dir, embeddings: Embeddings, vector_store: VectorStore,
                 reranker: Reranker, *, embedding_id: str | None = None,
                 chunking_config: ChunkingConfig | None = None,
                 max_documents: int = 1000, max_total_chunks: int = 20000):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root_dir / "catalog.sqlite3"
        self.embeddings, self.vector_store, self.reranker = embeddings, vector_store, reranker
        self.embedding_id = embedding_id or getattr(embeddings, "embedding_id", None)
        if not self.embedding_id:
            raise ValueError("Embedding 必须提供稳定模型身份，不能混用不同模型的索引。")
        self.dimension = vector_store.dimension
        self.splitter = SemanticHierarchySplitter(embeddings, chunking_config)
        config = {"version": 2, **asdict(self.splitter.config)}
        self.pipeline = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if not 1 <= max_documents <= 10000 or not 1 <= max_total_chunks <= 100000:
            raise ValueError("知识库容量配置无效。")
        self.max_documents, self.max_total_chunks = max_documents, max_total_chunks
        self._cache = OrderedDict()
        self._cache_lock = RLock()
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS kb_config (
                    kb_id TEXT PRIMARY KEY, embedding_id TEXT NOT NULL,
                    dimension INTEGER NOT NULL, pipeline TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    kb_id TEXT NOT NULL, source_id TEXT NOT NULL, name TEXT NOT NULL,
                    revision TEXT NOT NULL, checksum TEXT NOT NULL, chunk_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL, metadata TEXT NOT NULL,
                    PRIMARY KEY (kb_id, source_id)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, kb_id TEXT NOT NULL, source_id TEXT NOT NULL,
                    revision TEXT NOT NULL, parent_id TEXT NOT NULL, source TEXT NOT NULL,
                    section TEXT NOT NULL, text TEXT NOT NULL, parent_text TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_by_kb ON chunks(kb_id, revision);
                CREATE TABLE IF NOT EXISTS vector_cleanup (
                    kb_id TEXT NOT NULL, revision TEXT NOT NULL, state TEXT NOT NULL,
                    created_at TEXT NOT NULL, PRIMARY KEY (kb_id, revision)
                );
            """)

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_ids(kb_id: str, source_id: str | None = None):
        if not isinstance(kb_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", kb_id):
            raise ValueError("知识库标识须为 1～64 位英文、数字、点、下划线或连字符。")
        if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()
                                      or len(source_id) > 200 or "\x00" in source_id):
            raise ValueError("文档标识须为 1～200 个有效字符。")

    def _check_config(self, connection, kb_id: str, create: bool = False):
        row = connection.execute("SELECT * FROM kb_config WHERE kb_id=?", (kb_id,)).fetchone()
        if row and (row["embedding_id"], row["dimension"], row["pipeline"]) != (
                self.embedding_id, self.dimension, self.pipeline):
            raise KnowledgeError("Embedding 模型、维度或分块策略已变更，请在新知识库完整重建后切换。")
        if not row and create:
            connection.execute("INSERT INTO kb_config VALUES (?,?,?,?)",
                               (kb_id, self.embedding_id, self.dimension, self.pipeline))

    def upsert(self, kb_id: str, source_id: str, name: str, text: str,
               metadata: dict | None = None) -> dict:
        # 语义分块与最终索引共用一次导入预算；测试替身不实现预算时仍可显式注入。
        with getattr(self.embeddings, "operation_budget", nullcontext)():
            return self._upsert(kb_id, source_id, name, text, metadata)

    def _upsert(self, kb_id: str, source_id: str, name: str, text: str,
                metadata: dict | None = None) -> dict:
        self._validate_ids(kb_id, source_id)
        if not isinstance(name, str) or not name.strip() or len(name) > 255 or "\x00" in name:
            raise ValueError("文档名称须为 1～255 个有效字符。")
        if not isinstance(text, str) or not text.strip() or "\x00" in text:
            raise ValueError("文档没有可索引的文字。")
        if len(text.encode("utf-8")) > 2 * 1024 * 1024:
            raise ValueError("提取文本不能超过 2 MiB，请拆分文档。")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("文档元数据必须为对象。")
        serialized = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(serialized.encode("utf-8")) > 16384:
            raise ValueError("文档元数据过大。")
        checksum = hashlib.sha256((name + "\0" + text + "\0" + serialized).encode("utf-8")).hexdigest()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_config(connection, kb_id, create=True)
            old = connection.execute("SELECT * FROM documents WHERE kb_id=? AND source_id=?",
                                     (kb_id, source_id)).fetchone()
            count = connection.execute("SELECT COUNT(*) FROM documents WHERE kb_id=?", (kb_id,)).fetchone()[0]
            if old is None and count >= self.max_documents:
                raise ValueError("知识库文档数量已达到配置上限。")
            if old and old["checksum"] == checksum:
                return {**dict(old), "metadata": json.loads(old["metadata"]), "status": "unchanged"}
        prior_revision = old["revision"] if old else None
        revision = uuid4().hex
        chunks = self.splitter.split(kb_id, source_id, revision, name, text, metadata)
        vectors = validate_vectors(self.embeddings.embed_documents(
            [f"{chunk.section}\n{chunk.text}" for chunk in chunks]), len(chunks), self.dimension)
        now = datetime.now(timezone.utc).isoformat()
        # 发布前先准备向量；阶段记录让失败/中断产生的孤儿向量可定位。
        with self._connection() as connection:
            connection.execute("INSERT INTO vector_cleanup VALUES (?,?,?,?)", (kb_id, revision, "staged", now))
        try:
            self.vector_store.upsert(chunks, vectors)
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._check_config(connection, kb_id)
                active = connection.execute("SELECT revision FROM documents WHERE kb_id=? AND source_id=?",
                                            (kb_id, source_id)).fetchone()
                if (active["revision"] if active else None) != prior_revision:
                    raise ConcurrentUpdateError("同一文档已被其他请求更新，请刷新后重试。")
                count = connection.execute("SELECT COUNT(*) FROM documents WHERE kb_id=?", (kb_id,)).fetchone()[0]
                if active is None and count >= self.max_documents:
                    raise ValueError("知识库文档数量已达到配置上限。")
                retained = connection.execute("SELECT COUNT(*) FROM chunks WHERE kb_id=? AND source_id<>?",
                                              (kb_id, source_id)).fetchone()[0]
                if retained + len(chunks) > self.max_total_chunks:
                    raise ValueError("知识库切片数量已达到配置上限。")
                connection.execute("DELETE FROM chunks WHERE kb_id=? AND source_id=?", (kb_id, source_id))
                connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)", [
                    (c.id, c.kb_id, c.source_id, c.revision, c.parent_id, c.source,
                     c.section, c.text, c.parent_text, json.dumps(c.metadata, ensure_ascii=False))
                    for c in chunks])
                connection.execute("INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                                   (kb_id, source_id, name, revision, checksum, len(chunks), now, serialized))
                connection.execute("DELETE FROM vector_cleanup WHERE kb_id=? AND revision=?", (kb_id, revision))
                if prior_revision:
                    connection.execute("INSERT OR REPLACE INTO vector_cleanup VALUES (?,?,?,?)",
                                       (kb_id, prior_revision, "garbage", now))
        except Exception:
            # 未发布的版本永远不在活动过滤条件中，失败不会污染现有检索结果。
            with self._connection() as connection:
                connection.execute("UPDATE vector_cleanup SET state='garbage' WHERE kb_id=? AND revision=?",
                                   (kb_id, revision))
            self.cleanup_pending(kb_id)
            raise
        cleanup = self.cleanup_pending(kb_id)
        return {"status": "updated" if old else "created", "source_id": source_id, "name": name,
                "revision": revision, "checksum": checksum, "chunk_count": len(chunks), "updated_at": now,
                "metadata": metadata or {}, "cleanup_pending": cleanup["remaining"]}

    def delete(self, kb_id: str, source_id: str) -> dict:
        self._validate_ids(kb_id, source_id)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT revision FROM documents WHERE kb_id=? AND source_id=?",
                                     (kb_id, source_id)).fetchone()
            if row is None:
                return {"status": "not_found", "source_id": source_id}
            # 先撤销活动目录，随后才清理物理向量；清理失败也不会重新暴露已删除内容。
            connection.execute("DELETE FROM documents WHERE kb_id=? AND source_id=?", (kb_id, source_id))
            connection.execute("DELETE FROM chunks WHERE kb_id=? AND source_id=?", (kb_id, source_id))
            connection.execute("INSERT OR REPLACE INTO vector_cleanup VALUES (?,?,?,?)",
                               (kb_id, row["revision"], "garbage", datetime.now(timezone.utc).isoformat()))
        cleanup = self.cleanup_pending(kb_id)
        return {"status": "deleted", "source_id": source_id, "revision": row["revision"],
                "cleanup_pending": cleanup["remaining"]}

    def cleanup_pending(self, kb_id: str) -> dict:
        """重试已确认无效的向量；staged 不自动删，以免误删另一进程正在构建的版本。"""
        self._validate_ids(kb_id)
        with self._connection() as connection:
            pending = connection.execute("SELECT revision FROM vector_cleanup WHERE kb_id=? AND state='garbage' LIMIT 50",
                                         (kb_id,)).fetchall()
        cleaned = 0
        for row in pending:
            try:
                self.vector_store.delete(kb_id, [row["revision"]])
            except Exception:
                logger.warning("vector_cleanup_deferred")
                continue
            with self._connection() as connection:
                connection.execute("DELETE FROM vector_cleanup WHERE kb_id=? AND revision=? AND state='garbage'",
                                   (kb_id, row["revision"]))
            cleaned += 1
        with self._connection() as connection:
            remaining = connection.execute("SELECT COUNT(*) FROM vector_cleanup WHERE kb_id=?", (kb_id,)).fetchone()[0]
        return {"cleaned": cleaned, "remaining": remaining}

    def list_documents(self, kb_id: str) -> list[dict]:
        self._validate_ids(kb_id)
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM documents WHERE kb_id=? ORDER BY source_id", (kb_id,)).fetchall()
        return [{**dict(row), "metadata": json.loads(row["metadata"])} for row in rows]

    def _snapshot(self, kb_id: str) -> tuple[dict[str, str], list[IndexedChunk]]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            self._check_config(connection, kb_id)
            active = {row["source_id"]: row["revision"] for row in connection.execute(
                "SELECT source_id, revision FROM documents WHERE kb_id=?", (kb_id,))}
            rows = connection.execute("SELECT * FROM chunks WHERE kb_id=? ORDER BY id", (kb_id,)).fetchall()
        chunks = [IndexedChunk(**{**dict(row), "metadata": json.loads(row["metadata"])})
                  for row in rows if active.get(row["source_id"]) == row["revision"]]
        return active, chunks

    def validate_evidence(self, kb_id: str, evidence: list[Evidence]) -> bool:
        self._validate_ids(kb_id)
        with self._connection() as connection:
            connection.execute("BEGIN")
            for item in evidence:
                row = connection.execute(
                    "SELECT c.revision, c.source_id FROM chunks c JOIN documents d "
                    "ON c.kb_id=d.kb_id AND c.source_id=d.source_id AND c.revision=d.revision "
                    "WHERE c.kb_id=? AND c.id=?", (kb_id, item.id)).fetchone()
                if (row is None or item.metadata.get("kb_id") != kb_id
                        or row["revision"] != item.metadata.get("revision")
                        or row["source_id"] != item.metadata.get("source_id")):
                    return False
        return True

    def _lexical(self, kb_id: str, active: dict, chunks: list[IndexedChunk], query: str, limit: int):
        signature = tuple(sorted(active.items()))
        with self._cache_lock:
            cached = self._cache.get(kb_id)
            if cached is None or cached[0] != signature:
                corpus = [tokenize(c.section + "\n" + c.text) or ["__empty__"] for c in chunks]
                # BM25Plus(delta=0) 保持正 IDF，避免单文档/高频词在 Okapi 中产生负分。
                cached = (signature, BM25Plus(corpus, delta=0))
                self._cache[kb_id] = cached
                self._cache.move_to_end(kb_id)
                while len(self._cache) > 16:
                    self._cache.popitem(last=False)
            scores = cached[1].get_scores(tokenize(query))
        return sorted([(c.id, float(score)) for c, score in zip(chunks, scores) if score > 0],
                      key=lambda row: (-row[1], row[0]))[:limit]

    def search(self, kb_id: str, query: str, top_k: int = 5) -> list[Evidence]:
        self._validate_ids(kb_id)
        if not isinstance(query, str) or len(query) > 2000:
            raise ValueError("检索问题不能超过 2000 字符。")
        if not 1 <= top_k <= 20:
            raise ValueError("检索数量必须在 1～20。")
        if not query.strip() or not tokenize(query):
            return []
        started = perf_counter()
        for attempt in range(2):
            active, chunks = self._snapshot(kb_id)
            if not chunks:
                return []
            chunk_map = {chunk.id: chunk for chunk in chunks}
            limit = min(60, max(20, top_k * 5))
            lexical = self._lexical(kb_id, active, chunks, query, limit)
            vector = validate_vectors([self.embeddings.embed_query(query)], 1, self.dimension)[0]
            dense = self.vector_store.search(kb_id, vector, list(active.values()), limit)
            # 再验一次返回 id，即使存储层误返回跨库/旧版本，仍不能进上下文。
            dense = [(cid, score) for cid, score in dense
                     if cid in chunk_map and math.isfinite(score) and score > 0]
            fused = reciprocal_rank_fusion([lexical, dense])[:limit]
            lexical_scores, dense_scores = dict(lexical), dict(dense)
            candidates = []
            for identifier, score in fused:
                chunk = chunk_map[identifier]
                candidates.append(Evidence(identifier, chunk.source, chunk.section,
                    chunk.section + "\n" + chunk.text, score, {
                        **chunk.metadata, "kb_id": kb_id, "source_id": chunk.source_id,
                        "revision": chunk.revision, "parent_id": chunk.parent_id,
                        "child_text": chunk.text, "rrf_score": score,
                        "bm25_score": lexical_scores.get(identifier), "dense_score": dense_scores.get(identifier),
                    }))
            ranked = self.reranker.rerank(query, candidates, len(candidates)) if candidates else []
            allowed = {candidate.id: candidate for candidate in candidates}
            output, parents = [], set()
            for item in ranked:
                if item.id not in allowed or not math.isfinite(item.score):
                    raise KnowledgeError("重排器返回了不属于当前候选集的证据。")
                chunk = chunk_map[item.id]
                if chunk.parent_id not in parents:
                    output.append(replace(allowed[item.id], text=chunk.parent_text, score=item.score,
                                          metadata={**allowed[item.id].metadata,
                                                    "rerank_score": item.score,
                                                    "rerank_model": item.metadata.get("rerank_model", "explicit-test-double")}))
                    parents.add(chunk.parent_id)
                if len(output) == top_k:
                    break
            current = {doc["source_id"]: doc["revision"] for doc in self.list_documents(kb_id)}
            if current == active:
                # 只记录计数/耗时，不记录 query、正文、来源名或密钥。
                logger.info("rag_search_complete", extra={"candidate_count": len(candidates),
                            "result_count": len(output), "elapsed_ms": round((perf_counter() - started) * 1000)})
                return output
        raise ConcurrentUpdateError("知识库在检索期间持续更新，请稍后重试。")
