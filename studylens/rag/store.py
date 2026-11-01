"""向量存储适配器：生产 Milvus；显式内存替身仅供测试。"""
import hashlib
import json
import math
import re
from threading import RLock
from typing import Sequence

from .embeddings import validate_vectors
from .types import IndexedChunk, KnowledgeError


class InMemoryVectorStore:
    """测试替身：不持久化，不构成生产 Milvus 的降级方案。"""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self._rows = {}
        self._lock = RLock()

    def upsert(self, chunks: Sequence[IndexedChunk], vectors: list[list[float]]) -> None:
        checked = validate_vectors(vectors, len(chunks), self.dimension)
        with self._lock:
            for chunk, vector in zip(chunks, checked):
                self._rows[chunk.id] = (chunk.kb_id, chunk.revision, vector)

    def search(self, kb_id: str, vector: list[float], revisions: list[str],
               limit: int) -> list[tuple[str, float]]:
        vector = validate_vectors([vector], 1, self.dimension)[0]
        accepted = set(revisions)
        with self._lock:
            rows = list(self._rows.items())
        output = []
        for chunk_id, (stored_kb, revision, stored) in rows:
            if stored_kb == kb_id and revision in accepted:
                norm = math.sqrt(sum(v * v for v in vector) * sum(v * v for v in stored))
                output.append((chunk_id, sum(x * y for x, y in zip(vector, stored)) / norm))
        return sorted(output, key=lambda x: (-x[1], x[0]))[:limit]

    def delete(self, kb_id: str, revisions: list[str]) -> None:
        with self._lock:
            self._rows = {key: row for key, row in self._rows.items()
                          if not (row[0] == kb_id and row[1] in revisions)}


class MilvusVectorStore:
    def __init__(self, uri: str, token: str = "", collection: str = "studylens_chunks",
                 dimension: int = 1024, embedding_id: str = "", *, client=None):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,100}", collection):
            raise ValueError("Milvus collection 名称格式无效。")
        if not 2 <= dimension <= 32768 or not embedding_id:
            raise ValueError("必须提供明确的向量维度和 Embedding 模型身份。")
        from pymilvus import MilvusClient, DataType
        self.dimension = dimension
        suffix = hashlib.sha256(f"{embedding_id}|{dimension}".encode()).hexdigest()[:12]
        # 不同模型使用不同集合；另有 SQLite 配置检查阻止原知识库混用向量。
        self.collection = f"{collection}_{suffix}"
        try:
            self.client = client or MilvusClient(uri=uri, token=token, timeout=20)
            if not self.client.has_collection(self.collection, timeout=20):
                schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
                schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
                schema.add_field("kb_id", DataType.VARCHAR, max_length=64)
                schema.add_field("revision", DataType.VARCHAR, max_length=32)
                schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
                index = MilvusClient.prepare_index_params()
                index.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
                self.client.create_collection(
                    collection_name=self.collection, schema=schema, index_params=index,
                    consistency_level="Strong", timeout=30,
                )
            description = self.client.describe_collection(self.collection, timeout=20)
            fields = {field["name"]: field for field in description["fields"]}
            if (int(fields["vector"]["params"]["dim"]) != dimension
                    or not {"id", "kb_id", "revision", "vector"}.issubset(fields)):
                raise ValueError("Milvus 集合结构与当前模型不一致。")
        except (ValueError, KeyError, TypeError) as exc:
            raise KnowledgeError("Milvus 集合结构不匹配，请使用新集合重新建库。") from exc
        except Exception as exc:
            raise KnowledgeError("无法初始化 Milvus，请检查地址、鉴权和服务状态。") from exc

    @staticmethod
    def _filter(kb_id: str, revisions: list[str]) -> str:
        # 所有值使用 JSON 字符串转义；不把用户输入当作 Milvus 表达式。
        return f"kb_id == {json.dumps(kb_id)} and revision in {json.dumps(revisions)}"

    def upsert(self, chunks: Sequence[IndexedChunk], vectors: list[list[float]]) -> None:
        checked = validate_vectors(vectors, len(chunks), self.dimension)
        rows = [{"id": c.id, "kb_id": c.kb_id, "revision": c.revision, "vector": v}
                for c, v in zip(chunks, checked)]
        try:
            for offset in range(0, len(rows), 128):
                batch = rows[offset:offset + 128]
                result = self.client.upsert(collection_name=self.collection, data=batch, timeout=30)
                if result.get("upsert_count") != len(batch):
                    raise KnowledgeError("Milvus 未确认全部向量写入，版本不会发布。")
        except Exception as exc:
            raise KnowledgeError("Milvus 写入失败，旧知识版本保持可用。") from exc

    def search(self, kb_id: str, vector: list[float], revisions: list[str],
               limit: int) -> list[tuple[str, float]]:
        if not revisions:
            return []
        checked = validate_vectors([vector], 1, self.dimension)
        try:
            rows = self.client.search(
                collection_name=self.collection, data=checked, anns_field="vector",
                filter=self._filter(kb_id, revisions), limit=limit,
                search_params={"metric_type": "COSINE", "params": {}},
                consistency_level="Strong", timeout=20,
            )
            return [(str(row["id"]), float(row["distance"])) for row in rows[0]]
        except Exception as exc:
            raise KnowledgeError("Milvus 检索失败，本次不会使用未经确认的替代答案。") from exc

    def delete(self, kb_id: str, revisions: list[str]) -> None:
        if not revisions:
            return
        try:
            self.client.delete(collection_name=self.collection,
                               filter=self._filter(kb_id, revisions), timeout=30)
        except Exception as exc:
            raise KnowledgeError("旧向量尚未清理；活动版本过滤已阻止其参与检索。") from exc

    def close(self) -> None:
        self.client.close()

    def check_health(self) -> bool:
        """只确认当前集合和连接可用，不调用Embedding或问答模型。"""
        try:
            return bool(self.client.has_collection(collection_name=self.collection, timeout=5))
        except Exception:
            return False
