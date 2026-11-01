"""检索工程回归；所有向量/重排替身明确只用于离线测试。"""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

HAS_RAG = all(importlib.util.find_spec(name) is not None for name in
              ("langchain_core", "langchain_text_splitters", "rank_bm25"))
if HAS_RAG:
    from langchain_core.embeddings import Embeddings
    from studylens.rag.chunking import ChunkingConfig, SemanticHierarchySplitter
    from studylens.rag.embeddings import OpenAICompatibleEmbeddings, validate_vectors
    from studylens.rag.rerank import HTTPReranker
    from studylens.rag.retriever import KnowledgeBase, reciprocal_rank_fusion, tokenize
    from studylens.rag.store import InMemoryVectorStore, MilvusVectorStore
    from studylens.rag.types import ConcurrentUpdateError, Evidence, IndexedChunk, KnowledgeError, ProviderError

    class TopicEmbeddings(Embeddings):
        """仅测试：以固定词表制造可控的语义关系，不代表真实检索质量。"""
        embedding_id = "test-topic-embeddings-v1"

        def __init__(self):
            self.calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            return [self._vector(text) for text in texts]

        def embed_query(self, text):
            return self._vector(text)

        @staticmethod
        def _vector(text):
            text = text.lower()
            fruit = any(term in text for term in ("苹果", "香蕉", "水果", "apple", "fruit", "维生素"))
            code = any(term in text for term in ("polar", "译码", "sc", "crc", "decoder"))
            return [float(fruit) + .01, float(code) + .01, .05, .01]

    class TestReranker:
        """离线控制排序，生产装配不允许使用这个类。"""
        def rerank(self, query, candidates, top_k):
            return [replace(c, score=.9 - n * .005) for n, c in enumerate(candidates[:top_k])]

    class FailingStore(InMemoryVectorStore):
        fail_upsert = False
        fail_delete = False

        def upsert(self, chunks, vectors):
            super().upsert(chunks, vectors)
            if self.fail_upsert:
                raise KnowledgeError("injected vector write failure")

        def delete(self, kb_id, revisions):
            if self.fail_delete:
                raise KnowledgeError("injected delete failure")
            super().delete(kb_id, revisions)


@unittest.skipUnless(HAS_RAG, "需安装 requirements-rag.txt 才运行 RAG 核心测试")
class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.embeddings = TopicEmbeddings()
        self.store = FailingStore(4)
        self.kb = KnowledgeBase(self.temp.name, self.embeddings, self.store, TestReranker())

    def test_same_source_upsert_is_idempotent_and_update_removes_old_revision(self):
        first = self.kb.upsert("class", "notes", "课件.md", "# 学习\n苹果是水果。")
        calls = self.embeddings.calls
        again = self.kb.upsert("class", "notes", "课件.md", "# 学习\n苹果是水果。")
        self.assertEqual(again["status"], "unchanged")
        self.assertEqual(again["revision"], first["revision"])
        self.assertEqual(self.embeddings.calls, calls)
        old_evidence = self.kb.search("class", "苹果")
        updated = self.kb.upsert("class", "notes", "课件.md", "# 学习\n香蕉是水果。")
        self.assertEqual(updated["status"], "updated")
        self.assertNotEqual(updated["revision"], first["revision"])
        hits = self.kb.search("class", "香蕉")
        self.assertTrue(hits)
        self.assertTrue(all(h.metadata["revision"] == updated["revision"] for h in hits))
        self.assertFalse(self.kb.validate_evidence("class", old_evidence))
        self.assertTrue(self.kb.validate_evidence("class", hits))

    def test_failed_vector_update_keeps_old_document_usable(self):
        initial = self.kb.upsert("class", "n", "课件.md", "# 苹果\n苹果属于水果。")
        self.store.fail_upsert = True
        with self.assertRaises(KnowledgeError):
            self.kb.upsert("class", "n", "课件.md", "# 香蕉\n香蕉属于水果。")
        self.assertEqual(self.kb.list_documents("class")[0]["revision"], initial["revision"])
        self.assertIn("苹果", self.kb.search("class", "苹果")[0].text)

    def test_logical_delete_is_immediate_even_when_physical_cleanup_fails(self):
        self.kb.upsert("class", "n", "课件.md", "# 苹果\n苹果属于水果。")
        self.store.fail_delete = True
        result = self.kb.delete("class", "n")
        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["cleanup_pending"], 1)
        self.assertEqual(self.kb.search("class", "苹果"), [])
        self.assertTrue(self.store._rows)
        self.store.fail_delete = False
        self.assertEqual(self.kb.cleanup_pending("class")["remaining"], 0)
        self.assertFalse(self.store._rows)
        self.assertEqual(self.kb.delete("class", "n")["status"], "not_found")

    def test_concurrent_publish_does_not_overwrite_newer_committed_revision(self):
        self.kb.upsert("class", "n", "n.md", "# 水果\n苹果是水果。")
        second = KnowledgeBase(self.temp.name, self.embeddings, self.store, TestReranker())
        original = self.store.upsert
        touched = []
        def competing_update(chunks, vectors):
            original(chunks, vectors)
            if not touched:
                touched.append(True)
                second.upsert("class", "n", "n.md", "# 水果\n香蕉是水果。")
        with patch.object(self.store, "upsert", side_effect=competing_update):
            with self.assertRaises(ConcurrentUpdateError):
                self.kb.upsert("class", "n", "n.md", "# 水果\n苹果和香蕉均为水果。")
        active = self.kb.search("class", "水果")
        self.assertIn("香蕉", active[0].text)
        self.assertNotIn("苹果", active[0].text)

    def test_both_routes_are_scoped_to_current_knowledge_base(self):
        self.kb.upsert("a", "same", "a.md", "# 水果\n苹果属于水果。")
        self.kb.upsert("b", "same", "b.md", "# 水果\n香蕉属于水果。")
        hits = self.kb.search("a", "水果")
        self.assertTrue(hits)
        self.assertTrue(all(h.source == "a.md" and h.metadata["kb_id"] == "a" for h in hits))
        self.assertFalse(self.kb.validate_evidence("b", hits))

    def test_vector_store_cannot_inject_cross_base_ids(self):
        self.kb.upsert("a", "x", "a.md", "苹果属于水果。")
        self.kb.upsert("b", "y", "b.md", "香蕉属于水果。")
        foreign = self.kb.search("b", "水果")[0]
        with patch.object(self.store, "search", return_value=[(foreign.id, 1.0)]):
            hits = self.kb.search("a", "水果")
        self.assertTrue(all(hit.id != foreign.id for hit in hits))

    def test_parent_context_is_returned_once_and_contains_child(self):
        config = ChunkingConfig(parent_size=800, child_size=120, semantic_unit_size=40, min_child_size=30)
        self.kb = KnowledgeBase(self.temp.name, self.embeddings, self.store, TestReranker(), chunking_config=config)
        self.kb.upsert("class", "n", "课件.md", "# 苹果\n" + "苹果是水果，含有维生素。" * 25)
        hits = self.kb.search("class", "苹果", 5)
        self.assertEqual(len(hits), 1)
        self.assertGreater(len(hits[0].text), len(hits[0].metadata["child_text"]))
        self.assertIn(hits[0].metadata["child_text"], hits[0].text)

    def test_adjacent_semantic_change_creates_child_boundary(self):
        config = ChunkingConfig(parent_size=800, child_size=600, semantic_unit_size=60,
                                min_child_size=20, semantic_similarity_threshold=.8)
        splitter = SemanticHierarchySplitter(self.embeddings, config)
        text = "# 课程\n" + "苹果是水果，香蕉也富含维生素。" * 4 + "\n\n" + "Polar 的 SC 译码逐位判决。" * 4
        chunks = splitter.split("class", "n", "a" * 32, "n.md", text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(len({c.parent_id for c in chunks}), 1)
        self.assertTrue(any("Polar" not in c.text and "苹果" in c.text for c in chunks))
        self.assertTrue(any("苹果" not in c.text and "Polar" in c.text for c in chunks))

    def test_heading_hierarchy_and_pdf_page_survive_chunking(self):
        splitter = SemanticHierarchySplitter(self.embeddings)
        chunks = splitter.split("class", "n", "b" * 32, "n.pdf", "# 第 2 页\n## 译码\nSC 是逐次消除译码。")
        relevant = next(c for c in chunks if "SC" in c.text)
        self.assertEqual(relevant.section, "译码")
        self.assertEqual(relevant.metadata["heading_path"], ["第 2 页", "译码"])
        self.assertEqual(relevant.metadata["page"], 2)

    def test_source_indentation_newlines_and_heading_hierarchy_are_preserved(self):
        splitter = SemanticHierarchySplitter(self.embeddings)
        text = "# 第 2 页\r\n\r\n## 示例\r\n    if ready:\r\n        decode()\r\n\r\n## 结果\r\n保留缩进。\r\n"
        chunks = splitter.split("class", "n", "b" * 32, "n.md", text)
        parents = dict.fromkeys((chunk.parent_id, chunk.parent_text) for chunk in chunks)
        self.assertEqual("".join(body for _, body in parents), text)
        relevant = next(chunk for chunk in chunks if "decode()" in chunk.text)
        self.assertIn("    if ready:\r\n        decode()", relevant.parent_text)
        self.assertEqual(relevant.metadata["heading_path"], ["第 2 页", "示例"])

    def test_code_and_standalone_math_are_indivisible_even_above_child_target(self):
        config = ChunkingConfig(parent_size=300, child_size=80, semantic_unit_size=20, min_child_size=10)
        splitter = SemanticHierarchySplitter(self.embeddings, config)
        code = "````python\n# 这不是标题\n" + "    value += 1\n" * 6 + "```\n````\n"
        formula = "$$\nL = " + "x + " * 28 + "y\n$$\n"
        text = "# 示例\n开始。\n" + code + "\n公式如下。\n" + formula + "结尾。\n"
        chunks = splitter.split("class", "n", "b" * 32, "n.md", text)
        for protected in (code, formula):
            containing = [chunk for chunk in chunks if protected in chunk.text]
            self.assertEqual(len(containing), 1)
            self.assertIn(protected, containing[0].parent_text)
        self.assertTrue(all(chunk.metadata["heading_path"] == ["示例"] for chunk in chunks))
        parents = dict.fromkeys((chunk.parent_id, chunk.parent_text) for chunk in chunks)
        self.assertEqual("".join(body for _, body in parents), text)

    def test_oversized_or_unclosed_protected_blocks_fail_before_embeddings(self):
        config = ChunkingConfig(parent_size=100, child_size=80, semantic_unit_size=20, min_child_size=10)
        splitter = SemanticHierarchySplitter(self.embeddings, config)
        for text in ("# 公式\n$$\n" + "x" * 101 + "\n$$\n",
                     "~~~python\n" + "x" * 101 + "\n~~~\n",
                     "# 公式\n$$\nx + y\n", "```python\nprint('x')\n"):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                splitter.split("class", "n", "b" * 32, "n.md", text)
        self.assertEqual(self.embeddings.calls, 0)

    def test_snapshot_change_during_search_retries_with_new_version(self):
        self.kb.upsert("class", "n", "n.md", "# 苹果\n苹果是水果。")
        original = self.store.search
        touched = []
        def concurrent_update(*args, **kwargs):
            rows = original(*args, **kwargs)
            if not touched:
                touched.append(True)
                self.kb.upsert("class", "n", "n.md", "# 香蕉\n香蕉是水果。")
            return rows
        with patch.object(self.store, "search", side_effect=concurrent_update):
            hits = self.kb.search("class", "水果")
        self.assertTrue(hits)
        self.assertIn("香蕉", hits[0].text)
        self.assertNotIn("苹果", hits[0].text)

    def test_model_or_dimension_change_requires_reindex(self):
        self.kb.upsert("class", "n", "n.md", "苹果是水果。")
        changed = KnowledgeBase(self.temp.name, self.embeddings, self.store, TestReranker(), embedding_id="another-model")
        with self.assertRaisesRegex(KnowledgeError, "重建"):
            changed.search("class", "苹果")

    def test_empty_query_and_missing_knowledge_do_not_call_external_services(self):
        self.kb.embeddings = Mock()
        self.kb.reranker = Mock()
        self.assertEqual(self.kb.search("empty", "苹果"), [])
        self.assertEqual(self.kb.search("empty", "???"), [])
        self.kb.embeddings.embed_query.assert_not_called()
        self.kb.reranker.rerank.assert_not_called()

    def test_input_limits_prevent_unbounded_ingestion(self):
        with self.assertRaises(ValueError):
            self.kb.upsert("../outside", "n", "n.md", "苹果")
        with self.assertRaises(ValueError):
            self.kb.upsert("class", "n", "n.md", "a" * (2 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            self.kb.search("class", "a" * 2001)
        self.assertEqual(self.embeddings.calls, 0)

    def test_rrf_rewards_agreement_without_mixing_score_scales(self):
        fused = reciprocal_rank_fusion([[('a', 1000), ('b', .1)], [('c', .9), ('b', .8)]])
        self.assertEqual(fused[0][0], 'b')
        self.assertIn("scl", tokenize("SCL 译码"))
        self.assertIn("译码", tokenize("SCL 译码"))

    def test_reranker_failure_propagates_without_silent_fallback(self):
        self.kb.upsert("class", "n", "n.md", "苹果属于水果。")
        with patch.object(self.kb.reranker, "rerank", side_effect=ProviderError("failed")):
            with self.assertRaises(ProviderError):
                self.kb.search("class", "苹果")

    def test_catalog_survives_reopening(self):
        self.kb.upsert("class", "n", "n.md", "苹果属于水果。")
        reopened = KnowledgeBase(self.temp.name, self.embeddings, self.store, TestReranker())
        self.assertEqual(len(reopened.list_documents("class")), 1)
        self.assertTrue(reopened.search("class", "苹果"))


@unittest.skipUnless(HAS_RAG, "需安装 requirements-rag.txt")
class ProviderContractTests(unittest.TestCase):
    @staticmethod
    def _embedding_response(_base, _route, _key, payload, **kwargs):
        # 只模拟HTTP返回结构；不代表实际语义模型效果。
        return {"data": [{"index": index, "embedding": [1., .1, .1, .1]}
                         for index, _ in enumerate(payload["input"])]}

    def test_embedding_budget_rejects_oversized_batch_before_network(self):
        model = OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "test",
                                           batch_size=2, max_batches=1)
        with patch("studylens.rag.embeddings.post_json") as post:
            with self.assertRaisesRegex(ProviderError, "批次预算"):
                model.embed_documents(["one", "two", "three"])
            post.assert_not_called()

    def test_nested_embedding_operations_share_budget_and_reset_for_next_request(self):
        model = OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "test",
                                           batch_size=1, max_batches=2)
        with patch("studylens.rag.embeddings.post_json", side_effect=self._embedding_response) as post:
            with model.operation_budget():
                model.embed_documents(["one"])
                with model.operation_budget():
                    model.embed_documents(["two"])
                with self.assertRaisesRegex(ProviderError, "批次预算"):
                    model.embed_documents(["three"])
            self.assertEqual(post.call_count, 2)
            model.embed_query("another request")
            self.assertEqual(post.call_count, 3)

    def test_one_import_shares_semantic_and_index_budget_and_preserves_old_version(self):
        model = OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "test",
                                           dimensions=4, batch_size=32, max_batches=2)
        with tempfile.TemporaryDirectory() as root, patch(
                "studylens.rag.embeddings.post_json", side_effect=self._embedding_response) as post:
            kb = KnowledgeBase(root, model, InMemoryVectorStore(4), TestReranker())
            original = kb.upsert("class", "note", "n.md", "# 原稿\n苹果属于水果。")
            self.assertEqual(post.call_count, 2)
            post.reset_mock()
            # 语义阶段需要两批；最终索引阶段不能另起一个预算继续计费。
            text = "# 新稿\n" + "这是需要重新建立索引的长段落，香蕉属于水果。" * 300
            with self.assertRaisesRegex(ProviderError, "批次预算"):
                kb.upsert("class", "note", "n.md", text)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(kb.list_documents("class")[0]["revision"], original["revision"])
            self.assertIn("苹果", kb.search("class", "苹果")[0].text)

    def test_elapsed_embedding_budget_stops_before_next_batch(self):
        clock = [0.]
        model = OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "test",
                                           batch_size=1, max_batches=2, max_seconds=1)
        def late_response(*args, **kwargs):
            clock[0] = 2.
            return self._embedding_response(*args, **kwargs)
        with patch("studylens.rag.embeddings.monotonic", side_effect=lambda: clock[0]), patch(
                "studylens.rag.embeddings.post_json", side_effect=late_response) as post:
            with self.assertRaisesRegex(ProviderError, "时间预算"):
                model.embed_documents(["one", "two"])
            self.assertEqual(post.call_count, 1)
            self.assertEqual(post.call_args.kwargs["deadline"], 1.)

    def test_embedding_budget_configuration_is_bounded(self):
        for value in ("0", "129", "bad"):
            with self.subTest(value=value), patch.dict("os.environ", {"STUDYLENS_EMBEDDING_MAX_BATCHES": value}):
                with self.assertRaises(ValueError):
                    OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "test")
        for value in ("0", "181", "nan"):
            with self.subTest(value=value), patch.dict("os.environ", {"STUDYLENS_EMBEDDING_MAX_SECONDS": value}):
                with self.assertRaises(ValueError):
                    OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "test")

    def test_embedding_response_respects_indices_and_dimension(self):
        model = OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "test", dimensions=2)
        with patch("studylens.rag.embeddings.post_json", return_value={"data": [
            {"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}):
            self.assertEqual(model.embed_documents(["a", "b"]), [[1., 0.], [0., 1.]])
        with patch("studylens.rag.embeddings.post_json", return_value={"data": [
            {"index": 0, "embedding": [1, 2, 3]}]}):
            with self.assertRaises(ProviderError):
                model.embed_query("a")

    def test_fixed_dimension_models_do_not_receive_unsupported_dimension_parameter(self):
        fixed = OpenAICompatibleEmbeddings("https://api.siliconflow.cn/v1", "fake", "BAAI/bge-m3", dimensions=2)
        with patch("studylens.rag.embeddings.post_json", return_value={"data": [
            {"index": 0, "embedding": [1, 0]}]}) as post:
            fixed.embed_query("苹果")
            self.assertNotIn("dimensions", post.call_args.args[3])
        configurable = OpenAICompatibleEmbeddings("https://api.openai.com/v1", "fake", "text-embedding-3-small",
                                                   dimensions=2, request_dimensions=True)
        with patch("studylens.rag.embeddings.post_json", return_value={"data": [
            {"index": 0, "embedding": [1, 0]}]}) as post:
            configurable.embed_query("苹果")
            self.assertEqual(post.call_args.args[3]["dimensions"], 2)

    def test_invalid_vectors_are_rejected(self):
        for values in [[[0., 0.]], [[float('nan'), 1]], [[True, 1]], [[1, 2], [1, 2, 3]]]:
            with self.subTest(values=values), self.assertRaises(ProviderError):
                validate_vectors(values, len(values), 2)

    def test_rerank_filters_noise_and_rejects_foreign_indices(self):
        reranker = HTTPReranker("https://api.siliconflow.cn/v1", "fake", "test", min_score=.5)
        candidates = [Evidence("a", "n", "s", "苹果", .1), Evidence("b", "n", "s", "香蕉", .2)]
        with patch("studylens.rag.rerank.post_json", return_value={"results": [
            {"index": 0, "relevance_score": .9}, {"index": 1, "relevance_score": .1}]}):
            self.assertEqual([c.id for c in reranker.rerank("苹果", candidates, 2)], ["a"])
        for rows in [[{"index": 9, "relevance_score": .9}], [{"index": 0, "relevance_score": float("nan")}]]:
            with patch("studylens.rag.rerank.post_json", return_value={"results": rows}):
                with self.assertRaises(ProviderError):
                    reranker.rerank("苹果", candidates, 2)

    def test_http_is_bounded_and_redirects_are_disabled(self):
        from studylens.rag.embeddings import post_json
        response = Mock(status_code=200)
        response.iter_content.return_value = [b'{"ok":true}']
        manager = Mock()
        manager.__enter__ = Mock(return_value=response)
        manager.__exit__ = Mock(return_value=False)
        with patch("studylens.rag.embeddings.requests.post", return_value=manager) as post:
            self.assertEqual(post_json("https://api.siliconflow.cn/v1", "/embeddings", "fake", {}), {"ok": True})
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
            self.assertTrue(post.call_args.kwargs["stream"])
            response.iter_content.return_value = [b'x' * 20]
            with self.assertRaises(ProviderError):
                post_json("https://api.siliconflow.cn/v1", "/embeddings", "fake", {}, 10)

    def test_http_respects_remaining_operation_time(self):
        from studylens.rag.embeddings import post_json
        clock = [0.]
        response = Mock(status_code=200)
        def slow_body():
            clock[0] = 2.
            yield b'{"ok":true}'
        response.iter_content.side_effect = lambda **kwargs: slow_body()
        manager = Mock()
        manager.__enter__ = Mock(return_value=response)
        manager.__exit__ = Mock(return_value=False)
        with patch("studylens.rag.embeddings.monotonic", side_effect=lambda: clock[0]), patch(
                "studylens.rag.embeddings.requests.post", return_value=manager) as post:
            with self.assertRaisesRegex(ProviderError, "时间预算"):
                post_json("https://api.siliconflow.cn/v1", "/embeddings", "fake", {}, deadline=1.)
            self.assertEqual(post.call_args.kwargs["timeout"], (1., 1.))
            manager.__exit__.assert_called_once()


@unittest.skipUnless(HAS_RAG and importlib.util.find_spec("pymilvus"), "需安装 pymilvus")
class MilvusContractTests(unittest.TestCase):
    def _store(self):
        client = Mock()
        client.has_collection.return_value = False
        client.describe_collection.return_value = {"fields": [
            {"name": "id"}, {"name": "kb_id"}, {"name": "revision"},
            {"name": "vector", "params": {"dim": 4}},
        ]}
        return MilvusVectorStore("http://unused:19530", dimension=4, embedding_id="test-v1", client=client), client

    def test_explicit_schema_and_strong_revision_filter(self):
        store, client = self._store()
        schema = client.create_collection.call_args.kwargs["schema"].to_dict()
        self.assertFalse(schema["enable_dynamic_field"])
        self.assertEqual({f["name"] for f in schema["fields"]}, {"id", "kb_id", "revision", "vector"})
        client.search.return_value = [[{"id": "a", "distance": .9}]]
        self.assertEqual(store.search("class", [1, .1, .1, .1], ["f" * 32], 3), [("a", .9)])
        kwargs = client.search.call_args.kwargs
        self.assertEqual(kwargs["consistency_level"], "Strong")
        self.assertEqual(kwargs["filter"], 'kb_id == "class" and revision in ["' + "f" * 32 + '"]')
        self.assertIn('\\"', store._filter('bad" or true', ["x"]))

    def test_partial_vector_ack_is_failure(self):
        store, client = self._store()
        chunk = IndexedChunk("a" * 64, "class", "n", "b" * 32, "p", "n", "s", "x", "x")
        client.upsert.return_value = {"upsert_count": 0}
        with self.assertRaises(KnowledgeError):
            store.upsert([chunk], [[1, .1, .1, .1]])


if __name__ == "__main__":
    unittest.main()
