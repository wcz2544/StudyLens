"""真实 Milvus 验收；仅在显式提供 MILVUS_TEST_URI 时运行。"""
import importlib.util
import os
import tempfile
import unittest
from dataclasses import replace
from uuid import uuid4

HAS_RAG = all(importlib.util.find_spec(name) is not None for name in
              ("pymilvus", "langchain_core", "langchain_text_splitters", "rank_bm25"))


@unittest.skipUnless(HAS_RAG and os.getenv("MILVUS_TEST_URI"),
                     "未配置 MILVUS_TEST_URI，未执行真实 Milvus 集成")
class MilvusIntegrationTests(unittest.TestCase):
    def test_knowledge_base_lifecycle_with_real_milvus_and_test_models(self):
        """验证真实存储的发布/删除链路，不以替身模型宣称语义质量达标。"""
        from langchain_core.embeddings import Embeddings
        from studylens.rag.retriever import KnowledgeBase
        from studylens.rag.store import MilvusVectorStore

        class DeterministicTestEmbeddings(Embeddings):
            embedding_id = "milvus-lifecycle-test-embeddings-v1"

            def __init__(self):
                self.document_calls = 0

            def embed_documents(self, texts):
                self.document_calls += 1
                return [self.embed_query(text) for text in texts]

            def embed_query(self, text):
                # 固定、有限、非零向量，仅测试 Milvus 数据链路。
                return [1.0, (len(text) % 7 + 1) / 10, .2, .3]

        class DeterministicTestReranker:
            def rerank(self, query, candidates, top_k):
                return [replace(item, score=.9 - index * .001)
                        for index, item in enumerate(candidates[:top_k])]

        collection = "studylens_lifecycle_test_" + uuid4().hex
        embeddings = DeterministicTestEmbeddings()
        store = MilvusVectorStore(os.environ["MILVUS_TEST_URI"], os.getenv("MILVUS_TEST_TOKEN", ""),
                                  collection=collection, dimension=4, embedding_id=embeddings.embedding_id)
        reopened_store = None
        try:
            with tempfile.TemporaryDirectory(prefix="studylens-milvus-test-") as directory:
                kb = KnowledgeBase(directory, embeddings, store, DeterministicTestReranker())
                original_text = "# Storage policy\nQuota is 5GB. The old version marker is old_v1."
                first = kb.upsert("workspace", "policy", "policy.md", original_text)
                calls = embeddings.document_calls
                unchanged = kb.upsert("workspace", "policy", "policy.md", original_text)
                self.assertEqual(unchanged["status"], "unchanged")
                self.assertEqual(unchanged["revision"], first["revision"])
                self.assertEqual(embeddings.document_calls, calls)
                old_evidence = kb.search("workspace", "Storage quota")
                self.assertTrue(old_evidence)
                self.assertIn("5GB", old_evidence[0].text)

                updated = kb.upsert("workspace", "policy", "policy.md",
                                    "# Storage policy\nQuota is 10GB. The new version marker is new_v2.")
                self.assertEqual(updated["status"], "updated")
                hits = kb.search("workspace", "Storage quota")
                self.assertTrue(hits)
                self.assertTrue(all(hit.metadata["revision"] == updated["revision"] for hit in hits))
                self.assertTrue(all("old_v1" not in hit.text and "10GB" in hit.text for hit in hits))
                self.assertFalse(kb.validate_evidence("workspace", old_evidence))
                self.assertTrue(kb.validate_evidence("workspace", hits))

                # 重新构造 SQLite 目录与 Milvus 客户端，不依赖原实例的 BM25 缓存。
                reopened_store = MilvusVectorStore(
                    os.environ["MILVUS_TEST_URI"], os.getenv("MILVUS_TEST_TOKEN", ""),
                    collection=collection, dimension=4, embedding_id=embeddings.embedding_id,
                )
                reopened = KnowledgeBase(directory, embeddings, reopened_store, DeterministicTestReranker())
                self.assertEqual(reopened.list_documents("workspace"), kb.list_documents("workspace"))
                self.assertEqual(reopened.search("workspace", "Storage quota")[0].metadata["revision"],
                                 updated["revision"])
                self.assertEqual(reopened.delete("workspace", "policy")["status"], "deleted")
                self.assertEqual(kb.search("workspace", "Storage quota"), [])
                self.assertEqual(reopened.list_documents("workspace"), [])
                self.assertFalse(reopened.validate_evidence("workspace", hits))
                self.assertEqual(reopened.delete("workspace", "policy")["status"], "not_found")
        finally:
            try:
                # 随机测试集合是本测试唯一删除目标。
                store.client.drop_collection(store.collection)
            finally:
                if reopened_store is not None:
                    reopened_store.close()
                store.close()

    def test_scoped_search_update_and_delete(self):
        from studylens.rag.store import MilvusVectorStore
        from studylens.rag.types import IndexedChunk
        collection = "studylens_test_" + uuid4().hex
        store = MilvusVectorStore(os.environ["MILVUS_TEST_URI"], os.getenv("MILVUS_TEST_TOKEN", ""),
                                  collection=collection, dimension=4, embedding_id="test-v1")
        try:
            first = IndexedChunk("a" * 64, "class", "n", "1" * 32, "p", "n.md", "s", "苹果", "苹果")
            other = IndexedChunk("b" * 64, "other", "n", "2" * 32, "p", "n.md", "s", "香蕉", "香蕉")
            updated = IndexedChunk("c" * 64, "class", "n", "3" * 32, "p", "n.md", "s", "新版", "新版")
            store.upsert([first, other, updated], [[1., .1, .1, .1]] * 3)
            results = store.search("class", [1., .1, .1, .1], [first.revision], 5)
            self.assertEqual([row[0] for row in results], [first.id])
            current = store.search("class", [1., .1, .1, .1], [updated.revision], 5)
            self.assertEqual([row[0] for row in current], [updated.id])
            store.delete("class", [updated.revision])
            self.assertEqual(store.search("class", [1., .1, .1, .1], [updated.revision], 5), [])
        finally:
            # 只删除本测试刚创建的随机集合，不接触部署配置的业务集合。
            store.client.drop_collection(store.collection)
            store.close()


if __name__ == "__main__":
    unittest.main()
