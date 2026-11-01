"""服务边界测试使用明确的假服务，不消费模型额度或证明真实模型效果。"""
import unittest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from studylens.api import create_app
from studylens.service_config import Settings
from studylens.ingestion import extract_upload


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(admin_key="a" * 40, reader_key="r" * 40)
        self.kb = Mock()
        self.kb.list_documents.return_value = []
        self.kb.upsert.return_value = {"status": "created"}
        self.kb.delete.return_value = {"status": "deleted"}
        self.agent = Mock()
        self.agent.answer.return_value = {"answer": "测试回答", "citations": [], "trace": []}
        self.app = create_app(self.settings, (self.kb, self.agent))
        self.client = TestClient(self.app)
        self.reader = {"Authorization": "Bearer " + self.settings.reader_key}
        self.admin = {"Authorization": "Bearer " + self.settings.admin_key,
                      "X-External-Processing": "true"}

    def test_health_and_auth(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.get("/v1/documents").status_code, 401)
        response = self.client.get("/v1/documents", headers=self.reader)
        self.assertEqual(response.status_code, 200)
        self.assertIn("x-request-id", response.headers)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_reader_cannot_mutate(self):
        self.assertEqual(self.client.delete("/v1/documents/polar", headers=self.reader).status_code, 403)
        self.assertEqual(self.client.put("/v1/documents/polar?filename=a.md", headers=self.reader,
                                        content=b"text").status_code, 403)
        self.kb.delete.assert_not_called()
        self.kb.upsert.assert_not_called()

    def test_upload_and_delete_use_server_workspace(self):
        response = self.client.put("/v1/documents/polar", params={"filename": "note.md"},
                                   headers=self.admin, content="# SC\nSC uses one path".encode())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.kb.upsert.call_args.args[:3], ("default", "polar", "note.md"))
        self.assertEqual(self.client.delete("/v1/documents/polar", headers=self.admin).status_code, 200)
        self.kb.delete.assert_called_once_with("default", "polar")

    def test_bad_upload_and_missing_consent(self):
        response = self.client.put("/v1/documents/polar?filename=a.md", headers=self.admin, content=b"\xff")
        self.assertEqual(response.status_code, 422)
        no_consent = {"Authorization": self.admin["Authorization"]}
        self.assertEqual(self.client.put("/v1/documents/polar?filename=a.md", headers=no_consent,
                                        content=b"text").status_code, 400)
        self.kb.upsert.assert_not_called()

    def test_query_consent_length_and_scope(self):
        self.assertEqual(self.client.post("/v1/query", headers=self.reader,
                                         json={"question": "SC?"}).status_code, 400)
        self.assertEqual(self.client.post("/v1/query", headers=self.reader,
                                         json={"question": "x" * 501, "consent_external": True}).status_code, 422)
        r = self.client.post("/v1/query", headers=self.reader,
                             json={"question": "SC?", "consent_external": True, "kb_id": "other"})
        self.assertEqual(r.status_code, 200)
        self.agent.answer.assert_called_once_with("default", "SC?")

    def test_limits_stream_and_request_length(self):
        r = self.client.post("/v1/query", headers=self.reader, content=b"x" * 9000)
        self.assertEqual(r.status_code, 413)
        r = self.client.post("/v1/query", headers=self.reader, content=iter([b"x" * 5000, b"x" * 5000]))
        self.assertEqual(r.status_code, 413)
        self.agent.answer.assert_not_called()

    def test_busy_and_provider_failure_are_sanitized(self):
        self.app.state.task_lock.acquire()
        try:
            r = self.client.get("/v1/documents", headers=self.reader)
            self.assertEqual(r.status_code, 429)
        finally:
            self.app.state.task_lock.release()
        self.kb.list_documents.side_effect = RuntimeError("secret provider response")
        r = self.client.get("/v1/documents", headers=self.reader)
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("secret", r.text)

    def test_rate_limit(self):
        app = create_app(Settings(admin_key="a" * 40, reader_key="r" * 40, requests_per_minute=1),
                         (self.kb, self.agent))
        client = TestClient(app)
        self.assertEqual(client.get("/v1/documents", headers=self.reader).status_code, 200)
        self.assertEqual(client.get("/v1/documents", headers=self.reader).status_code, 429)

    def test_settings_fail_closed(self):
        with self.assertRaises(ValueError):
            Settings("weak", "weak")
        with patch.dict("os.environ", {}, clear=True):
            client = TestClient(create_app())
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(client.get("/v1/documents").status_code, 503)
        for values in ({"workspace": "_course"}, {"embedding_dimensions": 1}):
            with self.assertRaises(ValueError):
                Settings("a" * 40, "b" * 40, **values)

    def test_readiness_detects_disconnected_vector_store(self):
        self.kb.vector_store.check_health.return_value = False
        response = self.client.get("/v1/status", headers=self.reader)
        self.assertEqual(response.status_code, 503)
        self.agent.answer.assert_not_called()

    def test_original_source_name_and_invalid_paths(self):
        text, metadata = extract_upload("note.md", "# 译码\nSC保留一条路径".encode())
        self.assertIn("SC", text)
        self.assertEqual(metadata["original_name"], "note.md")
        with self.assertRaises(ValueError):
            extract_upload("../note.md", b"data")


if __name__ == "__main__":
    unittest.main()
