"""知识库入口及网络边界，不连接真实服务。"""
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from streamlit.testing.v1 import AppTest
from studylens.knowledge_ui import call_api, configured_api_url


class KnowledgeUITests(unittest.TestCase):
    def test_knowledge_mode_without_backend_is_explained(self):
        with patch("studylens.knowledge_ui.configured_api_url", return_value=""):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
            next(item for item in app.radio if item.label == "使用方式").set_value("持久知识库").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("尚未连接" in message.value for message in app.info))

    def test_unexpected_error_payload_is_still_a_user_error(self):
        with patch("studylens.knowledge_ui.requests.request") as request:
            request.return_value = Mock(status_code=503)
            request.return_value.json.return_value = ["unexpected"]
            with self.assertRaises(ValueError):
                call_api("https://api.example.com", "test", "GET", "/v1/documents")
            self.assertFalse(request.call_args.kwargs["allow_redirects"])

    def test_public_http_or_userinfo_is_rejected(self):
        for value in ("http://public.example.com", "https://user:pass@example.com", "https://example.com/?key=x"):
            with self.subTest(value=value), patch.dict("os.environ", {"STUDYLENS_API_URL": value}):
                with self.assertRaises(ValueError):
                    configured_api_url()


if __name__ == "__main__":
    unittest.main()
