"""公开上传边界与网络出口的回归测试；不发出实际网络请求。"""
import unittest
from unittest.mock import patch

from studylens.documents import load_documents
from studylens.limits import validate_extracted_text, validate_uploads
from studylens.llm import ModelConfig, ModelError, generate_answer
from studylens.retrieval import Retriever
from studylens.security import validate_model_base_url


class SecurityTests(unittest.TestCase):
    def test_only_exact_approved_endpoints_are_accepted(self):
        self.assertEqual(validate_model_base_url("https://API.DEEPSEEK.COM/"), "https://api.deepseek.com")
        invalid = [
            "http://api.deepseek.com", "https://127.0.0.1", "https://[::1]",
            "https://localhost", "https://service.internal", "https://169.254.169.254",
            "https://user:secret@api.deepseek.com", "https://api.deepseek.com:443",
            "https://api.deepseek.com.evil.com", "https://evil.com/api.deepseek.com",
            "https://api.deepseek.com/chat/completions", "https://api.deepseek.com/?x=1",
            "https://api.deepseek.com/#x", "https://api.deepseek.com/%2e%2e",
            "https://api.deepseek.com\\@evil.com", "https://api.deepseek.com\n.evil.com",
            "https://api.openai.com/v1/../private", "https://api.deepseek.com.",
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_model_base_url(value)

    def test_only_deployment_environment_can_extend_allowlist(self):
        url = "https://models.example.com/v1"
        with self.assertRaises(ValueError):
            validate_model_base_url(url)
        with patch.dict("os.environ", {"STUDYLENS_ALLOWED_MODEL_BASE_URLS": url}):
            self.assertEqual(validate_model_base_url(url), url)
            with self.assertRaises(ValueError):
                validate_model_base_url("https://models.example.com/other")
        with patch.dict("os.environ", {"STUDYLENS_ALLOWED_MODEL_BASE_URLS": "https://127.0.0.1"}):
            with self.assertRaises(ValueError):
                validate_model_base_url("https://127.0.0.1")

    @patch("studylens.llm.requests.post")
    def test_rejected_endpoint_never_reaches_http_client(self, post):
        hits = Retriever(load_documents([("note.txt", b"apple")])).search("apple")
        with self.assertRaises(ModelError):
            generate_answer("apple", hits, ModelConfig("fake", "https://127.0.0.1", "fake"))
        post.assert_not_called()

    def test_upload_type_count_and_size_limits(self):
        validate_uploads([("photo.jpg", b"x" * (3 * 1024 * 1024))])
        validate_uploads([("book.pdf", b"x" * (12 * 1024 * 1024))])
        for files in [[], [("n.txt", b"x")] * 11, [("a.exe", b"x")],
                      [("a.txt", b"x" * (512 * 1024 + 1))],
                      [("a.png", b"x" * (8 * 1024 * 1024 + 1))],
                      [("a.pdf", b"x" * (12 * 1024 * 1024 + 1))],
                      [("a.pdf", b"x" * (9 * 1024 * 1024))] * 3]:
            with self.subTest(count=len(files)), self.assertRaises(ValueError):
                validate_uploads(files)

    def test_text_extraction_has_a_separate_budget(self):
        with self.assertRaisesRegex(ValueError, "提取后的文字"):
            validate_extracted_text([("image.ocr.md", b"x" * (512 * 1024 + 1))])
        with self.assertRaisesRegex(ValueError, "2 MiB"):
            validate_extracted_text([("p.txt", b"x" * (512 * 1024))] * 5)


if __name__ == "__main__":
    unittest.main()
