"""验证输入边界、引用可信边界和网络失败；真实 API 需另行手动验收。"""
import json
import unittest
from unittest.mock import Mock, patch
import requests
from studylens.documents import load_documents, parse_document
from studylens.retrieval import Retriever
from studylens.llm import ModelConfig, ModelError, generate_answer, validate_answer


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.chunks = load_documents([("笔记.md", "# 学习\n## 苹果\n苹果是这里的检索目标。\n## 香蕉\n香蕉在第二节。\n".encode("utf-8"))])
        self.index = Retriever(self.chunks)
        self.hits = self.index.search("苹果", min_score=0)

    def test_heading_and_lines(self):
        apple = next(c for c in self.chunks if c.section == "苹果")
        self.assertEqual((apple.start_line, apple.end_line), (2, 3))
        self.assertIn("苹果是", apple.text)

    def test_long_chunk_preserves_content(self):
        text = "一二三四五六七八九十" * 80
        chunks = parse_document("a.txt", text.encode("utf-8"), "D1", 100, 20)
        rebuilt = chunks[0].text + "".join(c.text[20:] for c in chunks[1:])
        self.assertEqual(rebuilt, text)

    def test_invalid_documents(self):
        for name, data in [("a.md", b""), ("a.md", b"\xff"), ("a.pdf", b"abc"),
                           ("a.txt", b"a" * (512 * 1024 + 1))]:
            with self.subTest(name=name, size=len(data)):
                with self.assertRaises(ValueError):
                    load_documents([(name, data)])

    def test_bom_and_duplicate_names(self):
        chunks = load_documents([("a.txt", "\ufeff苹果".encode("utf-8")), ("a.txt", "香蕉".encode("utf-8"))])
        self.assertEqual(len({c.id for c in chunks}), 2)
        self.assertNotIn("\ufeff", chunks[0].text)

    def test_retrieval_and_empty_question(self):
        self.assertEqual(self.hits[0].chunk.section, "苹果")
        self.assertEqual(self.index.search(" "), [])
        self.assertEqual(self.index.search("XYZ"), [])

    def test_unknown_citation_rejected(self):
        with self.assertRaises(ModelError):
            validate_answer({"answer": "a", "citations": ["fake"], "insufficient": False}, {"D1-1"})

    def test_wrong_schema_rejected(self):
        for result in [[], {"answer": "a", "citations": "D1-1", "insufficient": False},
                       {"answer": "a", "citations": [], "insufficient": False},
                       {"answer": "a", "citations": [], "insufficient": "false"}]:
            with self.assertRaises(ModelError):
                validate_answer(result, {"D1-1"})

    def test_insufficient_overrides_unsupported_answer(self):
        result = validate_answer({"answer": "编造的答案", "citations": [], "insufficient": True}, set())
        self.assertIn("资料不足", result["answer"])

    @patch("studylens.llm.requests.post")
    def test_no_evidence_never_calls_network(self, post):
        result = generate_answer("问题", [], ModelConfig("", "", ""))
        self.assertTrue(result["insufficient"])
        post.assert_not_called()

    @patch("studylens.llm.requests.post")
    def test_mock_api_success(self, post):
        identifier = self.hits[0].chunk.id
        content = json.dumps({"answer": "苹果是目标。", "citations": [identifier], "insufficient": False})
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        answer = generate_answer("苹果", self.hits, ModelConfig("test-only", "https://example.invalid/v1", "test"))
        self.assertEqual(answer["citations"], [identifier])
        self.assertFalse(post.call_args.kwargs["allow_redirects"])
        self.assertEqual(post.call_args.args[0], "https://example.invalid/v1/chat/completions")

    @patch("studylens.llm.requests.post")
    def test_http_and_timeout_errors(self, post):
        config = ModelConfig("test-only", "https://example.invalid", "test")
        post.return_value = Mock(status_code=401)
        with self.assertRaisesRegex(ModelError, "401"):
            generate_answer("苹果", self.hits, config)
        post.side_effect = requests.Timeout()
        with self.assertRaisesRegex(ModelError, "超时"):
            generate_answer("苹果", self.hits, config)

    @patch("studylens.llm.requests.post")
    def test_malformed_and_truncated_responses(self, post):
        config = ModelConfig("test-only", "https://example.invalid", "test")
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {"choices": [{"message": {"content": "not-json"}}]}
        with self.assertRaises(ModelError):
            generate_answer("苹果", self.hits, config)
        post.return_value.json.return_value = {"choices": [{"finish_reason": "length"}]}
        with self.assertRaisesRegex(ModelError, "截断"):
            generate_answer("苹果", self.hits, config)


if __name__ == "__main__":
    unittest.main()
