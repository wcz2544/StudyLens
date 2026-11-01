"""真实SDK序列化、模拟HTTP传输；不调用服务商，不报告模型准确率。"""
import json
import unittest
from unittest.mock import patch
import httpx
from studylens.rag.models import LangChainChatModel


class ModelWireTests(unittest.TestCase):
    def test_provider_token_budget_reaches_wire_with_correct_field_name(self):
        for base, budget_field in [("https://api.deepseek.com", "max_tokens"),
                                   ("https://api.openai.com/v1", "max_completion_tokens")]:
            with self.subTest(base=base):
                model = LangChainChatModel(base, "test-not-real", "test-model", max_output_tokens=1500)
                captured = []
                def send(request, **kwargs):
                    captured.append(json.loads(request.content))
                    return httpx.Response(200, request=request, json={
                        "id": "test", "object": "chat.completion", "created": 1, "model": "test-model",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant", "content": '{"action":"clarify","queries":[],"clarification":"哪个？"}'}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    })
                try:
                    with patch.object(model._http, "send", side_effect=send):
                        result = model.invoke("route", {"question": "哪个更好？"})
                    self.assertEqual(result["action"], "clarify")
                    self.assertEqual(len(captured), 1)
                    self.assertEqual(captured[0][budget_field], 1500)
                    if base == "https://api.deepseek.com":
                        self.assertNotIn("max_completion_tokens", captured[0])
                        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})
                finally:
                    model.close()


if __name__ == "__main__":
    unittest.main()
