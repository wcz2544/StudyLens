"""Streamlit 页面交互冒烟测试，不打开浏览器、不调用真实 API。"""
import unittest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def test_search_feedback_clear_and_switch(self):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        self.assertEqual(len(app.exception), 0)
        next(r for r in app.radio if r.label == "笔记来源").set_value("使用内置示例笔记").run()
        next(i for i in app.text_input if i.label == "你想了解什么？").set_value("SC 和 SCL 译码有什么区别？")
        next(b for b in app.button if b.label == "查找并回答").click()
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.expander), 0)
        next(b for b in app.button if b.label == "记录反馈").click()
        app.run()
        self.assertEqual(len(app.session_state["feedback"]), 1)
        next(b for b in app.button if b.label == "清空本次问答与反馈").click()
        app.run()
        self.assertEqual(len(app.session_state["feedback"]), 0)
        next(r for r in app.radio if r.label == "笔记来源").set_value("上传自己的笔记").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("请上传" in item.value for item in app.info))

    def test_ai_requires_consent_and_config(self):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        next(r for r in app.radio if r.label == "笔记来源").set_value("使用内置示例笔记").run()
        next(r for r in app.radio if r.label == "运行模式").set_value("AI 问答（需要 API）").run()
        next(i for i in app.text_input if i.label == "你想了解什么？").set_value("SC 译码")
        next(b for b in app.button if b.label == "查找并回答").click()
        app.run()
        self.assertTrue(any("确认发送" in item.value for item in app.warning))
        self.assertEqual(len(app.exception), 0)
        # 同意发送但没有在侧栏填写密钥时，应显示提示且不发出网络请求。
        with patch("studylens.llm.requests.post") as post:
            next(c for c in app.checkbox if c.label.startswith("同意将问题")).check().run()
            next(b for b in app.button if b.label == "查找并回答").click()
            app.run()
            self.assertTrue(any("左侧栏填写" in item.value for item in app.error))
            self.assertEqual(len(app.exception), 0)
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
