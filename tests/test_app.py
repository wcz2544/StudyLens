"""Streamlit 页面交互冒烟测试，不打开浏览器、不调用真实 API。"""
import unittest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppTests(unittest.TestCase):
    def test_failed_upload_can_return_to_last_valid_document(self):
        class Uploaded:
            name = "笔记.md"

            def __init__(self, content):
                self.content = content

            def getvalue(self):
                return self.content

        valid = Uploaded("# 测试\n苹果是有效资料。".encode("utf-8"))
        with patch("streamlit.file_uploader", return_value=[valid]) as uploader:
            app = AppTest.from_file(str(APP)).run(timeout=30)
            original_hash = app.session_state["corpus_hash"]
            self.assertEqual(len(app.exception), 0)
            uploader.return_value = [Uploaded(b"\xff")]
            app.run(timeout=30)
            self.assertTrue(any("UTF-8" in error.value for error in app.error))
            uploader.return_value = [valid]
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["corpus_hash"], original_hash)
            self.assertIn("index", app.session_state)

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

    def test_open_corpus_option_loads_bundled_notes(self):
        app = AppTest.from_file(str(APP)).run(timeout=30)
        source = next(r for r in app.radio if r.label == "笔记来源")
        self.assertIn("使用开放虚构语料", source.options)
        source.set_value("使用开放虚构语料").run()
        self.assertEqual(len(app.exception), 0)
        captions = " ".join(item.value for item in app.caption)
        self.assertIn("5 份笔记", captions)
        self.assertTrue(any("灯塔调度导论" in item.value for item in app.caption))
        # 内置语料不需要上传控件，避免用户以为必须再选文件。
        self.assertTrue(next(f for f in app.get("file_uploader") if f.label == "上传自己的笔记").disabled)

        next(i for i in app.text_input if i.label == "你想了解什么？").set_value("LBS 的英文全称是什么？")
        next(b for b in app.button if b.label == "查找并回答").click()
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.expander), 0)
        self.assertIn("01_术语与角色.md", app.expander[0].label)

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
