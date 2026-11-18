"""静态站构建脚本的离线测试：内置语料、产物完整性与路径保护。"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_site.py"


def load_build_site():
    spec = importlib.util.spec_from_file_location("build_site", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build_site = load_build_site()

    def test_payload_merges_every_bundled_source(self):
        payload = self.build_site._corpora_payload()
        self.assertEqual(len(payload["corpora"]), 1, "网页只保留一个内置入口")
        corpus = payload["corpora"][0]
        self.assertEqual(corpus["label"], "内置示例笔记")
        self.assertEqual(len(corpus["files"]), 13)
        names = {file["name"] for file in corpus["files"]}
        for expected in ("01_学习方法.md", "01_术语与角色.md", "01_Rust_所有权.md", "05_Python_控制流.md"):
            self.assertIn(expected, names)
        self.assertTrue(corpus["questions"])
        self.assertEqual(len(corpus["questions"]), len(set(corpus["questions"])))

    def test_build_writes_deployable_site(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            self.build_site.build(output)
            for name in self.build_site.REQUIRED:
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue((output / ".nojekyll").is_file())
            samples = json.loads((output / "samples.json").read_text(encoding="utf-8"))
            self.assertEqual([item["id"] for item in samples["corpora"]], ["sample_notes"])
            published = samples["corpora"][0]
            self.assertEqual(len(published["files"]), 13)
            # 页面只从 samples.json 取资料，不能把笔记目录或脚本带进发布产物。
            self.assertFalse((output / "notes").exists())
            self.assertFalse(any(output.rglob("*.py")))

    def test_notes_keep_original_line_endings(self):
        """read_text 会把 CRLF 变成 LF，浏览器据此分段的字符偏移就会与 Python 不一致。"""
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "笔记.md").write_bytes("# 标题\r\n正文\r\n".encode("utf-8"))
            (Path(directory) / "README.md").write_bytes(b"# readme\r\n")
            files = self.build_site._read_notes(Path(directory))
            self.assertEqual([file["name"] for file in files], ["笔记.md"])
            self.assertIn("\r\n", files[0]["text"])

    def test_refuses_to_overwrite_repository_or_web_sources(self):
        for target in (ROOT, ROOT / "web"):
            with self.assertRaises(SystemExit):
                self.build_site._check_target(target)
        self.assertTrue((ROOT / "web" / "index.html").is_file(), "拒绝写入时不能破坏源文件")


if __name__ == "__main__":
    unittest.main()