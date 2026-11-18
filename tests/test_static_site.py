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

    def test_payload_contains_both_bundled_corpora(self):
        payload = self.build_site._corpora_payload()
        corpora = {item["id"]: item for item in payload["corpora"]}
        self.assertIn("sample_notes", corpora)
        self.assertIn("open_corpus", corpora)
        self.assertEqual(len(corpora["sample_notes"]["files"]), 3)
        self.assertEqual(len(corpora["open_corpus"]["files"]), 5)
        for item in corpora.values():
            self.assertTrue(item["label"])
            self.assertTrue(item["questions"], item["id"])
            for file in item["files"]:
                self.assertTrue(file["name"].endswith(".md"))
                self.assertTrue(file["text"].strip())

    def test_open_corpus_questions_come_from_the_paired_set(self):
        corpora = {item["id"]: item for item in self.build_site._corpora_payload()["corpora"]}
        questions = corpora["open_corpus"]["questions"]
        self.assertLessEqual(len(questions), 6, "示例按钮不宜过多")
        self.assertIn("LBS 的英文全称是什么？", questions)
        self.assertEqual(len(set(questions)), len(questions))

    def test_build_writes_deployable_site(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            self.build_site.build(output)
            for name in self.build_site.REQUIRED:
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue((output / ".nojekyll").is_file())
            samples = json.loads((output / "samples.json").read_text(encoding="utf-8"))
            self.assertEqual({item["id"] for item in samples["corpora"]}, {"sample_notes", "open_corpus"})
            # 页面只从 samples.json 取资料，不能把笔记目录或测试文件带进发布产物。
            self.assertFalse((output / "notes").exists())
            self.assertFalse(any(output.rglob("*.py")))

    def test_refuses_to_overwrite_repository_or_web_sources(self):
        for target in (ROOT, ROOT / "web"):
            with self.assertRaises(SystemExit):
                self.build_site._check_target(target)
        self.assertTrue((ROOT / "web" / "index.html").is_file(), "拒绝写入时不能破坏源文件")


if __name__ == "__main__":
    unittest.main()