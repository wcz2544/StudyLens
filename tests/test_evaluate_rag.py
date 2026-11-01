"""评测脚本回归；付费分支使用显式替身，只验证编排和统计，不计算真实模型效果。"""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import evaluate_rag as evaluation
from studylens.service_config import Settings


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(prefix="studylens-eval-tests-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = evaluation.read_cases(evaluation.ROOT / "evaluation/rag_cases.jsonl")
        self.case_path = self.root / "one_case.jsonl"
        self.case_path.write_text(json.dumps(self.cases[0], ensure_ascii=False) + "\n", encoding="utf-8")

    def _main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return evaluation.main(list(args))

    def _model_report(self):
        row = {
            "id": "R01", "question": "SC全称？", "expected_action": "answer",
            "rubric": "给出英文全称并引用", "retrieved": [], "recall_at_k": 1.0,
            "all_evidence_found": True, "forbidden_evidence_found": False,
            "behavior_correct_auto": True, "answer_terms_present_auto": True,
            "error": None, "model_requested": True,
            "answer": {"answer": "successive cancellation", "citations": ["e1"],
                       "claims": [{"text": "successive cancellation"}],
                       "insufficient": False, "needs_clarification": False, "error": None},
        }
        return {"run_id": "a" * 12, "backend": "production", "model_evaluation": "实际调用",
                "dataset_sha256": "b" * 64, "rows": [row], "metrics": {"top_k": 5}}

    def _labels(self, report, **changes):
        path = self.root / "labels.csv"
        evaluation.export_label_template(report, path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.update(correct="1", citations_supported="1", reviewer="测试标注员")
            row.update(changes)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=evaluation.LABEL_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_18_case_baseline_is_real_retrieval_and_never_model_accuracy(self):
        rows = evaluation.run_evaluation(self.cases, evaluation.BaselineCorpus(), run_id="test")
        result = evaluation.summarize(rows, 5)
        self.assertEqual(result["cases"], 18)
        self.assertEqual(result["answerable_cases"], 12)
        self.assertEqual(result["failed_cases"], 0)
        self.assertAlmostEqual(result["evidence_recall_at_k"], 0.8333)
        self.assertAlmostEqual(result["all_evidence_coverage"], 0.8333)
        self.assertEqual({row["id"] for row in rows if row["expected_action"] == "answer"
                          and not row["all_evidence_found"]}, {"R17", "R18"})
        self.assertIsNone(result["manual"]["answer_accuracy"])
        self.assertEqual(result["forbidden_evidence_cases"], 0)
        self.assertTrue(all(row["cleanup"] == "complete" for row in rows))

    def test_failed_answerable_case_stays_in_recall_denominator(self):
        kb = evaluation.BaselineCorpus()
        kb.search = Mock(side_effect=RuntimeError("private upstream error"))
        rows = evaluation.run_evaluation([self.cases[0]], kb)
        result = evaluation.summarize(rows, 5)
        self.assertEqual(result["answerable_cases"], 1)
        self.assertEqual(result["evidence_recall_at_k"], 0)
        self.assertEqual(result["failed_cases"], 1)
        self.assertNotIn("private", json.dumps(rows))
        self.assertEqual(kb.documents, {})

    def test_model_outage_does_not_count_as_correct_refusal(self):
        case = next(row for row in self.cases if row["expected_action"] == "refuse")
        answer = {"answer": "服务暂不可用", "insufficient": True, "citations": [],
                  "needs_clarification": False, "usage": {"search_calls": 0}, "error": "provider_failed"}
        rows = evaluation.run_evaluation([case], evaluation.BaselineCorpus(),
                                         Mock(answer=Mock(return_value=answer)))
        result = evaluation.summarize(rows, 5)
        self.assertEqual(result["behavior_accuracy_auto"], 0)
        self.assertEqual(result["model_failure_cases"], 1)

    def test_cleanup_uses_exact_current_run_and_model_suffix(self):
        base, identity, dim = "studylens_eval_123456abcdef", "fingerprinted-model-id", 4
        actual = base + "_" + hashlib.sha256(f"{identity}|{dim}".encode()).hexdigest()[:12]
        store = SimpleNamespace(collection=actual, dimension=dim, client=Mock())
        self.assertEqual(evaluation.cleanup_evaluation_collection(store, base, identity, dim), actual)
        store.client.drop_collection.assert_called_once_with(collection_name=actual, timeout=20)
        store.client.reset_mock()
        for wrong_base, wrong_id, wrong_dim in [("studylens_chunks", identity, dim),
                                               ("studylens_eval_000000000000", identity, dim),
                                               (base, "another-model", dim), (base, identity, 8)]:
            with self.subTest(base=wrong_base, identity=wrong_id, dim=wrong_dim):
                with self.assertRaises(ValueError):
                    evaluation.cleanup_evaluation_collection(store, wrong_base, wrong_id, wrong_dim)
        store.client.drop_collection.assert_not_called()

    def test_production_cli_uses_factory_fingerprint_and_isolated_state(self):
        settings = Settings(admin_key="a" * 40, reader_key="r" * 40)
        seen = {}

        def build(config):
            seen["config"] = config
            kb = evaluation.BaselineCorpus()  # 显式替身，验证CLI控制流，不输出模型成绩。
            kb.embedding_id, kb.dimension, kb.pipeline = "real-adapter-fingerprint", 1024, "chunk-version"
            suffix = hashlib.sha256(f"{kb.embedding_id}|{kb.dimension}".encode()).hexdigest()[:12]
            kb.vector_store = SimpleNamespace(collection=config.collection + "_" + suffix,
                                              dimension=1024, client=Mock(), close=Mock())
            seen["kb"] = kb
            return kb, Mock()

        output = self.root / "production_contract.json"
        with patch("studylens.service_config.Settings.from_env", return_value=settings), \
                patch("studylens.service_config.build_services", side_effect=build):
            code = self._main("--backend", "production", "--allow-paid", "--cases", str(self.case_path),
                              "--output", str(output))
        self.assertEqual(code, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["production_collection_cleanup"], "complete")
        self.assertEqual(report["model_evaluation"], "未运行")
        self.assertNotEqual(seen["config"].collection, settings.collection)
        self.assertNotEqual(seen["config"].data_dir, settings.data_dir)
        seen["kb"].vector_store.client.drop_collection.assert_called_once_with(
            collection_name=report["evaluation_collection_actual"], timeout=20)

    def test_paid_flags_are_checked_before_services_start(self):
        with patch("studylens.service_config.build_services") as build:
            for flags in [("--backend", "production"), ("--with-model",)]:
                with self.subTest(flags=flags), self.assertRaises(SystemExit) as error:
                    self._main(*flags)
                self.assertEqual(error.exception.code, 2)
            build.assert_not_called()

    def test_missing_production_configuration_is_not_a_tested_model(self):
        output = self.root / "missing.json"
        with patch.dict("os.environ", {}, clear=True):
            code = self._main("--backend", "production", "--allow-paid", "--with-model",
                              "--cases", str(self.case_path), "--output", str(output))
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertEqual(report["model_evaluation"], "未运行（启动失败）")
        self.assertIsNone(report["metrics"]["manual"]["answer_accuracy"])
        self.assertEqual(report["rows"], [])

    def test_manual_labels_bind_run_dataset_and_exact_answer(self):
        report = self._model_report()
        labels = self._labels(report)
        kwargs = {"run_id": report["run_id"], "dataset_sha256": report["dataset_sha256"]}
        metrics = evaluation.summarize(report["rows"], 5, labels, **kwargs)
        self.assertEqual(metrics["manual"]["answer_accuracy"], 1)
        self.assertEqual(metrics["manual"]["review_coverage"], 1)
        for identity in [dict(kwargs, run_id="c" * 12), dict(kwargs, dataset_sha256="d" * 64)]:
            with self.assertRaises(ValueError):
                evaluation.summarize(report["rows"], 5, labels, **identity)
        changed = copy.deepcopy(report["rows"])
        changed[0]["answer"]["answer"] = "another answer"
        with self.assertRaises(ValueError):
            evaluation.summarize(changed, 5, labels, **kwargs)

    def test_template_cannot_overwrite_labels_and_does_not_export_formulas(self):
        report = self._model_report()
        report["rows"][0]["question"] = "=1+1"
        path = self.root / "labels.csv"
        self.assertEqual(evaluation.export_label_template(report, path), 1)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["question"], "'=1+1")
        with self.assertRaises(FileExistsError):
            evaluation.export_label_template(report, path)
        report["backend"] = "baseline"
        with self.assertRaises(ValueError):
            evaluation.export_label_template(report, self.root / "invalid.csv")

    def test_citation_na_cannot_hide_factual_answer(self):
        report = self._model_report()
        labels = self._labels(report, citations_supported="N/A")
        with self.assertRaises(ValueError):
            evaluation.summarize(report["rows"], 5, labels, run_id=report["run_id"],
                                 dataset_sha256=report["dataset_sha256"])

    def test_blank_template_is_unmeasured_and_refusal_is_not_citation_success(self):
        report = self._model_report()
        row = report["rows"][0]
        row["expected_action"] = "refuse"
        row["answer"].update(answer="资料不足", insufficient=True, claims=[], citations=[])
        path = self.root / "blank.csv"
        evaluation.export_label_template(report, path)
        identity = {"run_id": report["run_id"], "dataset_sha256": report["dataset_sha256"]}
        result = evaluation.summarize(report["rows"], 5, path, **identity)
        self.assertEqual(result["manual"]["reviewed"], 0)
        self.assertIsNone(result["manual"]["answer_accuracy"])
        filled = self._labels(report, citations_supported="N/A")
        result = evaluation.summarize(report["rows"], 5, filled, **identity)
        self.assertEqual(result["manual"]["answer_accuracy"], 1)
        self.assertEqual(result["manual"]["citation_reviewed"], 0)
        self.assertIsNone(result["manual"]["citation_support_accuracy"])

    def test_grade_existing_is_offline_and_preserves_report(self):
        report = self._model_report()
        source = self.root / "original.json"
        source.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        original = source.read_bytes()
        labels = self._labels(report)
        output = self.root / "graded.json"
        with patch("studylens.service_config.build_services") as build:
            self.assertEqual(self._main("--grade-existing", str(source), "--manual-labels", str(labels),
                                        "--output", str(output)), 0)
            build.assert_not_called()
        graded = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(graded["graded_from_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(source.read_bytes(), original)
        with self.assertRaises(SystemExit):
            self._main("--grade-existing", str(source), "--manual-labels", str(labels), "--output", str(source))

    def test_output_must_not_overwrite_dataset(self):
        original = self.case_path.read_bytes()
        with self.assertRaises(SystemExit):
            self._main("--cases", str(self.case_path), "--output", str(self.case_path))
        self.assertEqual(self.case_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
