"""自建 RAG 开发集评测；默认离线，不把模拟模型测试算作真实准确率。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from studylens.documents import load_documents
from studylens.retrieval import Retriever
from studylens.rag.types import Evidence


ROOT = Path(__file__).resolve().parent
LABEL_FIELDS = ("run_id", "dataset_sha256", "id", "answer_sha256", "question",
                "answer_text", "correct", "citations_supported", "reviewer", "notes")


class BaselineCorpus:
    """真实 TF-IDF 检索适配；更新后重建内存索引，不伪称持久化混合检索。"""
    def __init__(self):
        self.documents = {}

    def upsert(self, kb_id, source_id, name, text, metadata=None):
        self.documents[(kb_id, source_id)] = (name, text)

    def delete(self, kb_id, source_id):
        self.documents.pop((kb_id, source_id), None)

    def search(self, kb_id, query, top_k=5):
        docs = [(key[1], *value) for key, value in self.documents.items() if key[0] == kb_id]
        if not docs:
            return []
        index = Retriever(load_documents([(name, text.encode("utf-8")) for _, name, text in docs]))
        sources = {name: source_id for source_id, name, _ in docs}
        return [Evidence(hit.chunk.id, hit.chunk.source, hit.chunk.section, hit.chunk.text, hit.score,
                         {"kb_id": kb_id, "source_id": sources[hit.chunk.source], "revision": "baseline"})
                for hit in index.search(query, top_k=top_k, min_score=0.08)]


def read_cases(path: Path) -> list[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("评测集为空或有重复题号。")
    for case in cases:
        if case["expected_action"] not in {"answer", "refuse", "clarify", "smalltalk"}:
            raise ValueError("评测集存在未知预期行为。")
    return cases


def _ratio(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def cleanup_evaluation_collection(store, base_collection: str, embedding_id: str, dimension: int) -> str:
    """核对随机评测前缀及模型后缀，只清理这一轮创建的实际集合。"""
    if not re.fullmatch(r"studylens_eval_[0-9a-f]{12}", base_collection):
        raise ValueError("拒绝清理非本轮评测集合。")
    if not isinstance(embedding_id, str) or not embedding_id or dimension != store.dimension:
        raise ValueError("评测集合的模型身份或维度不匹配。")
    suffix = hashlib.sha256(f"{embedding_id}|{dimension}".encode()).hexdigest()[:12]
    expected = f"{base_collection}_{suffix}"
    if store.collection != expected:
        raise ValueError("评测集合的实际名称与本轮配置不匹配。")
    store.client.drop_collection(collection_name=expected, timeout=20)
    return expected


def answer_fingerprint(row: dict) -> str:
    """把标签绑定到本题实际问题、输出和依据，防止把旧答案评分套到新答案上。"""
    reviewed = {key: row.get(key) for key in
                ("id", "question", "expected_action", "rubric", "answer", "retrieved")}
    content = json.dumps(reviewed, ensure_ascii=False, sort_keys=True, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def validate_grading_report(report: dict) -> None:
    if (not isinstance(report, dict) or report.get("backend") != "production"
            or report.get("model_evaluation") != "实际调用"
            or not re.fullmatch(r"[0-9a-f]{12}", str(report.get("run_id", "")))
            or not re.fullmatch(r"[0-9a-f]{64}", str(report.get("dataset_sha256", "")))
            or not isinstance(report.get("rows"), list)):
        raise ValueError("人工评分需要保留运行编号和题集指纹的真实问答报告；离线检索结果不能评分。")
    if not any(isinstance(row.get("answer"), dict) for row in report["rows"]):
        raise ValueError("本次报告没有可供人工阅读的问答结果，准确率仍未测。")


def export_label_template(report: dict, output: Path) -> int:
    validate_grading_report(report)
    rows = [row for row in report["rows"] if isinstance(row.get("answer"), dict)]
    output.parent.mkdir(parents=True, exist_ok=True)
    # 用排他创建保留已填写标签；不得悄悄覆盖人工劳动。
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "run_id": report["run_id"], "dataset_sha256": report["dataset_sha256"],
                "id": row["id"], "answer_sha256": answer_fingerprint(row),
                "question": _csv_display(row["question"]),
                "answer_text": _csv_display(row["answer"].get("answer", "")),
                "correct": "", "citations_supported": "", "reviewer": "", "notes": "",
            })
    return len(rows)


def _csv_display(value: str) -> str:
    # CSV常被Excel打开，用户问题/模型输出不能作为电子表格公式执行。
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def _manual_labels(path: Path | None, rows: list[dict], *,
                   run_id: str | None = None, dataset_sha256: str | None = None) -> dict:
    result = {"answer_accuracy": None, "citation_support_accuracy": None,
              "reviewed": 0, "citation_reviewed": 0,
              "model_cases": sum(row["answer"] is not None for row in rows),
              "requested_cases": sum(row.get("model_requested", False) for row in rows),
              "correct_count": 0, "review_coverage": None,
              "status": "未标注，不能报告真实回答准确率"}
    if path is None:
        return result
    if not run_id or not dataset_sha256:
        raise ValueError("人工评分必须指定本轮运行编号和题集指纹。")
    available = {row["id"]: row for row in rows if isinstance(row.get("answer"), dict)}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    seen, reviewed, correct, supported = set(), set(), [], []
    for label in labels:
        case_id = label.get("id")
        if case_id not in available or case_id in seen:
            raise ValueError("人工标注题号重复、未知或该题没有真实模型结果。")
        if (label.get("run_id") != run_id or label.get("dataset_sha256") != dataset_sha256
                or label.get("answer_sha256") != answer_fingerprint(available[case_id])):
            raise ValueError("标签来自另一轮运行、另一题集或另一份答案，已拒绝混用。")
        seen.add(case_id)
        # 模板允许分批评分；完全空白的评分栏不进入分母，半填写行则必须纠正。
        if all(not label.get(key, "").strip() for key in ("correct", "citations_supported", "reviewer")):
            continue
        if not label.get("reviewer", "").strip():
            raise ValueError("人工标注需要填写 reviewer。")
        reviewed.add(case_id)
        if label.get("correct") not in {"0", "1"}:
            raise ValueError("人工标注 correct 只能填写0或1。")
        correct.append(int(label["correct"]))
        answer = available[case_id]["answer"]
        # 正确拒答/澄清/闲聊没有待核查的事实引用，不进入引用正确率分母。
        if label.get("citations_supported") == "N/A":
            has_claims = bool(answer.get("citations") or answer.get("claims"))
            if has_claims or (not answer.get("insufficient") and
                              available[case_id]["expected_action"] != "smalltalk"):
                raise ValueError("有事实回答或引用时，citations_supported 必须填写0或1。")
        elif label.get("citations_supported") in {"0", "1"}:
            supported.append(int(label["citations_supported"]))
        else:
            raise ValueError("citations_supported 请填0、1，或无事实引用时的 N/A。")
    result.update(answer_accuracy=_ratio(correct), citation_support_accuracy=_ratio(supported),
                  reviewed=len(reviewed), citation_reviewed=len(supported), correct_count=sum(correct),
                  review_coverage=round(len(reviewed) / len(available), 4) if available else None,
                  status=("人工标注，仅代表已评分样本；必须同时报告覆盖题数、失败题数和请求题数"
                          if reviewed else "未填写任何评分，准确率仍未测"))
    return result


def run_evaluation(cases: list[dict], kb, agent=None, top_k: int = 5, run_id: str | None = None) -> list[dict]:
    """每题使用独立工作区，执行真实 upsert/delete 后检查最终可见内容。"""
    run_id = run_id or uuid4().hex[:12]
    output = []
    for index, case in enumerate(cases):
        kb_id = f"eval_{run_id}_{index}"
        started = perf_counter()
        row = {"id": case["id"], "category": case["category"], "question": case["question"],
               "expected_action": case["expected_action"], "rubric": case["rubric"],
               "recall_at_k": None, "all_evidence_found": None, "forbidden_evidence_found": False,
               "retrieval_empty": None, "retrieved": [], "answer": None,
               "model_requested": agent is not None,
               "behavior_correct_auto": False if agent is not None else None, "answer_terms_present_auto": None,
               "error": None, "cleanup": "pending"}
        source_ids = set()
        try:
            for document in case["documents"]:
                source_ids.add(document["source_id"])
                kb.upsert(kb_id, **document)
            for operation in case["operations"]:
                fields = {key: value for key, value in operation.items() if key != "action"}
                source_ids.add(operation["source_id"])
                if operation["action"] == "delete":
                    kb.delete(kb_id, operation["source_id"])
                elif operation["action"] == "upsert":
                    kb.upsert(kb_id, **fields)
                else:
                    raise ValueError("不支持的题集操作。")
            # 检索指标采用原问题直接搜索，独立于 Agent 改写策略，便于横向对比。
            hits = kb.search(kb_id, case["question"], top_k=top_k)
            row["retrieved"] = [{"id": hit.id, "source_id": hit.metadata.get("source_id"),
                                 "section": hit.section, "text": hit.text, "score": hit.score}
                                for hit in hits]
            expected = case["expected_evidence"]
            if expected:
                found = [any(hit.metadata.get("source_id") == item["source_id"]
                             and item["quote"] in hit.text for hit in hits) for item in expected]
                row["recall_at_k"] = sum(found) / len(found)
                row["all_evidence_found"] = all(found)
            row["retrieval_empty"] = not hits
            row["forbidden_evidence_found"] = any(
                forbidden in hit.text for forbidden in case.get("forbidden_evidence", []) for hit in hits)
            if agent is not None:
                answer = agent.answer(kb_id, case["question"])
                row["answer"] = answer
                behavior = case["expected_action"]
                row["behavior_correct_auto"] = {
                    "answer": not answer["insufficient"] and bool(answer["citations"]),
                    "refuse": answer["insufficient"] and not answer["citations"],
                    "clarify": answer["needs_clarification"],
                    "smalltalk": (not answer["insufficient"] and not answer["citations"]
                                  and answer["usage"]["search_calls"] == 0),
                }[behavior]
                # 模型掉线不等于正确拒答，不能因问题恰好无答案而算作成功。
                if answer.get("error") not in {None, "no_evidence", "insufficient_evidence", "unsupported_answer"}:
                    row["behavior_correct_auto"] = False
                terms = case.get("expected_answer_terms", [])
                if terms:
                    row["answer_terms_present_auto"] = all(
                        term.casefold() in answer["answer"].casefold() for term in terms)
        except Exception as exc:
            # 仅记录错误类别，不能把第三方原始错误、密钥和内部路径写进评测产物。
            row["error"] = type(exc).__name__
        finally:
            cleanup_ok = True
            for source_id in source_ids:
                try:
                    kb.delete(kb_id, source_id)
                except Exception:
                    cleanup_ok = False
            row["cleanup"] = "complete" if cleanup_ok else "failed"
            row["elapsed_seconds"] = round(perf_counter() - started, 4)
        output.append(row)
    return output


def summarize(rows: list[dict], top_k: int, manual: Path | None = None, *,
              run_id: str | None = None, dataset_sha256: str | None = None) -> dict:
    expected_answer = [row for row in rows if row["expected_action"] == "answer"]
    # 失败案例按0计入召回，不能通过跳过失败提高成绩。
    return {
        "cases": len(rows), "failed_cases": sum(row["error"] is not None for row in rows),
        "model_failure_cases": sum(bool(row["answer"]) and row["answer"].get("error") not in
                                   {None, "no_evidence", "insufficient_evidence", "unsupported_answer"}
                                   for row in rows),
        "top_k": top_k,
        "answerable_cases": len(expected_answer),
        "evidence_recall_at_k": _ratio([row["recall_at_k"] or 0 for row in expected_answer]),
        "all_evidence_coverage": _ratio([int(bool(row["all_evidence_found"])) for row in expected_answer]),
        "forbidden_evidence_cases": sum(row["forbidden_evidence_found"] for row in rows),
        "behavior_accuracy_auto": _ratio([int(row["behavior_correct_auto"]) for row in rows
                                            if row["behavior_correct_auto"] is not None]),
        "answer_terms_presence_auto": _ratio([int(row["answer_terms_present_auto"]) for row in rows
                                                if row["answer_terms_present_auto"] is not None]),
        "manual": _manual_labels(manual, rows, run_id=run_id, dataset_sha256=dataset_sha256),
        "warning": "行为/关键词自动指标不等于语义正确率；自建题集是开发集，不是独立盲测集。",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["baseline", "production"], default="baseline")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/rag_cases.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-paid", action="store_true", help="明确允许生产Embedding与Rerank调用费用")
    parser.add_argument("--with-model", action="store_true", help="进一步运行真实Agent模型问答并产生费用")
    parser.add_argument("--manual-labels", type=Path)
    parser.add_argument("--grade-existing", type=Path, help="仅汇总已有结果的人工评分，不重跑模型")
    parser.add_argument("--export-label-template", type=Path, help="从已有真实问答报告导出与该轮绑定的人工评分CSV")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/generated/rag_evaluation.json")
    args = parser.parse_args(argv)
    if args.grade_existing:
        if (bool(args.manual_labels) == bool(args.export_label_template)
                or args.with_model or args.allow_paid):
            parser.error("--grade-existing 需二选一：--manual-labels 或 --export-label-template；不能同时请求收费调用。")
        source_bytes = args.grade_existing.read_bytes()
        report = json.loads(source_bytes.decode("utf-8"))
        validate_grading_report(report)
        if args.export_label_template:
            if args.export_label_template.resolve() == args.grade_existing.resolve():
                parser.error("评分模板不能覆盖原始评测结果。")
            count = export_label_template(report, args.export_label_template)
            print(json.dumps({"template": str(args.export_label_template), "cases": count,
                              "run_id": report["run_id"]}, ensure_ascii=False, indent=2))
            return 0
        if args.output.resolve() in {args.grade_existing.resolve(), args.manual_labels.resolve()}:
            parser.error("评分输出请使用新文件，保留原始结果和人工标签。")
        report["metrics"] = summarize(report["rows"], report["metrics"]["top_k"], args.manual_labels,
                                      run_id=report["run_id"], dataset_sha256=report["dataset_sha256"])
        report["graded_from"] = str(args.grade_existing)
        report["graded_from_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        report["labels_sha256"] = hashlib.sha256(args.manual_labels.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["metrics"]["manual"], ensure_ascii=False, indent=2))
        return 0
    if args.manual_labels or args.export_label_template:
        parser.error("请使用 --grade-existing 读取已有答案，避免评分时再次调用模型。")
    if args.output.resolve() == args.cases.resolve():
        parser.error("评测输出不能覆盖题集。")
    if not 1 <= args.top_k <= 8:
        parser.error("top-k 必须在1到8之间。")
    if args.backend == "production" and not args.allow_paid:
        parser.error("生产检索会调用Embedding和Rerank，请明确添加 --allow-paid。")
    if args.with_model and args.backend != "production":
        parser.error("真实模型评测请使用 --backend production --allow-paid --with-model。")
    cases = read_cases(args.cases)
    run_id = uuid4().hex[:12]
    report = {
        "schema_version": 2,
        "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "model_evaluation": "实际调用" if args.with_model else "未运行",
        "dataset_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "dataset_role": "synthetic_development_set_not_independent_test",
        "production_collection_cleanup": "not_applicable", "rows": [],
        "startup_error": None,
    }
    if args.backend == "baseline":
        report["retrieval_config"] = {"method": "char_tfidf", "min_score": 0.08, "top_k": args.top_k}
        report["rows"] = run_evaluation(cases, BaselineCorpus(), top_k=args.top_k, run_id=run_id)
    else:
        from studylens.service_config import Settings, build_services
        try:
            settings = Settings.from_env()
            collection = f"studylens_eval_{run_id}"
            if collection == settings.collection:
                raise ValueError("评测集合不得与正式集合相同。")
            # 独立本地目录 + 独立Milvus集合 + 每题独立kb，避免改变现有正式知识库。
            with TemporaryDirectory(prefix="studylens-eval-") as temp:
                kb = agent = None
                report["evaluation_collection"] = collection
                report["models"] = {"embedding": settings.embedding_model,
                                    "rerank": settings.rerank_model,
                                    "chat": settings.chat_model if args.with_model else None}
                try:
                    kb, agent = build_services(replace(settings, data_dir=temp, collection=collection))
                    report["evaluation_collection_actual"] = kb.vector_store.collection
                    report["retrieval_config"] = {
                        "method": "bm25_vector_rrf_rerank_parent_context", "top_k": args.top_k,
                        "embedding_id": kb.embedding_id, "dimension": kb.dimension,
                        "chunking_fingerprint": kb.pipeline,
                    }
                    agent.top_k = args.top_k
                    report["rows"] = run_evaluation(cases, kb, agent if args.with_model else None,
                                                     top_k=args.top_k, run_id=run_id)
                finally:
                    if agent is not None:
                        try:
                            agent.model.close()
                        except Exception:
                            report["model_client_cleanup"] = "failed"
                    if kb is not None:
                        try:
                            # 工厂使用Embedding适配器的模型指纹；不能另拼原始字符串猜集合后缀。
                            cleanup_evaluation_collection(kb.vector_store, collection,
                                                          kb.embedding_id, kb.dimension)
                            report["production_collection_cleanup"] = "complete"
                        except Exception:
                            report["production_collection_cleanup"] = "failed; inspect evaluation_collection"
                        finally:
                            kb.vector_store.close()
                    else:
                        report["production_collection_cleanup"] = "not_verified; setup_failed"
        except Exception as exc:
            report["startup_error"] = type(exc).__name__
            if not report["rows"]:
                report["model_evaluation"] = "未运行（启动失败）"
    report["metrics"] = summarize(report["rows"], args.top_k, args.manual_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "backend": args.backend,
                      "startup_error": report["startup_error"], "metrics": report["metrics"]},
                     ensure_ascii=False, indent=2))
    cleanup_failed = any(row["cleanup"] != "complete" for row in report["rows"])
    return int(bool(report["startup_error"] or report["metrics"]["failed_cases"] or cleanup_failed
                    or report["production_collection_cleanup"].startswith("failed")))


if __name__ == "__main__":
    raise SystemExit(main())
