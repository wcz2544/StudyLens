"""离线检索评测，不调用大模型，不产生 API 费用。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from studylens.documents import load_documents
from studylens.retrieval import Retriever

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--output", default="evaluation/results.csv")
    parser.add_argument("--summary", help="将计数、证据覆盖率和失败案例保存为 JSON。")
    parser.add_argument("--source-commit", default="working-tree")
    parser.add_argument("--check-baseline", help="与已保存基线比较；检索覆盖回退时以非零状态退出。")
    args = parser.parse_args()
    if args.check_baseline and args.summary:
        # 比较时禁止把待比较结果写回参考文件，否则会变成结果与自身比较。
        baseline_path = (ROOT / args.check_baseline).resolve()
        if (ROOT / args.summary).resolve() == baseline_path or (ROOT / args.output).resolve() == baseline_path:
            parser.error("比较基线时，输出路径不能覆盖基线文件。")
    files = [(p.name, p.read_bytes()) for p in sorted((ROOT / "data" / "sample_notes").glob("*.md"))]
    index = Retriever(load_documents(files))
    with (ROOT / "evaluation" / "questions.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    single, single_hit, multi, multi_all, missing, empty = 0, 0, 0, 0, 0, 0
    output_rows = []
    for row in rows:
        hits = index.search(row["question"], args.top_k, args.min_score)
        found = {f"{h.chunk.source}::{h.chunk.section}" for h in hits}
        expected = {s for s in row["expected_sections"].split("|") if s}
        all_hit = bool(expected) and expected.issubset(found)
        if row["kind"] == "single":
            single += 1
            single_hit += int(all_hit)
        elif row["kind"] == "multi":
            multi += 1
            multi_all += int(all_hit)
        else:
            missing += 1
            empty += int(not hits)
        output_rows.append({
            **row, "top_k": args.top_k, "min_score": args.min_score,
            "retrieved_sections": "|".join(f"{h.chunk.source}::{h.chunk.section}" for h in hits),
            "scores": "|".join(f"{h.score:.4f}" for h in hits),
            "all_expected_found": int(all_hit) if expected else "N/A",
            "retrieval_empty": int(not hits),
            # 下列字段由用户阅读真实模型回答后填写；离线脚本不能代替回答评测。
            "answer_correct_manual": "", "citation_supported_manual": "",
            "refusal_correct_manual": "", "manual_notes": "",
        })
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"单段问题 Hit@{args.top_k}: {single_hit}/{single}")
    print(f"多段问题全部证据覆盖: {multi_all}/{multi}")
    print(f"无答案问题的空检索比例: {empty}/{missing}（不是模型拒答正确率）")
    print(f"结果写入：{output}")
    summary = {
        "source_commit": args.source_commit,
        "configuration": {"top_k": args.top_k, "min_score": args.min_score},
        "corpus_sha256": hashlib.sha256(b"".join(name.encode("utf-8") + b"\0" + data for name, data in files)).hexdigest(),
        "questions_sha256": hashlib.sha256((ROOT / "evaluation/questions.csv").read_bytes()).hexdigest(),
        "question_count": len(rows),
        "single_hit": {"passed": single_hit, "total": single, "rate": single_hit / single if single else None},
        "multi_all_evidence": {"passed": multi_all, "total": multi, "rate": multi_all / multi if multi else None},
        "unanswerable_empty_retrieval": {"passed": empty, "total": missing, "rate": empty / missing if missing else None},
        "answer_accuracy": None,
        "citation_support_accuracy": None,
        "model_refusal_accuracy": None,
        "scope": "离线检索评测；未调用模型，未测回答正确率。三份教学示例构成开发集，不代表真实业务泛化效果。",
        "failure_cases": [row for row in output_rows if row["all_expected_found"] == 0
                          or (row["kind"] == "unanswerable" and not row["retrieval_empty"])],
    }
    if args.summary:
        summary_path = Path(args.summary)
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.check_baseline:
        baseline_path = Path(args.check_baseline)
        if not baseline_path.is_absolute():
            baseline_path = ROOT / baseline_path
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        # 数据集变更需主动建立新基线，不能用换题悄悄掩盖指标回退。
        keys = ("question_count", "questions_sha256", "corpus_sha256", "configuration")
        if any(summary[key] != baseline[key] for key in keys):
            raise SystemExit("评测输入或参数与基线不一致，请审核后建立新基线。")
        for metric in ("single_hit", "multi_all_evidence", "unanswerable_empty_retrieval"):
            if summary[metric]["passed"] < baseline[metric]["passed"]:
                raise SystemExit(f"检索指标回退：{metric}")


if __name__ == "__main__":
    main()
