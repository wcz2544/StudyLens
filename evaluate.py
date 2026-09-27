"""离线检索评测，不调用大模型，不产生 API 费用。"""
import argparse
import csv
from pathlib import Path
from studylens.documents import load_documents
from studylens.retrieval import Retriever

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--output", default="evaluation/results.csv")
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
