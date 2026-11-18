"""对网页内置的全部语料做一次离线检索评测（不调用模型，不产生费用）。

网页上的“内置示例笔记”把三份来源合成一个索引，因此这里也按同一方式建库：

    data/sample_notes       30 题开发集（evaluation/questions.csv）
    data/open_corpus/notes  36 题配对集（data/open_corpus/questions.csv）
    data/public_notes       31 题公开资料集（evaluation/public_notes_questions.csv）

    python scripts/evaluate_bundled.py --write-baseline
    python scripts/evaluate_bundled.py --check-baseline evaluation/bundled_baseline.json

指标口径与 evaluate.py 一致：单段题看命中，多段题看全部证据覆盖，无答案题只看
“检索是否为空”。它不是回答正确率，也没有经过真实模型调用。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studylens.documents import load_documents  # noqa: E402
from studylens.retrieval import Retriever  # noqa: E402

NOTE_DIRS = (
    ROOT / "data" / "sample_notes",
    ROOT / "data" / "open_corpus" / "notes",
    ROOT / "data" / "public_notes",
)
QUESTION_SETS = (
    ("sample_notes", ROOT / "evaluation" / "questions.csv"),
    ("open_corpus", ROOT / "data" / "open_corpus" / "questions.csv"),
    ("public_notes", ROOT / "evaluation" / "public_notes_questions.csv"),
)
DEFAULT_SUMMARY = ROOT / "evaluation" / "generated" / "bundled_eval.json"
DEFAULT_CSV = ROOT / "evaluation" / "generated" / "bundled_eval.csv"


def load_bundled_notes() -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for directory in NOTE_DIRS:
        if not directory.is_dir():
            raise SystemExit(f"找不到语料目录：{directory.relative_to(ROOT)}")
        files.extend(
            (path.name, path.read_bytes())
            for path in sorted(directory.glob("*.md"))
            if path.name.lower() != "readme.md"
        )
    if not files:
        raise SystemExit("内置语料为空，无法评测。")
    return files


def read_questions(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"找不到题目文件：{path.relative_to(ROOT)}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def evaluate(index: Retriever, rows: list[dict], top_k: int, min_score: float) -> dict:
    counters = {name: {"passed": 0, "total": 0} for name in ("single", "multi", "unanswerable")}
    details = []
    for row in rows:
        hits = index.search(row["question"], top_k, min_score)
        found = {f"{hit.chunk.source}::{hit.chunk.section}" for hit in hits}
        expected = {item for item in row.get("expected_sections", "").split("|") if item}
        kind = row.get("kind") or "single"
        if kind == "unanswerable":
            passed = not hits
        else:
            passed = bool(expected) and expected.issubset(found)
        bucket = counters.get(kind, counters["single"])
        bucket["total"] += 1
        bucket["passed"] += int(passed)
        details.append({
            "id": row.get("id", ""),
            "kind": kind,
            "question": row["question"],
            "expected_sections": row.get("expected_sections", ""),
            "retrieved": [f"{hit.chunk.source}::{hit.chunk.section}" for hit in hits],
            "scores": [round(hit.score, 4) for hit in hits],
            "passed": int(passed),
        })
    return {"counts": counters, "questions": details}


def summarise(report: dict) -> dict:
    summary: dict = {}
    for name, section in report.items():
        summary[name] = {
            metric: {
                "passed": values["passed"],
                "total": values["total"],
                "rate": (values["passed"] / values["total"]) if values["total"] else None,
            }
            for metric, values in section["counts"].items()
        }
        summary[name]["question_count"] = sum(values["total"] for values in section["counts"].values())
        summary[name]["failure_ids"] = [
            item["id"] for item in section["questions"] if not item["passed"]
        ]
    total = {
        metric: {
            "passed": sum(report[name]["counts"][metric]["passed"] for name in report),
            "total": sum(report[name]["counts"][metric]["total"] for name in report),
        }
        for metric in ("single", "multi", "unanswerable")
    }
    for metric, values in total.items():
        values["rate"] = (values["passed"] / values["total"]) if values["total"] else None
    summary["overall"] = {
        **total,
        "question_count": sum(values["total"] for values in total.values()),
    }
    return summary


def digest(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        hasher.update(path.name.encode("utf-8") + b"\0" + content)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.08)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--write-baseline", help="把本次结果写成基线 JSON（审核数字后再执行）")
    parser.add_argument("--check-baseline", help="与已保存基线比较；任何指标回退都会失败")
    parser.add_argument("--source-commit", default="working-tree")
    args = parser.parse_args()

    note_files = load_bundled_notes()
    note_paths = [
        path
        for directory in NOTE_DIRS
        for path in sorted(directory.glob("*.md"))
        if path.name.lower() != "readme.md"
    ]
    index = Retriever(load_documents(note_files))

    report = {}
    for name, path in QUESTION_SETS:
        report[name] = evaluate(index, read_questions(path), args.top_k, args.min_score)

    summary = summarise(report)
    payload = {
        "source_commit": args.source_commit,
        "configuration": {"top_k": args.top_k, "min_score": args.min_score},
        "corpus": {
            "files": len(note_files),
            "chunks": len(index.chunks),
            "sha256": digest(note_paths),
            # 用 POSIX 分隔符，避免 Windows 与 Linux 的基线字符串不一致。
            "sources": [directory.relative_to(ROOT).as_posix() for directory in NOTE_DIRS],
        },
        "question_sets": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(
                    path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                ).hexdigest(),
            }
            for name, path in QUESTION_SETS
        },
        "summary": summary,
        "details": report,
        "scope": "离线检索评测：未调用模型，未测回答正确率。分数只反映字符 TF-IDF 在这些开发集上的行为。",
    }

    output = Path(args.summary)
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = Path(args.csv)
    csv_path = csv_path if csv_path.is_absolute() else ROOT / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["set", "id", "kind", "question", "expected_sections", "retrieved", "scores", "passed"],
        )
        writer.writeheader()
        for name, section in report.items():
            for row in section["questions"]:
                writer.writerow({
                    "set": name,
                    "id": row["id"],
                    "kind": row["kind"],
                    "question": row["question"],
                    "expected_sections": row["expected_sections"],
                    "retrieved": "|".join(row["retrieved"]),
                    "scores": "|".join(f"{score:.4f}" for score in row["scores"]),
                    "passed": row["passed"],
                })

    overall = summary["overall"]
    print(f"内置语料：{len(note_files)} 份文件，{len(index.chunks)} 个段落")
    for name in report:
        values = summary[name]
        print(
            f"  {name:<13} 单段 {values['single']['passed']}/{values['single']['total']}"
            f" · 多段 {values['multi']['passed']}/{values['multi']['total']}"
            f" · 无答案空检索 {values['unanswerable']['passed']}/{values['unanswerable']['total']}"
        )
    print(
        f"  合计 {overall['question_count']} 题：单段 {overall['single']['passed']}/{overall['single']['total']}"
        f" · 多段 {overall['multi']['passed']}/{overall['multi']['total']}"
        f" · 无答案 {overall['unanswerable']['passed']}/{overall['unanswerable']['total']}"
    )
    print(f"结果写入：{output.relative_to(ROOT)} 与 {csv_path.relative_to(ROOT)}")

    if args.write_baseline:
        baseline = Path(args.write_baseline)
        baseline = baseline if baseline.is_absolute() else ROOT / baseline
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"基线已更新：{baseline.relative_to(ROOT)}")

    if args.check_baseline:
        baseline = Path(args.check_baseline)
        baseline = baseline if baseline.is_absolute() else ROOT / baseline
        if not baseline.is_file():
            raise SystemExit(f"找不到基线：{baseline}")
        reference = json.loads(baseline.read_text(encoding="utf-8"))
        for key in ("configuration", "corpus", "question_sets"):
            if payload[key] != reference.get(key):
                raise SystemExit(f"评测输入与基线不一致（{key}），请审核后重建基线。")
        for name, metrics in reference["summary"].items():
            if name == "overall":
                continue
            for metric, values in metrics.items():
                if metric in {"question_count", "failure_ids"}:
                    continue
                current = summary[name][metric]["passed"]
                if current < values["passed"]:
                    raise SystemExit(f"检索指标回退：{name}.{metric} {current} < {values['passed']}")
        print("与基线比较通过：没有指标回退。")


if __name__ == "__main__":
    main()