"""导出“Python 基线 ↔ 浏览器实现”的检索对照数据。

tests/static/core.test.mjs 用这份 JSON 校验 web/assets/core.js 的分段、排序和分数
是否与 studylens/documents.py、studylens/retrieval.py 完全一致。

    python scripts/export_web_fixtures.py          # 重新生成
    python scripts/export_web_fixtures.py --check  # CI 中校验是否过期
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studylens.documents import load_documents  # noqa: E402
from studylens.retrieval import Retriever  # noqa: E402

FIXTURE = ROOT / "tests" / "static" / "fixtures" / "retrieval.json"
SAMPLE_NOTES = ROOT / "data" / "sample_notes"

SAMPLE_QUERIES = [
    ("SC 和 SCL 译码有什么区别？", 3, 0.08),
    ("分段长度和重叠是多少？", 3, 0.08),
    ("反馈怎么保存", 3, 0.0),
    ("误比特率 BER 和误帧率 FER 有什么区别？", 5, 0.05),
    ("完全无关的问题：火星土壤成分", 3, 0.08),
    ("复习卡片有几个字段", 2, 0.2),
    ("每周什么时候自测", 1, 0.08),
]

# 合成样例专门覆盖：代码块里的井号、超长小节的滑动窗口、emoji 造成的码点偏移。
SYNTHETIC_MARKDOWN = "\n".join([
    "# 长小节",
    "",
    "开头一句，带一个表情 🚀 和英文 SCL。",
    "",
    "## 代码块",
    "```text",
    "# 代码里的井号不是标题",
    "```",
    "",
    "## 超长段落",
    "很长的一段：" + ("字符偏移必须按码点计算，😀 也要算一个字。" * 25),
    "",
    "### ",
    "",
    "## 结尾",
    "最后一行用于检查行号。",
    "",
])

SYNTHETIC_TEXT = "# 看起来像标题\n第一行\n第二行\n"


def _sample_files() -> list[tuple[str, bytes]]:
    return [(path.name, path.read_bytes()) for path in sorted(SAMPLE_NOTES.glob("*.md"))]


def _chunk_json(chunk) -> dict:
    return {
        "id": chunk.id,
        "source": chunk.source,
        "section": chunk.section,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
    }


def _case(name: str, files: list[tuple[str, bytes]], queries: list[tuple[str, int, float]]) -> dict:
    chunks = load_documents(files)
    retriever = Retriever(chunks)
    return {
        "name": name,
        "files": [{"name": file_name, "text": data.decode("utf-8-sig")} for file_name, data in files],
        "chunks": [_chunk_json(chunk) for chunk in chunks],
        "queries": [
            {
                "question": question,
                "top_k": top_k,
                "min_score": min_score,
                "hits": [
                    {"id": hit.chunk.id, "score": hit.score}
                    for hit in retriever.search(question, top_k, min_score)
                ],
            }
            for question, top_k, min_score in queries
        ],
    }


def build_payload() -> dict:
    import sklearn

    sample_files = _sample_files()
    if not sample_files:
        raise SystemExit("data/sample_notes 为空，无法生成对照数据。")
    return {
        "generated_by": "scripts/export_web_fixtures.py",
        "sklearn_version": sklearn.__version__,
        "cases": [
            _case("sample_notes", sample_files, SAMPLE_QUERIES),
            _case(
                "synthetic_markdown",
                [("合成样例.md", SYNTHETIC_MARKDOWN.encode("utf-8"))],
                [
                    ("码点偏移", 3, 0.05),
                    ("代码里的井号", 3, 0.0),
                    ("结尾行号", 2, 0.0),
                ],
            ),
            _case(
                "plain_text",
                [("plain.txt", SYNTHETIC_TEXT.encode("utf-8"))],
                [
                    ("看起来像标题", 2, 0.0),
                    ("第二行", 2, 0.0),
                ],
            ),
        ],
    }


def serialise(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验现有文件是否与当前代码一致")
    args = parser.parse_args()

    payload = build_payload()
    content = serialise(payload)
    if args.check:
        if not FIXTURE.is_file():
            raise SystemExit(f"缺少对照数据：{FIXTURE}")
        existing = FIXTURE.read_text(encoding="utf-8")
        if existing != content:
            for number, (old, new) in enumerate(zip(existing.splitlines(), content.splitlines()), 1):
                if old != new:
                    raise SystemExit(
                        "对照数据已过期，请运行 python scripts/export_web_fixtures.py 后重新提交。\n"
                        f"第 {number} 行：\n  现有：{old}\n  期望：{new}"
                    )
            raise SystemExit("对照数据行数不一致，请重新生成后提交。")
        print(f"对照数据与当前实现一致：{FIXTURE}")
        return

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(content, encoding="utf-8")
    chunks = sum(len(case["chunks"]) for case in payload["cases"])
    print(f"已写入 {FIXTURE}：{len(payload['cases'])} 组样例，{chunks} 个段落。")


if __name__ == "__main__":
    main()
