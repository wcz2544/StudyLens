"""导出“Python 基线 ↔ 浏览器实现”的检索对照数据。

tests/static/core.test.mjs 用这份 JSON 校验 web/assets/core.js 的分段、排序和分数
是否与 studylens/documents.py、studylens/retrieval.py 完全一致。内置笔记以相对路径
登记，避免把整份语料复制进 JSON。

    python scripts/export_web_fixtures.py          # 重新生成
    python scripts/export_web_fixtures.py --check  # CI 中校验是否过期
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studylens.documents import load_documents  # noqa: E402
from studylens.retrieval import Retriever  # noqa: E402

FIXTURE = ROOT / "tests" / "static" / "fixtures" / "retrieval.json"
SAMPLE_NOTES = ROOT / "data" / "sample_notes"
OPEN_CORPUS = ROOT / "data" / "open_corpus" / "notes"
PUBLIC_NOTES = ROOT / "data" / "public_notes"

SAMPLE_QUERIES = [
    ("SC 和 SCL 译码有什么区别？", 3, 0.08),
    ("分段长度和重叠是多少？", 3, 0.08),
    ("反馈怎么保存", 3, 0.0),
    ("误比特率 BER 和误帧率 FER 有什么区别？", 5, 0.05),
    ("完全无关的问题：火星土壤成分", 3, 0.08),
    ("复习卡片有几个字段", 2, 0.2),
    ("每周什么时候自测", 1, 0.08),
]

OPEN_QUERIES = [
    ("LBS 的英文全称是什么？", 3, 0.08),
    ("调度员、值守员和审核员分别负责什么？", 3, 0.08),
    ("每周几整理信标日志？每周几自测？", 3, 0.08),
    ("告警等级怎么划分？", 3, 0.0),
]

PUBLIC_QUERIES = [
    ("所有权有哪三条规则？", 3, 0.08),
    ("同一作用域里可以同时存在几个可变引用？", 3, 0.08),
    ("用列表实现堆栈时用什么方法实现后进先出？", 3, 0.08),
    ("range() 函数返回什么？", 3, 0.08),
    ("借用检查器如何判断引用有效？", 3, 0.08),
]

# 合并后的索引与网页“内置示例笔记”一致，用来锁定跨语料时的排序行为。
BUNDLED_QUERIES = [
    ("SC 和 SCL 译码有什么区别？", 3, 0.08),
    ("LBS 的英文全称是什么？", 3, 0.08),
    ("所有权有哪三条规则？", 3, 0.08),
    ("用列表实现堆栈时用什么方法实现后进先出？", 3, 0.08),
    ("每周几整理信标日志？每周几自测？", 3, 0.08),
    ("借用检查器如何判断引用有效？", 3, 0.08),
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


def _paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"找不到语料目录：{directory}")
    paths = [path for path in sorted(directory.glob("*.md")) if path.name.lower() != "readme.md"]
    if not paths:
        raise SystemExit(f"语料目录为空：{directory}")
    return paths


def _read(paths: list[Path]) -> list[tuple[str, bytes]]:
    return [(path.name, path.read_bytes()) for path in paths]


def _chunk_json(chunk) -> dict:
    # 只保存正文的哈希：对照数据不复制整份语料，仍能逐段校验是否一致。
    return {
        "id": chunk.id,
        "source": chunk.source,
        "section": chunk.section,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        "chars": len(chunk.text),
    }


def _case(name: str, files: list[tuple[str, bytes]], queries, file_refs: list[dict]) -> dict:
    chunks = load_documents(files)
    retriever = Retriever(chunks)
    return {
        "name": name,
        "files": file_refs,
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


def _path_case(name: str, paths: list[Path], queries) -> dict:
    refs = [{"name": path.name, "path": path.relative_to(ROOT).as_posix()} for path in paths]
    return _case(name, _read(paths), queries, refs)


def _inline_case(name: str, files: list[tuple[str, bytes]], queries) -> dict:
    refs = [{"name": file_name, "text": data.decode("utf-8-sig")} for file_name, data in files]
    return _case(name, files, queries, refs)


def build_payload() -> dict:
    import sklearn

    sample = _paths(SAMPLE_NOTES)
    open_corpus = _paths(OPEN_CORPUS)
    public = _paths(PUBLIC_NOTES)
    return {
        "generated_by": "scripts/export_web_fixtures.py",
        "sklearn_version": sklearn.__version__,
        "cases": [
            _path_case("sample_notes", sample, SAMPLE_QUERIES),
            _path_case("open_corpus", open_corpus, OPEN_QUERIES),
            _path_case("public_notes", public, PUBLIC_QUERIES),
            _path_case("bundled", sample + open_corpus + public, BUNDLED_QUERIES),
            _inline_case(
                "synthetic_markdown",
                [("合成样例.md", SYNTHETIC_MARKDOWN.encode("utf-8"))],
                [
                    ("码点偏移", 3, 0.05),
                    ("代码里的井号", 3, 0.0),
                    ("结尾行号", 2, 0.0),
                ],
            ),
            _inline_case(
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