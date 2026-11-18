"""把 web/ 静态页面组装成可直接发布的站点目录（默认 _site/）。

GitHub Actions 会在部署前执行本脚本；本地预览同样先构建再启动静态服务器：

    python scripts/build_site.py --serve

构建只复制文件、读取 data/sample_notes 与 data/open_corpus 并写入 samples.json，
不执行任何网络请求。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SAMPLE_NOTES = ROOT / "data" / "sample_notes"
OPEN_CORPUS = ROOT / "data" / "open_corpus"
REQUIRED = ("index.html", "assets/app.js", "assets/core.js", "assets/model.js", "assets/styles.css")

# 问句示例只用于按钮，不代表评测口径；真正的评测仍按 questions.csv 全量执行。
SAMPLE_NOTE_QUESTIONS = (
    "SC 和 SCL 译码有什么区别？",
    "分段长度和重叠是多少？",
    "误比特率 BER 和误帧率 FER 有什么区别？",
)
MAX_CORPUS_QUESTIONS = 6


def _check_target(output: Path) -> None:
    resolved = output.resolve()
    if resolved == ROOT or resolved == WEB or WEB in resolved.parents or resolved in WEB.parents:
        raise SystemExit(f"拒绝写入 {resolved}：输出目录不能是仓库根目录或 web/ 本身。")


def _read_notes(directory: Path) -> list[dict]:
    if not directory.is_dir():
        raise SystemExit(f"找不到笔记目录：{directory}")
    files = [
        {"name": path.name, "text": path.read_text(encoding="utf-8")}
        for path in sorted(directory.glob("*.md"))
    ]
    if not files:
        raise SystemExit(f"笔记目录为空：{directory}")
    return files


def _open_corpus_questions() -> list[str]:
    path = OPEN_CORPUS / "questions.csv"
    if not path.is_file():
        return []
    questions: list[str] = []
    # 优先取有明确证据的单段题与多段题；缺证题留作人工体验，不放进示例按钮。
    for kind in ("single", "multi"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                question = (row.get("question") or "").strip()
                if row.get("kind") == kind and question and question not in questions:
                    questions.append(question)
                if len(questions) >= MAX_CORPUS_QUESTIONS:
                    return questions
    return questions


def _corpora_payload() -> dict:
    corpora = [
        {
            "id": "sample_notes",
            "label": "内置示例笔记（3 份）",
            "description": "项目自带的学习、检索与译码笔记，用于快速体验。",
            "questions": list(SAMPLE_NOTE_QUESTIONS),
            "files": _read_notes(SAMPLE_NOTES),
        }
    ]
    if (OPEN_CORPUS / "notes").is_dir():
        corpora.append({
            "id": "open_corpus",
            "label": "开放虚构语料 · 灯塔调度导论（5 份）",
            "description": "虚构课程讲义，覆盖术语、调度周期、告警、实验记录与答疑；以 CC0 发布。",
            "questions": _open_corpus_questions(),
            "files": _read_notes(OPEN_CORPUS / "notes"),
        })
    return {"generated_from": "data/sample_notes + data/open_corpus", "corpora": corpora}


def build(output: Path) -> int:
    _check_target(output)
    if not (WEB / "index.html").is_file():
        raise SystemExit(f"找不到静态页面入口：{WEB / 'index.html'}")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(WEB, output)

    payload = _corpora_payload()
    (output / "samples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # 部署产物不经过 Jekyll；保留该标记，便于改用分支发布时行为一致。
    (output / ".nojekyll").write_text("", encoding="utf-8")

    missing = [name for name in REQUIRED if not (output / name).is_file()]
    missing += [] if (output / "samples.json").is_file() else ["samples.json"]
    if missing:
        raise SystemExit(f"构建产物缺少文件：{', '.join(missing)}")

    count = sum(1 for path in output.rglob("*") if path.is_file())
    summary = "、".join(f"{item['label']} {len(item['files'])} 份" for item in payload["corpora"])
    print(f"已生成 {output}：{count} 个文件；内置语料：{summary}。")
    return count


def serve(output: Path, port: int) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(output), **kwargs)

        def log_message(self, format, *args):  # noqa: A002 - 保持父类签名
            sys.stderr.write("%s %s\n" % (self.address_string(), format % args))

    url = f"http://127.0.0.1:{port}/"
    print(f"本地预览：{url}（Ctrl+C 停止）")
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n已停止本地预览。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site", help="输出目录，默认 _site/")
    parser.add_argument("--serve", action="store_true", help="构建后启动本地静态服务器")
    parser.add_argument("--port", type=int, default=8000, help="本地预览端口，默认 8000")
    args = parser.parse_args()
    build(args.output)
    if args.serve:
        serve(args.output, args.port)


if __name__ == "__main__":
    main()