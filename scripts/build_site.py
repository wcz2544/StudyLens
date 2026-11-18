"""把 web/ 静态页面组装成可直接发布的站点目录（默认 _site/）。

GitHub Actions 会在部署前执行本脚本；本地预览同样先构建再启动静态服务器：

    python scripts/build_site.py --serve

构建只复制文件、读取 data/sample_notes 并写入 samples.json，不执行任何网络请求。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SAMPLE_NOTES = ROOT / "data" / "sample_notes"
REQUIRED = ("index.html", "assets/app.js", "assets/core.js", "assets/model.js", "assets/styles.css")


def _check_target(output: Path) -> None:
    resolved = output.resolve()
    if resolved == ROOT or resolved == WEB or WEB in resolved.parents or resolved in WEB.parents:
        raise SystemExit(f"拒绝写入 {resolved}：输出目录不能是仓库根目录或 web/ 本身。")


def _sample_payload() -> dict:
    if not SAMPLE_NOTES.is_dir():
        raise SystemExit(f"找不到示例笔记目录：{SAMPLE_NOTES}")
    files = []
    for path in sorted(SAMPLE_NOTES.glob("*.md")):
        files.append({"name": path.name, "text": path.read_text(encoding="utf-8")})
    if not files:
        raise SystemExit("示例笔记目录为空，静态版的默认体验会失效。")
    return {"generated_from": "data/sample_notes", "files": files}


def build(output: Path) -> int:
    _check_target(output)
    if not (WEB / "index.html").is_file():
        raise SystemExit(f"找不到静态页面入口：{WEB / 'index.html'}")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(WEB, output)

    payload = _sample_payload()
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
    print(f"已生成 {output}：{count} 个文件，示例笔记 {len(payload['files'])} 份。")
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
