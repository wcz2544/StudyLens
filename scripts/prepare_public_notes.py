"""把公开资料抓取并整理为 data/public_notes 下的 Markdown 笔记。

只抓取明确允许再分发的资料：
- 《Rust 程序设计语言》中文版（rust-lang-cn/book-cn），MIT OR Apache-2.0；
- Python 官方文档中文版教程（docs.python.org/zh-cn），PSF-2.0。

生成的文件会提交到仓库，网页构建时直接读取，不依赖网络。需要更新来源时运行：

    python scripts/prepare_public_notes.py

抓取失败会保留已有文件并以非零状态退出，避免悄悄生成半份语料。
"""
from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "public_notes"
CACHE = ROOT / "var" / "public_notes_cache"
RUST_BASE = "https://raw.githubusercontent.com/rust-lang-cn/book-cn/master/src"
PYTHON_BASE = "https://docs.python.org/zh-cn/3/tutorial"
USER_AGENT = "StudyLens-public-notes/1.0 (+https://github.com/wcz2544/StudyLens)"


class MarkdownConverter(HTMLParser):
    """只保留标题、段落、列表、代码与定义列表，够检索与引用使用。"""

    SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript"}
    VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.in_pre = False
        self.heading: int | None = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.VOID_TAGS:
            return
        if tag == "pre":
            self.in_pre = True
            self.parts.append("\n```text\n")
        elif tag in {"h1", "h2", "h3", "h4"}:
            self.heading = int(tag[1])
            self.parts.append("\n\n" + "#" * self.heading + " ")
        elif tag in {"p", "div", "section", "dl", "dd"}:
            self.parts.append("\n\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "dt":
            self.parts.append("\n- ")
        elif tag == "code" and not self.in_pre:
            self.parts.append("`")
        elif tag == "a" and attrs.get("class") == "headerlink":
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth:
            if tag == "a":
                self.skip_depth -= 1
            return
        if tag == "pre":
            self.in_pre = False
            self.parts.append("\n```\n")
        elif tag in {"h1", "h2", "h3", "h4"}:
            self.heading = None
            self.parts.append("\n")
        elif tag == "code" and not self.in_pre:
            self.parts.append("`")

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.in_pre:
            self.parts.append(data)
        else:
            self.parts.append(re.sub(r"\s+", " ", data))


def collapse(text: str) -> str:
    text = text.replace("\u00b6", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def fetch(url: str, cache_name: str) -> str:
    """优先使用本地缓存，避免网络波动让生成结果不可复现。"""
    cached = CACHE / cache_name
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=40) as response:  # noqa: S310 - 只访问固定白名单地址
        text = response.read().decode("utf-8")
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def html_to_markdown(html: str) -> str:
    marker = html.find('<div class="body" role="main"')
    if marker != -1:
        html = html[marker:]
    # 页脚的“目录 / 上一主题 / 下一主题”属于站点导航，不是教程内容。
    for cut in ('<div class="related"', '<footer', '<div class="sphinxsidebar"'):
        index = html.find(cut)
        if index != -1:
            html = html[:index]
    converter = MarkdownConverter()
    converter.feed(html)
    return collapse("".join(converter.parts))


def source_header(title: str, source: str, license_note: str) -> str:
    return (
        f"# {title}\n\n"
        f"> 来源：{source}\n"
        f"> 许可证：{license_note}。本文为节选整理，保留原意；编号与措辞以原文档为准。\n"
        f"> 抓取日期：2026-11-18。\n\n"
    )


RUST_CHAPTERS = [
    ("ch04-01-what-is-ownership.md", "01_Rust_所有权.md", "Rust 程序设计语言 · 什么是所有权"),
    ("ch04-02-references-and-borrowing.md", "02_Rust_借用与引用.md", "Rust 程序设计语言 · 引用与借用"),
    ("ch04-03-slices.md", "03_Rust_切片.md", "Rust 程序设计语言 · 切片"),
]
PYTHON_PAGES = [
    ("datastructures.html", "04_Python_数据结构.md", "Python 教程 · 数据结构"),
    ("controlflow.html", "05_Python_控制流.md", "Python 教程 · 更多控制流工具"),
]


def build_notes() -> dict[str, str]:
    notes: dict[str, str] = {}
    for remote, local, title in RUST_CHAPTERS:
        body = fetch(f"{RUST_BASE}/{remote}", remote)
        notes[local] = (
            source_header(
                title,
                "《Rust 程序设计语言》中文版（rust-lang-cn/book-cn）第 4 章",
                "MIT OR Apache-2.0",
            )
            + body.strip()
            + "\n"
        )
    for remote, local, title in PYTHON_PAGES:
        body = html_to_markdown(fetch(f"{PYTHON_BASE}/{remote}", f"python-{remote}"))
        notes[local] = (
            source_header(
                title,
                "Python 官方文档中文版教程（docs.python.org/zh-cn）",
                "Python Software Foundation License 2.0",
            )
            + body
        )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验已提交文件是否可用，不访问网络")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if args.check:
        missing = [name for name in (local for _, local, _ in (*RUST_CHAPTERS, *PYTHON_PAGES))
                   if not (OUTPUT / name).is_file()]
        if missing:
            raise SystemExit(f"缺少公开资料：{', '.join(missing)}")
        for name in missing:
            pass
        print(f"公开资料齐全：{len(list(OUTPUT.glob('*.md')))} 份。")
        return

    try:
        notes = build_notes()
    except Exception as error:  # noqa: BLE001 - 网络失败时给出可读提示
        raise SystemExit(f"抓取失败，已保留原文件：{error}") from error
    for name, content in notes.items():
        if len(content.encode("utf-8")) > 512 * 1024:
            raise SystemExit(f"{name} 超过单文件上限，请缩小节选范围。")
        (OUTPUT / name).write_text(content, encoding="utf-8")
    total = sum((OUTPUT / name).stat().st_size for name in notes)
    print(f"已写入 {len(notes)} 份公开资料，共 {total} 字节。")


if __name__ == "__main__":
    main()