"""UTF-8 笔记读取与按标题分段，保留能追溯到原文的行号。"""
from dataclasses import dataclass
from pathlib import Path
import re

MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024
MAX_FILES = 10


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    section: str
    start_line: int
    end_line: int
    text: str


def parse_document(name: str, data: bytes, doc_id: str,
                   chunk_size: int = 500, overlap: int = 80) -> list[Chunk]:
    if Path(name).suffix.lower() not in {".txt", ".md"}:
        raise ValueError("仅支持 .txt 和 .md 文件。")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"{name} 超过 512 KB，请先缩小文件。")
    if not 0 <= overlap < chunk_size:
        raise ValueError("重叠长度必须小于分段长度。")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} 不是 UTF-8，请在编辑器中另存为 UTF-8。") from exc
    if not content.strip() or "\x00" in content:
        raise ValueError(f"{name} 为空或不是正常文本。")

    chunks: list[Chunk] = []
    section = "正文"
    block: list[str] = []
    block_start = 1
    in_code = False

    def flush() -> None:
        text = "".join(block)
        # 滑动窗口控制上下文大小，重叠区域减少句子被截断导致的信息丢失。
        offset = 0
        while offset < len(text):
            end = min(offset + chunk_size, len(text))
            fragment = text[offset:end]
            if fragment.strip():
                first = block_start + text[:offset].count("\n")
                last = first + fragment.rstrip("\r\n").count("\n")
                chunks.append(Chunk(
                    f"{doc_id}-{len(chunks) + 1}", name, section,
                    first, last, fragment,
                ))
            if end == len(text):
                break
            offset = end - overlap

    for number, line in enumerate(content.splitlines(keepends=True), 1):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line) if not in_code else None
        if heading and Path(name).suffix.lower() == ".md":
            flush()
            block = []
            section = heading.group(1)
            block_start = number
        if not block:
            block_start = number
        block.append(line)
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
    flush()
    return chunks


def load_documents(files: list[tuple[str, bytes]]) -> list[Chunk]:
    if not files or len(files) > MAX_FILES:
        raise ValueError("请选择 1～10 份笔记。")
    if sum(len(data) for _, data in files) > MAX_TOTAL_BYTES:
        raise ValueError("文件总大小不能超过 2 MB。")
    chunks = []
    for index, (name, data) in enumerate(files, 1):
        chunks.extend(parse_document(name, data, f"D{index}"))
    return chunks
