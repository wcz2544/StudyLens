"""安全地将 PDF 与 DOCX 笔记转换为带结构的 Markdown 文本。"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
import pypdfium2 as pdfium

from studylens.ocr import extract_image_text

OFFICE_SUFFIXES = {".pdf", ".docx"}
MAX_OFFICE_BYTES = 12 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_DOCX_FILES = 1000


@dataclass(frozen=True)
class OfficeText:
    text: str
    detail: str
    ocr_pages: tuple[int, ...] = ()


def is_office_document(name: str) -> bool:
    return Path(name).suffix.lower() in OFFICE_SUFFIXES


def _validate_size(name: str, data: bytes) -> None:
    if not data or len(data) > MAX_OFFICE_BYTES:
        raise ValueError(f"{name} 为空或超过 12 MB，请压缩或拆分后重试。")


def extract_docx_text(name: str, data: bytes) -> OfficeText:
    """按文档顺序提取 DOCX 标题、段落和表格。"""
    _validate_size(name, data)
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_DOCX_FILES or sum(item.file_size for item in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise ValueError(f"{name} 解压后的内容过大，请压缩或拆分后重试。")
        document = Document(BytesIO(data))
    except (BadZipFile, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and "解压后的内容过大" in str(exc):
            raise
        raise ValueError(f"{name} 不是有效的 DOCX 文件或文件已经损坏。") from exc

    lines: list[str] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = item.style.name if item.style else ""
            if style.startswith("Heading"):
                level_text = style.removeprefix("Heading").strip()
                level = min(max(int(level_text), 1), 6) if level_text.isdigit() else 2
                lines.append(f"{'#' * level} {text}")
            else:
                lines.append(text)
        elif isinstance(item, Table):
            for row in item.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))

    text = "\n\n".join(lines).strip()
    if not text:
        raise ValueError(f"{name} 没有可读取的文字；如果内容全是图片，请先导出为 PDF 或图片。")
    return OfficeText(text, f"已提取 DOCX 中的段落和表格，共 {len(lines)} 项。")


def extract_pdf_text(name: str, data: bytes, engine_factory) -> OfficeText:
    """优先读取 PDF 文字层；没有文字层的页面渲染后交给 OCR。"""
    _validate_size(name, data)
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise ValueError(f"{name} 已加密，请先移除密码。")
        page_count = len(reader.pages)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{name} 不是有效的 PDF 文件或文件已经损坏。") from exc
    if not 1 <= page_count <= MAX_PDF_PAGES:
        raise ValueError(f"{name} 页数必须在 1～{MAX_PDF_PAGES} 页之间。")

    page_texts: list[str] = []
    pages_for_ocr: list[int] = []
    renderer = None
    try:
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < 10:
                # 扫描页没有可靠文字层时才渲染，减少普通 PDF 的处理时间。
                renderer = renderer or pdfium.PdfDocument(data)
                bitmap = renderer[index].render(scale=2)
                image_buffer = BytesIO()
                bitmap.to_pil().save(image_buffer, format="PNG")
                try:
                    recognized = extract_image_text(
                        f"{name}-第{index + 1}页.png", image_buffer.getvalue(), engine_factory()
                    )
                    text = recognized.text
                    pages_for_ocr.append(index + 1)
                except ValueError:
                    text = ""
            if text:
                page_texts.append(f"# 第 {index + 1} 页\n{text}")
    finally:
        if renderer is not None:
            renderer.close()

    if not page_texts:
        raise ValueError(f"{name} 没有识别出可用文字，请检查文件是否清晰。")
    detail = f"已提取 PDF 共 {page_count} 页"
    if pages_for_ocr:
        detail += f"；其中第 {', '.join(map(str, pages_for_ocr))} 页使用 OCR"
    return OfficeText("\n\n".join(page_texts), detail + "。", tuple(pages_for_ocr))
