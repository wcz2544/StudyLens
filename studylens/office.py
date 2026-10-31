"""安全地将 PDF 与 DOCX 笔记转换为带结构的 Markdown 文本。"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from math import ceil
from time import monotonic
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from pypdf import PdfReader
import pypdfium2 as pdfium

from studylens.ocr import MAX_IMAGE_PIXELS, extract_image_text
from studylens.limits import MAX_OFFICE_BYTES

OFFICE_SUFFIXES = {".pdf", ".docx"}
MAX_PDF_PAGES = 30
MAX_PDF_SECONDS = 90
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_DOCX_FILES = 1000


@dataclass(frozen=True)
class OfficeText:
    text: str
    detail: str
    ocr_pages: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    skipped_pages: tuple[int, ...] = ()


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
    except (BadZipFile, KeyError, ValueError, XMLSyntaxError, OSError) as exc:
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


def _render_pdf_page(renderer, index: int) -> bytes:
    """先限制渲染尺寸，再分配位图，避免等图片生成后才发现像素超限。"""
    page = renderer[index]
    try:
        width, height = page.get_size()
        if width <= 0 or height <= 0 or ceil(width * 2) * ceil(height * 2) > MAX_IMAGE_PIXELS:
            raise ValueError("页面渲染尺寸超过 2000 万像素")
        bitmap = page.render(scale=2)
        try:
            image_buffer = BytesIO()
            with bitmap.to_pil() as image:
                image.save(image_buffer, format="PNG")
            return image_buffer.getvalue()
        finally:
            bitmap.close()
    finally:
        page.close()


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
    warnings: list[str] = []
    skipped_pages: list[int] = []
    renderer = None
    started = monotonic()
    try:
        for index, page in enumerate(reader.pages):
            number = index + 1
            if monotonic() - started > MAX_PDF_SECONDS:
                skipped_pages.extend(range(number, page_count + 1))
                warnings.append(f"处理超过 {MAX_PDF_SECONDS} 秒，第 {number}～{page_count} 页尚未读取。")
                break
            try:
                text = (page.extract_text() or "").strip()
            except Exception:
                text = ""
                warnings.append(f"第 {number} 页文字层读取失败，已尝试 OCR。")
            if len(text) < 10:
                # OCR 失败时保留原文字层，不把仍可使用的短文字静默丢弃。
                try:
                    if renderer is None:
                        renderer = pdfium.PdfDocument(data)
                    recognized = extract_image_text(
                        f"{name}-第{number}页.png", _render_pdf_page(renderer, index), engine_factory()
                    )
                    text = recognized.text
                    pages_for_ocr.append(number)
                except Exception:
                    retained = "已保留文字层，内容可能不完整" if text else "该页未导入"
                    warnings.append(f"第 {number} 页 OCR 失败（组件、尺寸、时限或识别问题）；{retained}。")
            if text:
                page_texts.append(f"# 第 {number} 页\n{text}")
            else:
                skipped_pages.append(number)
    finally:
        if renderer is not None:
            renderer.close()

    if not page_texts:
        raise ValueError(f"{name} 没有识别出可用文字，请检查文件是否清晰。")
    detail = f"PDF 共 {page_count} 页，已提取 {len(page_texts)} 页"
    if pages_for_ocr:
        detail += f"；其中第 {', '.join(map(str, pages_for_ocr))} 页使用 OCR"
    if skipped_pages:
        detail += f"；第 {', '.join(map(str, skipped_pages))} 页未导入"
    return OfficeText("\n\n".join(page_texts), detail + "。", tuple(pages_for_ocr),
                      tuple(warnings), tuple(skipped_pages))
