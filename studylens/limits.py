"""上传前与提取后使用明确、统一的容量限制；所有大小按二进制字节计算。"""
from pathlib import Path
from .documents import MAX_FILE_BYTES, MAX_FILES, MAX_TOTAL_BYTES

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_OFFICE_BYTES = 12 * 1024 * 1024
MAX_RAW_TOTAL_BYTES = 24 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = MAX_RAW_TOTAL_BYTES
UPLOAD_HELP = "最多 10 份；文本 512 KiB/份，图片 8 MiB/份，PDF/DOCX 12 MiB/份，原文件总计 24 MiB。提取文字每份最多 512 KiB、总计 2 MiB。"


def validate_uploads(files: list[tuple[str, bytes]]) -> None:
    """在 OCR 或 PDF 渲染前拒绝超限输入，避免先消耗资源再提示失败。"""
    if not files or len(files) > MAX_FILES:
        raise ValueError("请选择 1～10 份笔记。")
    if sum(len(data) for _, data in files) > MAX_RAW_TOTAL_BYTES:
        raise ValueError("上传原文件总大小不能超过 24 MiB，请分批处理。")
    for name, data in files:
        suffix = Path(name).suffix.lower()
        if suffix in {".txt", ".md"}:
            limit, label = MAX_FILE_BYTES, "512 KiB"
        elif suffix in {".png", ".jpg", ".jpeg"}:
            limit, label = MAX_IMAGE_BYTES, "8 MiB"
        elif suffix in {".pdf", ".docx"}:
            limit, label = MAX_OFFICE_BYTES, "12 MiB"
        else:
            raise ValueError(f"{name} 的文件格式不受支持。")
        if not data or len(data) > limit:
            raise ValueError(f"{name} 为空或超过该格式的 {label} 限制，请压缩或拆分。")


def validate_extracted_text(files: list[tuple[str, bytes]]) -> None:
    for name, data in files:
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"{name} 提取后的文字超过 512 KiB，请拆分原文件。")
    if sum(len(data) for _, data in files) > MAX_TOTAL_BYTES:
        raise ValueError("提取后的文字总计超过 2 MiB，请分批处理。")
