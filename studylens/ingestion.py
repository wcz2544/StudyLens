"""统一服务端文件入口；保留原文件名称与提取警告。"""
from pathlib import Path
from studylens.documents import parse_document
from studylens.limits import validate_uploads, validate_extracted_text
from studylens.ocr import create_ocr_engine, extract_image_text, is_image
from studylens.office import extract_docx_text, extract_pdf_text


def extract_upload(name: str, data: bytes) -> tuple[str, dict]:
    if not name or len(name) > 200 or Path(name).name != name or "\\" in name:
        raise ValueError("请使用不包含目录的文件名，最多200字符。")
    validate_uploads([(name, data)])
    suffix = Path(name).suffix.lower()
    metadata = {"original_name": name, "warnings": [], "format": suffix}
    if is_image(name):
        result = extract_image_text(name, data, create_ocr_engine())
        text = result.text
        metadata["ocr_confidence"] = result.confidence
        metadata["warnings"] = ["OCR文字需与原图核对；置信度不代表答案正确率。"]
    elif suffix == ".pdf":
        result = extract_pdf_text(name, data, create_ocr_engine)
        text = result.text
        metadata.update(warnings=list(result.warnings), ocr_pages=list(result.ocr_pages),
                        skipped_pages=list(result.skipped_pages))
    elif suffix == ".docx":
        result = extract_docx_text(name, data)
        text = result.text
        metadata["warnings"] = list(result.warnings)
    else:
        # 复用已验证的编码与内容检查，但不使用基础分块结果替代语义分块。
        parse_document(name, data, 1)
        text = data.decode("utf-8-sig")
    validate_extracted_text([(name + ".md", text.encode("utf-8"))])
    return text, metadata
