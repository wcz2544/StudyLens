"""将图片笔记在本地识别为可供现有检索流程使用的 UTF-8 文本。"""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from shutil import which
from types import SimpleNamespace

from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


@dataclass(frozen=True)
class OCRText:
    text: str
    confidence: float
    line_count: int


def is_image(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_SUFFIXES


def create_ocr_engine():
    """云端优先使用轻量 Tesseract；本地未安装时兼容已有 RapidOCR。"""
    if which("tesseract"):
        import pytesseract
        from pytesseract import Output

        def tesseract_engine(data: bytes):
            with Image.open(BytesIO(data)) as image:
                result = pytesseract.image_to_data(
                    image.convert("RGB"), lang="chi_sim+eng",
                    config="--psm 6", output_type=Output.DICT,
                )
            grouped: dict[tuple[int, int, int], list[tuple[str, float]]] = {}
            for index, text in enumerate(result["text"]):
                text = text.strip()
                if not text:
                    continue
                key = (result["block_num"][index], result["par_num"][index], result["line_num"][index])
                confidence = max(float(result["conf"][index]), 0.0) / 100
                grouped.setdefault(key, []).append((text, confidence))
            lines = [" ".join(word for word, _ in words) for words in grouped.values()]
            scores = [sum(score for _, score in words) / len(words) for words in grouped.values()]
            return SimpleNamespace(txts=tuple(lines), scores=tuple(scores))

        return tesseract_engine

    try:
        from rapidocr import RapidOCR
        return RapidOCR()
    except ImportError as exc:
        raise ValueError("OCR 组件尚未安装，请联系应用维护者。") from exc


def extract_image_text(name: str, data: bytes, engine=None) -> OCRText:
    """校验图片并识别文字；engine 参数便于测试时替换真实 OCR。"""
    if not is_image(name):
        raise ValueError(f"{name} 不是支持的图片格式，请上传 JPG、JPEG 或 PNG。")
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{name} 为空或超过 8 MB，请压缩后重试。")

    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"{name} 像素过大，请缩小到 2000 万像素以内。")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{name} 不是有效图片或文件已经损坏。") from exc

    if engine is None:
        # 延迟导入可避免纯文本用户承担 OCR 模型的启动时间。
        from rapidocr import RapidOCR
        engine = RapidOCR()

    try:
        result = engine(data)
    except Exception as exc:
        raise ValueError(f"{name} 文字识别失败，请稍后重试或改用文字版文件。") from exc
    lines = [text.strip() for text in (getattr(result, "txts", None) or ()) if text.strip()]
    if not lines:
        raise ValueError(f"{name} 未识别出文字，请换用更清晰、光线更均匀的图片。")
    scores = list(getattr(result, "scores", None) or ())
    confidence = sum(scores) / len(scores) if scores else 0.0
    return OCRText("\n".join(lines), confidence, len(lines))
