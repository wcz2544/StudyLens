"""真实Tesseract烟雾测试；仅验证印刷英文路径，不代表中文/手写准确率。"""
from io import BytesIO
from shutil import which
import unittest
from PIL import Image, ImageDraw, ImageFont
from studylens.ocr import create_ocr_engine, extract_image_text


@unittest.skipUnless(which("tesseract"), "本机无Tesseract，真实OCR在Linux CI/容器验收")
class OCRRuntimeTests(unittest.TestCase):
    def test_real_tesseract_text_enters_document_pipeline(self):
        canvas = Image.new("RGB", (900, 130), "white")
        ImageDraw.Draw(canvas).text((30, 25), "POLAR CODE SC SCL", fill="black",
                                   font=ImageFont.load_default(size=48))
        raw = BytesIO()
        canvas.save(raw, format="PNG")
        result = extract_image_text("runtime.png", raw.getvalue(), create_ocr_engine())
        self.assertIn("POLAR", result.text.upper())
        self.assertGreater(result.line_count, 0)


if __name__ == "__main__":
    unittest.main()
