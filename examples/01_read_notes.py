"""从 reference 根目录运行：python examples/01_read_notes.py"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "data" / "sample_notes" / "01_学习方法.md"
print(path.read_text(encoding="utf-8"))
