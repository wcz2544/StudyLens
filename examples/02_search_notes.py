"""从 reference 根目录运行：python -m examples.02_search_notes"""
from pathlib import Path
from studylens.documents import load_documents
from studylens.retrieval import Retriever

root = Path(__file__).resolve().parents[1]
files = [(p.name, p.read_bytes()) for p in sorted((root / "data" / "sample_notes").glob("*.md"))]
index = Retriever(load_documents(files))
question = "每周几整理笔记？"
for hit in index.search(question):
    print(f"\n[{hit.chunk.id}] {hit.chunk.source} / {hit.chunk.section} ({hit.score:.3f})")
    print(hit.chunk.text)
