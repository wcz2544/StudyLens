"""字符 TF-IDF 基线：不下载模型，不需要显卡，也不依赖中文分词库。"""
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from .documents import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("没有可索引的文本段落。")
        self.chunks = chunks
        # 单字及连续 2～3 字片段兼顾中文和 SC/SCL 等英文术语。
        # L2 归一化后，向量点积就是余弦相似度；得分不是正确概率。
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 3), sublinear_tf=True,
            norm="l2", max_features=50000,
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, question: str, top_k: int = 3,
               min_score: float = 0.08) -> list[Hit]:
        if not question.strip():
            return []
        if not 1 <= top_k <= 10 or not 0 <= min_score <= 1:
            raise ValueError("检索参数超出范围。")
        query = self.vectorizer.transform([question.strip()])
        scores = (self.matrix @ query.T).toarray().ravel()
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked[:top_k]
                if scores[i] > 0 and scores[i] >= min_score]
