"""检索层与问答层之间的稳定数据契约。"""
from dataclasses import dataclass, field
from typing import Protocol, Sequence


class KnowledgeError(RuntimeError):
    """可向调用层报告的知识库错误，不包含第三方响应和密钥。"""


class ProviderError(KnowledgeError):
    pass


class ConcurrentUpdateError(KnowledgeError):
    pass


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    section: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IndexedChunk:
    id: str
    kb_id: str
    source_id: str
    revision: str
    parent_id: str
    source: str
    section: str
    text: str
    parent_text: str
    metadata: dict = field(default_factory=dict)


class VectorStore(Protocol):
    dimension: int

    def upsert(self, chunks: Sequence[IndexedChunk], vectors: list[list[float]]) -> None: ...

    def search(self, kb_id: str, vector: list[float], revisions: list[str],
               limit: int) -> list[tuple[str, float]]: ...

    def delete(self, kb_id: str, revisions: list[str]) -> None: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[Evidence],
               top_k: int) -> list[Evidence]: ...
