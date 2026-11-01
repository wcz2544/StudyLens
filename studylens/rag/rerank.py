"""调用真实重排模型；服务失败时明确报错，不假装完成了重排。"""
from dataclasses import replace
import math
from typing import Sequence

from .embeddings import post_json
from .types import Evidence, ProviderError


class HTTPReranker:
    """SiliconFlow /rerank 协议，返回 index 与 relevance_score。"""

    def __init__(self, base_url: str, api_key: str, model: str, min_score: float = 0.25):
        from studylens.security import validate_model_base_url
        self.base_url = validate_model_base_url(base_url)
        if not api_key.strip() or not model.strip():
            raise ValueError("必须配置重排密钥和模型名称。")
        if not 0 <= min_score <= 1:
            raise ValueError("重排过滤阈值必须在 0～1 之间。")
        self._api_key, self.model, self.min_score = api_key, model, min_score

    def rerank(self, query: str, candidates: Sequence[Evidence], top_k: int) -> list[Evidence]:
        if not candidates:
            return []
        if len(candidates) > 100 or not 1 <= top_k <= 100:
            raise ValueError("重排候选数量超出限制。")
        result = post_json(self.base_url, "/rerank", self._api_key, {
            "model": self.model, "query": query,
            "documents": [c.text for c in candidates],
            "top_n": min(top_k, len(candidates)), "return_documents": False,
        }, max_response_bytes=1024 * 1024)
        rows = result.get("results")
        if not isinstance(rows, list):
            raise ProviderError("重排服务的返回结构不正确。")
        seen, output = set(), []
        for row in rows:
            if not isinstance(row, dict):
                raise ProviderError("重排服务的条目格式不正确。")
            index, score = row.get("index"), row.get("relevance_score")
            if (type(index) is not int or not 0 <= index < len(candidates) or index in seen
                    or isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= 1):
                raise ProviderError("重排服务返回了无效编号或分数。")
            seen.add(index)
            if score >= self.min_score:
                candidate = candidates[index]
                output.append(replace(candidate, score=float(score), metadata={
                    **candidate.metadata, "rerank_score": float(score), "rerank_model": self.model,
                }))
        return sorted(output, key=lambda c: (-c.score, c.id))[:top_k]
