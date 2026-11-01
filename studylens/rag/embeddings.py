"""LangChain Embeddings 适配器：真实 HTTP 调用，有界批次、响应和超时。"""
import hashlib
import json
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from time import monotonic

import requests
from langchain_core.embeddings import Embeddings

from .types import ProviderError


def post_json(base_url: str, route: str, api_key: str, payload: dict,
              max_response_bytes: int = 8 * 1024 * 1024, *, deadline: float | None = None) -> dict:
    # 地址只来自部署配置，并复用服务商白名单；不允许响应重定向携带密钥。
    from studylens.security import validate_model_base_url
    base = validate_model_base_url(base_url)
    # 阶段总预算传到HTTP层；已有请求无法撤回计费，但到期后不再发下一批。
    request_deadline = min(monotonic() + 45, deadline) if deadline is not None else monotonic() + 45
    remaining = _remaining(request_deadline)
    try:
        with requests.post(
            base + route, json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=(min(5, remaining), min(45, remaining)), allow_redirects=False, stream=True,
        ) as response:
            if response.status_code != 200:
                raise ProviderError(f"模型服务返回 HTTP {response.status_code}，请检查部署配置。")
            body = bytearray()
            for part in response.iter_content(chunk_size=65536):
                _remaining(request_deadline)
                body.extend(part)
                if len(body) > max_response_bytes:
                    raise ProviderError("模型服务响应过大，已停止读取。")
            result = json.loads(body.decode("utf-8"))
            _remaining(request_deadline)
    except requests.Timeout as exc:
        raise ProviderError("模型服务超时，请稍后重试。") from exc
    except (requests.RequestException, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderError("模型服务连接失败或返回格式错误。") from exc
    if not isinstance(result, dict):
        raise ProviderError("模型服务未返回 JSON 对象。")
    return result


def _remaining(deadline: float) -> float:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise ProviderError("模型处理时间预算已用完，已停止后续请求；请拆分资料后重试。")
    return remaining


@dataclass
class _EmbeddingBudget:
    deadline: float
    remaining_batches: int


def validate_vectors(vectors, count: int, dimension: int | None = None) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ProviderError("向量服务返回的条目数不匹配。")
    checked = []
    expected = dimension
    for vector in vectors:
        if not isinstance(vector, (list, tuple)) or len(vector) < 2:
            raise ProviderError("向量服务返回了无效向量。")
        if expected is None:
            expected = len(vector)
        if len(vector) != expected:
            raise ProviderError("向量维度发生变化，请重新建立知识库索引。")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in vector):
            raise ProviderError("向量包含无效数值。")
        normalized = [float(v) for v in vector]
        if not any(normalized):
            raise ProviderError("向量服务返回了全零向量。")
        checked.append(normalized)
    return checked


class OpenAICompatibleEmbeddings(Embeddings):
    """实现 LangChain embed_documents/embed_query；不自动下载或伪造模型。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 dimensions: int | None = None, batch_size: int = 32, *,
                 request_dimensions: bool = False, max_batches: int | None = None,
                 max_seconds: float | None = None):
        from studylens.security import validate_model_base_url
        self.base_url = validate_model_base_url(base_url)
        if not api_key.strip() or not model.strip():
            raise ValueError("必须配置 Embedding 密钥和模型名称。")
        if dimensions is not None and not 2 <= dimensions <= 32768:
            raise ValueError("Embedding 维度超出范围。")
        if not 1 <= batch_size <= 64:
            raise ValueError("Embedding 批次必须为 1～64。")
        if request_dimensions and dimensions is None:
            raise ValueError("请求自定义向量维度时必须给出 dimensions。")
        try:
            max_batches = int(os.getenv("STUDYLENS_EMBEDDING_MAX_BATCHES", "16")) if max_batches is None else max_batches
            max_seconds = float(os.getenv("STUDYLENS_EMBEDDING_MAX_SECONDS", "90")) if max_seconds is None else max_seconds
        except ValueError as exc:
            raise ValueError("Embedding处理预算配置无效。") from exc
        if type(max_batches) is not int or not 1 <= max_batches <= 128:
            raise ValueError("Embedding批次预算必须在1到128之间。")
        if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
                or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 180):
            raise ValueError("Embedding时间预算必须在1到180秒之间。")
        self._api_key, self.model = api_key, model
        self.dimensions, self.batch_size = dimensions, batch_size
        self.request_dimensions = request_dimensions
        self.max_batches, self.max_seconds = max_batches, max_seconds
        # ContextVar隔离并发任务；同一导入里的嵌套分块/索引复用计数，不能重新充值预算。
        self._budget = ContextVar(f"studylens_embedding_budget_{id(self)}", default=None)
        # 只存模型身份，不把密钥写入 SQLite、日志或 fingerprint。
        config = f"{self.base_url}|{model}|{dimensions or 'provider-default'}"
        self.embedding_id = hashlib.sha256(config.encode("utf-8")).hexdigest()

    @contextmanager
    def operation_budget(self):
        """包围整个导入；嵌套调用继续使用同一批次计数和截止时间。"""
        token = None
        if self._budget.get() is None:
            token = self._budget.set(_EmbeddingBudget(monotonic() + self.max_seconds, self.max_batches))
        try:
            yield
        finally:
            if token is not None:
                self._budget.reset(token)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding输入必须为非空文本列表。")
        if not texts:
            return []
        with self.operation_budget():
            return self._embed_documents(texts)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        budget = self._budget.get()
        _remaining(budget.deadline)
        batches = math.ceil(len(texts) / self.batch_size)
        # 整批预检在第一次HTTP前完成，避免已知超量的输入只花掉部分费用然后失败。
        if batches > budget.remaining_batches:
            raise ProviderError("Embedding批次预算不足，已停止后续请求；请拆分资料后重试。")
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            _remaining(budget.deadline)
            budget.remaining_batches -= 1
            batch = texts[offset:offset + self.batch_size]
            payload = {"model": self.model, "input": batch, "encoding_format": "float"}
            # bge-m3 为固定 1024 维；只校验返回维度，不给不支持的模型发送 dimensions。
            # 支持可变维度的服务商需显式开启 request_dimensions。
            if self.request_dimensions:
                payload["dimensions"] = self.dimensions
            result = post_json(self.base_url, "/embeddings", self._api_key, payload, deadline=budget.deadline)
            _remaining(budget.deadline)
            try:
                data = result["data"]
                if not isinstance(data, list) or len(data) != len(batch):
                    raise ValueError
                indices = [row["index"] for row in data]
                if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(batch))):
                    raise ValueError
                ordered = [row["embedding"] for row in sorted(data, key=lambda x: x["index"])]
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError("向量服务的返回结构不正确。") from exc
            expected = self.dimensions or (len(vectors[0]) if vectors else None)
            vectors.extend(validate_vectors(ordered, len(batch), expected))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
