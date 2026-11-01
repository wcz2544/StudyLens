"""仅由部署者提供知识库服务配置；不接收浏览器指定的模型或数据库地址。"""
from dataclasses import dataclass, field
from pathlib import Path
import os
import re


@dataclass(frozen=True)
class Settings:
    admin_key: str = field(repr=False)
    reader_key: str = field(repr=False)
    workspace: str = "default"
    data_dir: str = "var/knowledge"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = field(default="", repr=False)
    collection: str = "studylens_chunks"
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_key: str = field(default="", repr=False)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_key: str = field(default="", repr=False)
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_min_score: float = 0.25
    chat_base_url: str = "https://api.deepseek.com"
    chat_key: str = field(default="", repr=False)
    chat_model: str = "deepseek-flash"
    requests_per_minute: int = 30

    def __post_init__(self):
        if min(len(self.admin_key), len(self.reader_key)) < 32:
            raise ValueError("知识库管理和只读密钥均需至少32字符。")
        if self.admin_key == self.reader_key:
            raise ValueError("管理和只读密钥必须不同。")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", self.workspace):
            raise ValueError("工作区名称格式不正确。")
        if not 1 <= self.requests_per_minute <= 300:
            raise ValueError("每分钟请求上限需在1到300之间。")
        if not 2 <= self.embedding_dimensions <= 32768:
            raise ValueError("Embedding维度配置不正确。")
        if not 0 <= self.rerank_min_score <= 1:
            raise ValueError("重排阈值必须位于0到1之间。")

    @classmethod
    def from_env(cls):
        # 明确映射便于部署审计；不能把用户输入覆盖到环境配置中。
        mapping = {
            "admin_key": "STUDYLENS_ADMIN_KEY", "reader_key": "STUDYLENS_READER_KEY",
            "workspace": "STUDYLENS_WORKSPACE", "data_dir": "STUDYLENS_DATA_DIR",
            "milvus_uri": "MILVUS_URI", "milvus_token": "MILVUS_TOKEN",
            "collection": "MILVUS_COLLECTION", "embedding_base_url": "EMBEDDING_BASE_URL",
            "embedding_key": "EMBEDDING_API_KEY", "embedding_model": "EMBEDDING_MODEL",
            "embedding_dimensions": "EMBEDDING_DIMENSIONS", "rerank_base_url": "RERANK_BASE_URL",
            "rerank_key": "RERANK_API_KEY", "rerank_model": "RERANK_MODEL",
            "rerank_min_score": "RERANK_MIN_SCORE",
            "chat_base_url": "LLM_BASE_URL", "chat_key": "LLM_API_KEY", "chat_model": "LLM_MODEL",
            "requests_per_minute": "STUDYLENS_REQUESTS_PER_MINUTE",
        }
        values = {key: os.environ[name] for key, name in mapping.items() if name in os.environ}
        values.setdefault("admin_key", "")
        values.setdefault("reader_key", "")
        for key in ("embedding_dimensions", "requests_per_minute"):
            if key in values:
                values[key] = int(values[key])
        if "rerank_min_score" in values:
            values["rerank_min_score"] = float(values["rerank_min_score"])
        return cls(**values)


def build_services(settings: Settings):
    """缺少真实服务时明确失败，不自动切换模拟向量或模拟答案。"""
    from studylens.security import validate_model_base_url
    from studylens.rag.embeddings import OpenAICompatibleEmbeddings
    from studylens.rag.store import MilvusVectorStore
    from studylens.rag.rerank import HTTPReranker
    from studylens.rag.retriever import KnowledgeBase
    from studylens.rag.agent import RagAgent
    from studylens.rag.models import LangChainChatModel

    if not all((settings.embedding_key, settings.rerank_key, settings.chat_key, settings.milvus_token)):
        raise ValueError("请配置Embedding、Rerank、问答服务密钥及Milvus访问凭据。")
    for url in (settings.embedding_base_url, settings.rerank_base_url, settings.chat_base_url):
        validate_model_base_url(url)
    # embedding标识进入知识库版本；更换模型时需新建集合并重新导入。
    embeddings = OpenAICompatibleEmbeddings(settings.embedding_base_url, settings.embedding_key,
                                           settings.embedding_model, dimensions=settings.embedding_dimensions)
    embedding_id = embeddings.embedding_id
    store = MilvusVectorStore(settings.milvus_uri, settings.milvus_token,
                              collection=settings.collection, dimension=settings.embedding_dimensions,
                              embedding_id=embedding_id)
    model = None
    try:
        reranker = HTTPReranker(settings.rerank_base_url, settings.rerank_key, settings.rerank_model,
                               min_score=settings.rerank_min_score)
        kb = KnowledgeBase(Path(settings.data_dir), embeddings, store, reranker, embedding_id=embedding_id)
        model = LangChainChatModel(settings.chat_base_url, settings.chat_key, settings.chat_model)
        return kb, RagAgent(kb, model)
    except Exception:
        store.close()
        if model is not None:
            model.close()
        raise
