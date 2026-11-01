"""有预算、有明确工具边界的检索 Agent；模型只提建议，控制权留在程序内。"""
from __future__ import annotations

import math
import re
import time
import unicodedata
from typing import Protocol, TYPE_CHECKING

from .models import ModelFailure, StructuredModel

if TYPE_CHECKING:
    from .types import Evidence


class KnowledgeSource(Protocol):
    def search(self, kb_id: str, query: str, top_k: int = 5) -> list[Evidence]: ...
    def validate_evidence(self, kb_id: str, evidence: list[Evidence]) -> bool: ...


class InvalidModelOutput(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


_REFUSAL = "当前知识库没有足够且可核对的证据回答这个问题，请补充资料或缩小问题范围。"
_SMALLTALK = "你好，我可以根据当前知识库查找资料、比较信息并给出引用。请告诉我你想了解的具体问题。"


def _query_key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def _queries(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > 10:
        raise InvalidModelOutput("检索计划格式错误。")
    output = []
    seen = set()
    for query in value:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise InvalidModelOutput("检索词格式错误。")
        query = query.strip()
        key = _query_key(query)
        if key not in seen:
            output.append(query)
            seen.add(key)
    return output


def _object(result: object, keys: set[str]) -> dict:
    # 拒绝模型插入 kb_id、url、tool_name 等未授权字段。
    if not isinstance(result, dict) or set(result) != keys:
        raise InvalidModelOutput("模型返回了未约定的字段。")
    return result


class RagAgent:
    def __init__(self, knowledge_base: KnowledgeSource, model: StructuredModel,
                 top_k: int = 5, max_rounds: int = 2, max_queries: int = 3,
                 max_seconds: float = 180, max_context_chars: int = 16_000):
        if not 1 <= top_k <= 8 or not 1 <= max_rounds <= 2 or not 1 <= max_queries <= 3:
            raise ValueError("Agent 检索预算超出允许范围。")
        if not 1 <= max_seconds <= 300 or not 1000 <= max_context_chars <= 24_000:
            raise ValueError("Agent 时间或上下文预算超出允许范围。")
        self.knowledge_base = knowledge_base
        self.model = model
        self.top_k, self.max_rounds, self.max_queries = top_k, max_rounds, max_queries
        self.max_seconds, self.max_context_chars = max_seconds, max_context_chars

    def as_runnable(self):
        """供 LangChain 上层流程组合使用；作用域只能由调用方传入。"""
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda request: self.answer(request["kb_id"], request["question"]))

    def answer(self, kb_id: str, question: str) -> dict:
        if not isinstance(kb_id, str) or not kb_id.strip() or len(kb_id) > 128:
            raise ValueError("知识库标识不合法。")
        if not isinstance(question, str) or not question.strip() or len(question) > 1000:
            raise ValueError("请输入 1～1000 字的问题。")
        question = question.strip()
        started = time.monotonic()
        trace: list[dict] = []
        model_calls = 0
        searched: list[str] = []
        evidence: list[Evidence] = []

        def event(stage: str, status: str, **fields):
            trace.append({"stage": stage, "status": status, **fields})

        def budget():
            # 阶段间预算与服务商请求超时配合；不伪称能撤销已到达服务商的请求。
            if time.monotonic() - started >= self.max_seconds:
                raise BudgetExceeded()

        def call(stage: str, payload: dict) -> dict:
            nonlocal model_calls
            budget()
            if model_calls >= 4:
                raise BudgetExceeded()
            model_calls += 1
            value = self.model.invoke(stage, payload)
            budget()
            event(stage, "completed")
            return value

        def finish(answer: str = _REFUSAL, *, insufficient: bool = True,
                   clarification: bool = False, citations=None, claims=None,
                   error: str | None = None) -> dict:
            return {
                "answer": answer, "citations": citations or [], "claims": claims or [],
                "insufficient": insufficient, "needs_clarification": clarification,
                "evidence": [self._public_evidence(item) for item in evidence],
                "trace": trace, "error": error,
                "usage": {"model_calls": model_calls, "search_calls": len(searched),
                          "elapsed_seconds": round(time.monotonic() - started, 3)},
            }

        def search_round(queries: list[str], number: int) -> list[Evidence]:
            found = {item.id: item for item in evidence}
            for query in queries:
                budget()
                if len(searched) >= self.max_queries:
                    event("retrieve", "query_budget_reached", round=number)
                    break
                if _query_key(query) in {_query_key(q) for q in searched}:
                    event("retrieve", "duplicate_skipped", round=number)
                    continue
                searched.append(query)
                try:
                    hits = self.knowledge_base.search(kb_id, query, top_k=self.top_k)
                except Exception:
                    event("retrieve", "failed", round=number)
                    raise
                for hit in hits:
                    if not isinstance(hit.text, str) or not hit.text.strip():
                        continue
                    if not math.isfinite(float(hit.score)):
                        continue
                    # 工具仍须验证作用域，避免实现错误使其他知识库片段进入提示词。
                    if hit.metadata.get("kb_id") != kb_id:
                        raise ValueError("检索结果作用域不匹配。")
                    previous = found.get(hit.id)
                    if previous is None or hit.score > previous.score:
                        found[hit.id] = hit
                event("retrieve", "completed", round=number, hits=len(hits), query=query)
            return sorted(found.values(), key=lambda hit: (-hit.score, hit.id))[:self.top_k]

        # 明确的寒暄无需付费调用；未知问题仍交给受限决策组件。
        if re.fullmatch(r"(你好|您好|谢谢|谢谢你|多谢|再见|hi|hello|thanks)[！!。.?？\s]*", question, re.I):
            event("route", "smalltalk_without_search")
            return finish(_SMALLTALK, insufficient=False)

        try:
            plan = _object(call("route", {"question": question}),
                           {"action", "queries", "clarification"})
            queries = _queries(plan["queries"])
            clarification = plan["clarification"]
            if (plan["action"] not in {"retrieve", "clarify", "smalltalk"}
                    or not isinstance(clarification, str) or len(clarification) > 300):
                raise InvalidModelOutput("非法决策。")
            if plan["action"] == "clarify":
                if queries or not clarification.strip():
                    raise InvalidModelOutput("澄清计划错误。")
                event("route", "clarification_required")
                return finish(clarification.strip(), clarification=True)
            if plan["action"] == "smalltalk":
                if queries or clarification:
                    raise InvalidModelOutput("闲聊计划错误。")
                event("route", "smalltalk")
                return finish(_SMALLTALK, insufficient=False)
            if clarification:
                raise InvalidModelOutput("检索与澄清计划冲突。")
            queries = queries or [question]
            # 首轮最多两次，余下预算留给证据不足时的动态改写。
            evidence = search_round(queries[:min(2, self.max_queries)], 1)
            if not evidence:
                event("retrieve", "empty_stop")
                return finish(error="no_evidence")
            if not self.knowledge_base.validate_evidence(kb_id, evidence):
                event("freshness", "changed")
                evidence = []
                return finish("资料刚刚更新，请重新提问以使用最新版本。", error="knowledge_changed")

            assessment = _object(call("assess", {
                "question": question, "evidence": self._context(evidence),
                "searched": searched, "remaining_queries": self.max_queries - len(searched),
            }), {"sufficient", "queries", "clarification"})
            rewrites = _queries(assessment["queries"])
            clarification = assessment["clarification"]
            if (type(assessment["sufficient"]) is not bool
                    or not isinstance(clarification, str) or len(clarification) > 300):
                raise InvalidModelOutput("证据评估格式错误。")
            if assessment["sufficient"] and (rewrites or clarification):
                raise InvalidModelOutput("证据评估相互矛盾。")
            if not assessment["sufficient"]:
                if clarification.strip():
                    if rewrites:
                        raise InvalidModelOutput("澄清和改写计划冲突。")
                    event("assess", "clarification_required")
                    return finish(clarification.strip(), clarification=True)
                if not rewrites or self.max_rounds < 2 or len(searched) >= self.max_queries:
                    event("assess", "insufficient_stop")
                    return finish(error="insufficient_evidence")
                prior_ids = {hit.id for hit in evidence}
                evidence = search_round(rewrites[:1], 2)
                if not ({hit.id for hit in evidence} - prior_ids):
                    event("rewrite", "no_new_evidence_stop")
                    return finish(error="insufficient_evidence")

            if not self.knowledge_base.validate_evidence(kb_id, evidence):
                event("freshness", "changed")
                evidence = []
                return finish("资料刚刚更新，请重新提问以使用最新版本。", error="knowledge_changed")
            context = self._context(evidence)
            draft = call("draft", {"question": question, "evidence": context})
            claims = self._validate_claims(draft, {item["id"]: item["text"] for item in context})
            if not claims:
                event("draft", "insufficient_stop")
                return finish(error="insufficient_evidence")
            audit = _object(call("audit", {"question": question, "claims": claims, "evidence": context}),
                            {"supported", "answers_question", "unsupported_claims"})
            if (type(audit["supported"]) is not bool or type(audit["answers_question"]) is not bool
                    or not isinstance(audit["unsupported_claims"], list)
                    or any(type(i) is not int or not 0 <= i < len(claims) for i in audit["unsupported_claims"])):
                raise InvalidModelOutput("审核结果格式错误。")
            if not audit["supported"] or not audit["answers_question"] or audit["unsupported_claims"]:
                event("audit", "unsupported_stop")
                return finish(error="unsupported_answer")
            if not self.knowledge_base.validate_evidence(kb_id, evidence):
                event("freshness", "changed")
                evidence = []
                return finish("资料刚刚更新，请重新提问以使用最新版本。", error="knowledge_changed")
            citations = list(dict.fromkeys(citation["id"] for claim in claims for citation in claim["citations"]))
            event("answer", "verified_structure_and_reviewed")
            return finish("\n".join(claim["text"] for claim in claims), insufficient=False,
                          citations=citations, claims=claims)
        except BudgetExceeded:
            event("budget", "exhausted")
            return finish("本次处理已达到预算，请缩小问题范围后重试。", error="budget_exceeded")
        except InvalidModelOutput:
            event("validation", "invalid_model_output")
            return finish("模型返回内容未通过检查，本次不提供未经核对的答案。", error="invalid_model_output")
        except ModelFailure:
            event("model", "failed")
            return finish("模型服务暂不可用，本次未生成答案。请稍后重试。", error="model_unavailable")
        except Exception:
            # 边界处保守拒答，响应不包含数据库路径、服务商响应或底层异常详情。
            event("pipeline", "failed")
            return finish("知识库处理暂不可用，本次未生成答案。请稍后重试。", error="pipeline_unavailable")

    def _context(self, evidence: list[Evidence]) -> list[dict]:
        budget = self.max_context_chars
        context = []
        for item in evidence:
            if budget <= 0:
                break
            # 截断只用于模型上下文预算；引文也必须在实际发送的文本里存在。
            text = item.text[:min(5000, budget)]
            context.append({"id": item.id, "source": item.source, "section": item.section, "text": text})
            budget -= len(text)
        return context

    @staticmethod
    def _public_evidence(item: Evidence) -> dict:
        safe_metadata = {key: value for key, value in item.metadata.items()
                         if key in {"source_id", "revision", "page", "parent_id", "method"}}
        return {"id": item.id, "source": item.source, "section": item.section,
                "text": item.text, "score": item.score, "metadata": safe_metadata}

    @staticmethod
    def _validate_claims(draft: object, allowed: dict[str, str]) -> list[dict]:
        draft = _object(draft, {"insufficient", "claims"})
        if type(draft["insufficient"]) is not bool or not isinstance(draft["claims"], list):
            raise InvalidModelOutput("答案格式错误。")
        if draft["insufficient"]:
            if draft["claims"]:
                raise InvalidModelOutput("拒答不能同时包含结论。")
            return []
        if not 1 <= len(draft["claims"]) <= 8:
            raise InvalidModelOutput("结论数量超出预算。")
        claims = []
        for value in draft["claims"]:
            claim = _object(value, {"text", "citations"})
            if (not isinstance(claim["text"], str) or not claim["text"].strip()
                    or len(claim["text"]) > 600 or not isinstance(claim["citations"], list)
                    or not 1 <= len(claim["citations"]) <= 5):
                raise InvalidModelOutput("结论缺少引用。")
            citations = []
            for item in claim["citations"]:
                citation = _object(item, {"id", "quote"})
                identifier, quote = citation["id"], citation["quote"]
                if (not isinstance(identifier, str) or identifier not in allowed
                        or not isinstance(quote, str) or not 2 <= len(quote.strip()) <= 1000
                        or quote not in allowed[identifier]):
                    raise InvalidModelOutput("引用必须逐字存在于对应证据。")
                citations.append({"id": identifier, "quote": quote})
            claims.append({"text": claim["text"].strip(), "citations": citations})
        return claims
