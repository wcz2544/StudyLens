"""网页入口：管理会话与交互，核心处理委托给 studylens 模块。"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
import streamlit as st
from studylens.documents import load_documents
from studylens.retrieval import Retriever
from studylens.llm import ModelConfig, ModelError, generate_answer

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", encoding="utf-8")
st.set_page_config(page_title="StudyLens 笔记问答", page_icon="📚", layout="wide")


def setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return default


st.title("StudyLens · 有出处的笔记问答")
st.caption("上传自己的 UTF-8 笔记，查看相关原文，再选择是否让 AI 组织回答。")
with st.sidebar:
    st.header("资料与设置")
    use_sample = st.checkbox("使用内置示例笔记", value=True)
    uploaded = st.file_uploader("或上传 TXT / Markdown", type=["txt", "md"],
                                accept_multiple_files=True, disabled=use_sample)
    st.caption("最多 10 份；单份 512 KB；总计 2 MB。")
    mode = st.radio("运行模式", ["只检索原文（无需 API）", "AI 问答（需要 API）"])
    top_k = st.slider("最多参考段落数", 1, 5, 3)
    min_score = st.slider("最低匹配分数", 0.0, 0.5, 0.08, 0.01)
    st.caption("匹配分数不是正确率。提高阈值会过滤更多结果，也可能漏掉答案。")
    consent = False
    if mode.startswith("AI"):
        consent = st.checkbox("同意将问题及检索到的笔记片段发送至配置的模型服务商")
    if st.button("清空本次问答与反馈"):
        st.session_state.pop("result", None)
        st.session_state["feedback"] = []

files = ([(p.name, p.read_bytes()) for p in sorted((ROOT / "data" / "sample_notes").glob("*.md"))]
         if use_sample else [(f.name, f.getvalue()) for f in (uploaded or [])])
if not files:
    # 删除资料时同时清除旧索引和旧答案，避免用户误以为旧内容属于新资料。
    for key in ("index", "corpus_hash", "result", "result_context"):
        st.session_state.pop(key, None)
    st.info("请上传笔记，或勾选“使用内置示例笔记”。")
    st.stop()

# 只在当前用户会话中保留索引，不使用跨用户共享的全局缓存。
fingerprint = hashlib.sha256()
for name, data in files:
    fingerprint.update(name.encode("utf-8") + b"\0")
    fingerprint.update(hashlib.sha256(data).digest())
corpus_hash = fingerprint.hexdigest()
if st.session_state.get("corpus_hash") != corpus_hash:
    st.session_state.pop("result", None)
    st.session_state.pop("index", None)
    try:
        st.session_state["index"] = Retriever(load_documents(files))
        st.session_state["corpus_hash"] = corpus_hash
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

index = st.session_state["index"]
st.caption(f"已载入 {len(files)} 份笔记，共 {len(index.chunks)} 个段落。")
context = (corpus_hash, mode, top_k, min_score)
if st.session_state.get("result_context") != context:
    st.session_state.pop("result", None)
    st.session_state["result_context"] = context

with st.form("question_form"):
    question = st.text_input("你想了解什么？", placeholder="SC 和 SCL 译码有什么区别？", max_chars=500)
    submitted = st.form_submit_button("查找并回答")

if submitted:
    st.session_state.pop("result", None)
    if not question.strip():
        st.warning("请先输入问题。")
    elif mode.startswith("AI") and not consent:
        st.warning("使用 AI 问答前，请确认发送资料片段。")
    else:
        started = perf_counter()
        hits = index.search(question, top_k, min_score)
        answer = None
        error = None
        if mode.startswith("AI"):
            config = ModelConfig(setting("LLM_API_KEY"),
                                 setting("LLM_BASE_URL", "https://api.deepseek.com"),
                                 setting("LLM_MODEL"))
            try:
                with st.spinner("正在根据原文组织回答…"):
                    answer = generate_answer(question, hits, config)
            except ModelError as exc:
                error = str(exc)
        st.session_state["result"] = {
            "question": question.strip(), "hits": hits, "answer": answer,
            "error": error, "seconds": round(perf_counter() - started, 3),
            "mode": mode, "corpus_hash": corpus_hash,
        }

result = st.session_state.get("result")
if result:
    st.subheader("本次结果")
    st.write(result["question"])
    if result["error"]:
        st.error(result["error"])
    elif result["answer"]:
        st.write(result["answer"]["answer"])
        if result["answer"]["citations"]:
            st.caption("回答引用：" + "、".join(result["answer"]["citations"]))
        st.caption("引用编号已检查；原文是否真正支持结论仍需你核对。")
    elif not result["hits"]:
        st.info("没有找到达到阈值的段落。请调整问题、补充笔记或降低阈值。")
    else:
        st.info("当前为检索模式：下面显示匹配原文，没有调用大模型。")
    for hit in result["hits"]:
        chunk = hit.chunk
        label = (f"[{chunk.id}] {chunk.source} · {chunk.section} · "
                 f"行 {chunk.start_line}–{chunk.end_line} · 匹配 {hit.score:.3f}")
        with st.expander(label, expanded=True):
            st.code(chunk.text, language=None)
    st.caption(f"本次耗时：{result['seconds']} 秒")
    with st.form("feedback_form"):
        rating = st.radio("这次结果有帮助吗？", ["有帮助", "没帮助"], horizontal=True)
        comment = st.text_input("原因（可选）", max_chars=300)
        if st.form_submit_button("记录反馈"):
            st.session_state.setdefault("feedback", []).append({
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "question": result["question"], "mode": result["mode"],
                "corpus_hash": result["corpus_hash"],
                "answer": result["answer"], "error": result["error"],
                "retrieved_ids": [h.chunk.id for h in result["hits"]],
                "seconds": result["seconds"], "rating": rating, "comment": comment,
            })
            st.success("已记录到当前会话，可在下方下载。")

feedback = st.session_state.get("feedback", [])
if feedback:
    st.download_button("下载本次会话反馈 JSON", json.dumps(feedback, ensure_ascii=False, indent=2),
                       file_name="studylens_feedback.json", mime="application/json")
    st.caption("反馈只保存在当前会话；关闭或刷新导致会话重建后可能丢失，请及时下载。")
