"""知识库网页：只有部署者配置的服务地址可接收访问凭据和资料。"""
import hashlib
import os
from urllib.parse import urlsplit
import requests
import streamlit as st

from studylens.limits import UPLOAD_HELP, validate_uploads


def configured_api_url():
    value = os.getenv("STUDYLENS_API_URL", "")
    if not value:
        try:
            value = st.secrets.get("STUDYLENS_API_URL", "")
        except (FileNotFoundError, KeyError):
            pass
    if not value:
        return ""
    parsed = urlsplit(value)
    # 私密凭据通过HTTPS传送；容器/本机部署可使用固定的内部地址。
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "api"})):
        raise ValueError("知识库服务地址配置不正确，请联系维护者。")
    return value.rstrip("/")


def call_api(base_url, key, method, path, **kwargs):
    headers = {"Authorization": f"Bearer {key}", **kwargs.pop("headers", {})}
    try:
        # 写入请求不自动重试，防止重复Embedding计费；由稳定source_id保证手动重试幂等。
        response = requests.request(method, base_url + path, headers=headers, timeout=(10, 240),
                                    allow_redirects=False, **kwargs)
        if response.status_code >= 300:
            try:
                payload = response.json()
                detail = payload.get("detail") if isinstance(payload, dict) else None
            except ValueError:
                detail = None
            safe = detail if isinstance(detail, str) else "服务暂时不可用，请稍后重试。"
            raise ValueError(f"{safe}（HTTP {response.status_code}）")
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("服务返回格式不正确。")
        return data
    except requests.RequestException as exc:
        raise ValueError("无法完成知识库请求，请检查服务状态后重试。") from exc


def render_knowledge_workspace():
    st.subheader("我的知识库")
    st.caption("资料可持续更新；每次回答会先查找资料，再筛选依据并核对引用。")
    try:
        base = configured_api_url()
    except ValueError as exc:
        st.error(str(exc))
        return
    if not base:
        st.info("知识库服务尚未连接。维护者配置完成后即可使用；当前可选择“临时笔记”体验。")
        return
    with st.sidebar:
        key = st.text_input("知识库访问凭据", type="password", key="kb_access_key")
        st.caption("只读凭据用于问答；管理凭据可以导入、替换或删除资料。")
    scope = hashlib.sha256((base + key).encode("utf-8")).hexdigest()
    if st.session_state.get("kb_scope") != scope:
        for name in ("kb_documents", "kb_answer"):
            st.session_state.pop(name, None)
        st.session_state["kb_scope"] = scope
    if not key:
        st.info("请在左侧填写维护者提供的访问凭据。")
        return

    def request(method, path, **kwargs):
        return call_api(base, key, method, path, **kwargs)

    if st.button("连接 / 刷新资料"):
        try:
            st.session_state["kb_documents"] = request("GET", "/v1/documents")["documents"]
            st.success("知识库已连接。")
        except ValueError as exc:
            st.error(str(exc))
    docs = st.session_state.get("kb_documents", [])
    if docs:
        with st.expander(f"当前资料（{len(docs)}份）"):
            for doc in docs:
                st.write(f"{doc['name']} · 编号 {doc['source_id']} · 更新于 {doc.get('updated_at', '未知')}")
                for warning in doc.get("metadata", {}).get("warnings", []):
                    st.warning(warning)

    with st.expander("管理资料：新增、替换、删除"):
        st.caption("同一份资料请使用同一个编号；重新导入会替换旧版本。")
        with st.form("kb_upload_form"):
            source = st.text_input("资料编号", placeholder="polar-notes", max_chars=80)
            upload = st.file_uploader("选择资料", type=["md", "txt", "pdf", "docx", "png", "jpg", "jpeg"])
            st.caption(UPLOAD_HELP)
            consent = st.checkbox("同意将提取的资料文字发送给配置的向量模型，并保存到知识库")
            submit_upload = st.form_submit_button("导入 / 替换资料")
        if submit_upload:
            try:
                if not upload or not consent:
                    raise ValueError("请选择资料并确认处理方式。")
                import re
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", source):
                    raise ValueError("资料编号请使用字母、数字、短横线或下划线。")
                validate_uploads([(upload.name, upload.getvalue())])
                with st.spinner("正在提取并更新知识库…"):
                    result = request("PUT", f"/v1/documents/{source}", params={"filename": upload.name},
                                     data=upload.getvalue(), headers={"X-External-Processing": "true",
                                                                     "Content-Type": "application/octet-stream"})
                st.session_state.pop("kb_answer", None)
                st.session_state.pop("kb_documents", None)
                st.success("资料已处理完成。点击“连接 / 刷新资料”查看最新版本。")
                for warning in result.get("warnings", []):
                    st.warning(warning)
            except ValueError as exc:
                st.error(str(exc))
        with st.form("kb_delete_form"):
            delete_id = st.text_input("要删除的资料编号", max_chars=80)
            confirmed = st.checkbox("确认从知识库删除这份资料")
            delete_pressed = st.form_submit_button("删除资料")
        if delete_pressed:
            try:
                import re
                if not confirmed or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", delete_id):
                    raise ValueError("请填写有效资料编号并确认删除。")
                request("DELETE", f"/v1/documents/{delete_id}")
                st.session_state.pop("kb_answer", None)
                st.session_state.pop("kb_documents", None)
                st.success("删除请求已完成。刷新资料列表可核对结果。")
            except ValueError as exc:
                st.error(str(exc))

    with st.form("kb_question_form"):
        question = st.text_input("向知识库提问", max_chars=500)
        external = st.checkbox("同意将问题及相关片段发送给配置的向量、重排和问答模型")
        submit_question = st.form_submit_button("查找依据并回答")
    if submit_question:
        try:
            st.session_state.pop("kb_answer", None)
            if not external or not question.strip():
                raise ValueError("请输入问题并确认处理方式。")
            with st.spinner("正在查找、筛选并核对依据…"):
                st.session_state["kb_answer"] = request("POST", "/v1/query", json={
                    "question": question, "consent_external": True})
        except ValueError as exc:
            st.error(str(exc))
    answer = st.session_state.get("kb_answer")
    if answer:
        st.write(answer["answer"])
        st.caption("这是本次查询时的资料快照。引用核验可减少错误，重要结论仍需对照原文。")
        for item in answer.get("evidence", []):
            if item.get("id") in answer.get("citations", []):
                with st.expander(f"依据：{item['source']} · {item['section']}"):
                    st.code(item["text"], language=None)
        with st.expander("查看本次处理步骤"):
            st.json(answer.get("trace", []))
        st.download_button("下载本次回答与依据", __import__("json").dumps(answer, ensure_ascii=False, indent=2),
                           file_name="studylens_answer.json", mime="application/json")
