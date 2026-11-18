/**
 * 静态版页面逻辑：全部在浏览器内完成，不向本项目自己的服务器发送任何数据。
 * 只有主动选择 AI 问答时，问题与检索片段才会直接发给你填写的模型服务商。
 */
import { loadDocuments, Retriever, isSupportedTextFile } from "./core.js";
import { ModelError, generateAnswer } from "./model.js";

const $ = (selector) => document.querySelector(selector);

const dom = {
  status: $("#status"),
  alerts: $("#alerts"),
  workspace: $("#workspace"),
  sourceRadios: [...document.querySelectorAll('input[name="source-mode"]')],
  corpusHint: $("#corpus-hint"),
  uploadArea: $("#upload-area"),
  fileInput: $("#file-input"),
  fileList: $("#file-list"),
  modeRadios: [...document.querySelectorAll('input[name="run-mode"]')],
  aiSettings: $("#ai-settings"),
  apiKey: $("#api-key"),
  baseUrl: $("#base-url"),
  modelName: $("#model-name"),
  consent: $("#consent"),
  topK: $("#top-k"),
  topKValue: $("#top-k-value"),
  minScore: $("#min-score"),
  minScoreValue: $("#min-score-value"),
  clearButton: $("#clear-button"),
  form: $("#question-form"),
  question: $("#question"),
  submit: $('button[type="submit"]'),
  sampleQuestions: $("#sample-questions"),
  result: $("#result"),
  feedback: $("#feedback"),
};

const state = {
  corpora: new Map(),
  retriever: null,
  corpusHash: null,
  corpusNames: [],
  chunkCount: 0,
  result: null,
  resultContext: null,
  feedback: [],
  busy: false,
};

function h(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value !== null && value !== undefined && value !== false) {
      node.setAttribute(key, value === true ? "" : String(value));
    }
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

function formatBytes(size) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KiB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MiB`;
}

function setStatus(message, tone = "") {
  dom.status.textContent = message;
  if (tone) dom.status.dataset.tone = tone;
  else delete dom.status.dataset.tone;
}

function clearAlerts() {
  dom.alerts.replaceChildren();
}

function showAlert(kind, message) {
  const node = h("p", { class: `alert alert--${kind}`, role: "alert", text: message });
  dom.alerts.append(node);
}

function setBusy(message) {
  state.busy = Boolean(message);
  dom.workspace.setAttribute("aria-busy", state.busy ? "true" : "false");
  dom.submit.disabled = state.busy;
  dom.clearButton.disabled = state.busy;
  if (message) setStatus(message, "busy");
}

const paint = () => new Promise((resolve) => { setTimeout(resolve, 20); });

function selectedSource() {
  return dom.sourceRadios.find((radio) => radio.checked)?.value || "sample_notes";
}

function selectedMode() {
  return dom.modeRadios.find((radio) => radio.checked)?.value || "search";
}

function modeLabel() {
  return selectedMode() === "ai" ? "AI 问答（需要 API）" : "只检索原文（无需 API）";
}

function currentContext() {
  return JSON.stringify([
    state.corpusHash,
    selectedMode(),
    dom.topK.value,
    dom.minScore.value,
    dom.baseUrl.value,
    dom.modelName.value,
    dom.apiKey.value.length > 0,
  ]);
}

function syncContext() {
  if (state.result && state.resultContext !== currentContext()) {
    state.result = null;
    renderResult();
    renderFeedback();
  }
}

async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** 打开 file:// 时 secure context 不成立，退化哈希只用于判断资料是否变化。 */
function fallbackHash(bytes) {
  let hash = 2166136261;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

async function digest(bytes) {
  if (globalThis.crypto?.subtle) return sha256Hex(bytes);
  return fallbackHash(bytes);
}

async function corpusHash(records) {
  const encoder = new TextEncoder();
  const parts = [];
  for (const record of records) {
    parts.push(`${record.name}\u0000${await digest(record.bytes)}`);
  }
  return digest(encoder.encode(parts.join("\n")));
}

function updateStatus() {
  if (!state.retriever) {
    setStatus("请选择内置语料，或上传自己的笔记。");
    return;
  }
  setStatus(`已载入 ${state.corpusNames.length} 份笔记，共 ${state.chunkCount} 个段落；检索只在这个浏览器里完成。`);
}

function clearCorpus() {
  state.retriever = null;
  state.corpusHash = null;
  state.corpusNames = [];
  state.chunkCount = 0;
  state.result = null;
  state.resultContext = null;
  renderResult();
  renderFeedback();
}

async function toRecords(fileList) {
  const files = [...fileList];
  const unsupported = files.find((file) => !isSupportedTextFile(file.name));
  if (unsupported) {
    throw new Error(
      `静态版只支持 UTF-8 的 .txt / .md；${unsupported.name} 需要服务端解析，请使用仓库里的 Streamlit 或 Docker 版本。`,
    );
  }
  return Promise.all(files.map(async (file) => ({
    name: file.name,
    bytes: new Uint8Array(await file.arrayBuffer()),
  })));
}

async function applyCorpus(records) {
  const hash = await corpusHash(records);
  if (state.retriever && state.corpusHash === hash) {
    updateStatus();
    return;
  }
  setBusy("正在建立索引…");
  try {
    await paint();
    const chunks = loadDocuments(records);
    // 新索引完整生成后再替换；失败导入不会破坏原有可用资料。
    const retriever = new Retriever(chunks);
    state.retriever = retriever;
    state.corpusHash = hash;
    state.corpusNames = records.map((record) => record.name);
    state.chunkCount = chunks.length;
    state.result = null;
    state.resultContext = null;
    clearAlerts();
    renderResult();
    renderFeedback();
  } catch (error) {
    showAlert("error", error.message);
  } finally {
    setBusy(null);
    updateStatus();
  }
}

async function loadCorpusFromUI() {
  clearAlerts();
  const source = selectedSource();
  if (source !== "upload") {
    const corpus = state.corpora.get(source);
    if (!corpus) {
      clearCorpus();
      showAlert("warn", "内置语料没有加载成功，请上传自己的 .md / .txt 笔记。");
      return;
    }
    await applyCorpus(corpus.files);
    return;
  }
  const files = dom.fileInput.files;
  if (!files || files.length === 0) {
    clearCorpus();
    updateStatus();
    return;
  }
  try {
    await applyCorpus(await toRecords(files));
  } catch (error) {
    showAlert("error", error.message);
  }
}

function renderResult() {
  dom.result.replaceChildren();
  const result = state.result;
  if (!result) {
    dom.result.hidden = true;
    return;
  }
  dom.result.hidden = false;
  dom.result.append(
    h("h2", { class: "result__title", text: "本次结果" }),
    h("p", { class: "result__question", text: result.question }),
  );

  if (result.error) {
    dom.result.append(h("p", { class: "alert alert--error", role: "alert", text: result.error }));
  } else if (result.answer) {
    dom.result.append(h("p", { class: "answer", text: result.answer.answer }));
    if (result.answer.citations.length > 0) {
      dom.result.append(h("p", {
        class: "citations",
        text: `回答引用：${result.answer.citations.join("、")}`,
      }));
    }
    dom.result.append(h("p", { class: "citations", text: "引用编号已检查；原文是否真正支持结论仍需你核对。" }));
  } else if (result.hits.length === 0) {
    dom.result.append(h("p", {
      class: "alert alert--info",
      text: "没有找到达到阈值的段落。请调整问题、补充笔记或降低阈值。",
    }));
  } else {
    dom.result.append(h("p", {
      class: "alert alert--info",
      text: "当前为检索模式：下面显示匹配原文，没有调用大模型。",
    }));
  }

  for (const hit of result.hits) {
    const { chunk } = hit;
    const label = `[${chunk.id}] ${chunk.source} · ${chunk.section} · 行 ${chunk.startLine}–${chunk.endLine} · 匹配 ${hit.score.toFixed(3)}`;
    dom.result.append(h("details", { class: "hit", open: true }, [
      h("summary", { text: label }),
      h("pre", { text: chunk.text }),
    ]));
  }

  dom.result.append(h("p", { class: "result__meta", text: `本次耗时：${result.seconds.toFixed(2)} 秒` }));
}

function renderFeedback() {
  dom.feedback.replaceChildren();
  const hasResult = Boolean(state.result && !state.result.error);
  if (!hasResult && state.feedback.length === 0) {
    dom.feedback.hidden = true;
    return;
  }
  dom.feedback.hidden = false;

  if (hasResult) {
    const ratingName = `rating-${Math.random().toString(36).slice(2, 8)}`;
    const commentId = `comment-${Math.random().toString(36).slice(2, 8)}`;
    const form = h("div", { class: "feedback" }, [
      h("h3", { text: "这次结果有帮助吗？" }),
      h("div", { class: "feedback__row" }, [
        h("label", { class: "choice" }, [
          h("input", { type: "radio", name: ratingName, value: "有帮助" }),
          h("span", { text: "有帮助" }),
        ]),
        h("label", { class: "choice" }, [
          h("input", { type: "radio", name: ratingName, value: "没帮助" }),
          h("span", { text: "没帮助" }),
        ]),
      ]),
      h("label", { class: "field__label", for: commentId, text: "原因（可选）" }),
      h("input", { type: "text", id: commentId, maxlength: 300, placeholder: "例如：引用的段落没有直接回答这个问题" }),
      h("div", { class: "feedback__actions" }, [
        h("button", {
          class: "button button--small",
          type: "button",
          text: "记录反馈",
          onclick: () => {
            const rating = form.querySelector("input[type=radio]:checked")?.value;
            if (!rating) {
              showAlert("warn", "请选择“有帮助”或“没帮助”。");
              return;
            }
            const comment = form.querySelector("input[type=text]").value;
            state.feedback.push({
              time_utc: new Date().toISOString(),
              question: state.result.question,
              mode: state.result.mode,
              corpus_hash: state.result.corpusHash,
              answer: state.result.answer,
              error: state.result.error,
              retrieved_ids: state.result.hits.map((hit) => hit.chunk.id),
              seconds: state.result.seconds,
              rating,
              comment,
            });
            clearAlerts();
            showAlert("info", "已记录到当前会话，可在下方下载。");
            renderFeedback();
          },
        }),
      ]),
    ]);
    dom.feedback.append(form);
  }

  if (state.feedback.length > 0) {
    dom.feedback.append(h("div", { class: "feedback__actions", style: "margin-top: 12px" }, [
      h("button", {
        class: "button button--ghost button--small",
        type: "button",
        text: "下载本次会话反馈 JSON",
        onclick: downloadFeedback,
      }),
    ]));
    dom.feedback.append(h("p", {
      class: "hint",
      text: "反馈只保存在当前标签页；关闭或刷新页面后会丢失，请及时下载。",
    }));
  }
}

function downloadFeedback() {
  const blob = new Blob([JSON.stringify(state.feedback, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = h("a", { href: url, download: "studylens_feedback.json" });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function handleSubmit(event) {
  event.preventDefault();
  if (state.busy) return;
  clearAlerts();
  const question = dom.question.value.trim();
  if (!question) {
    showAlert("warn", "请先输入问题。");
    return;
  }
  if (!state.retriever) {
    showAlert("error", "请先载入笔记。");
    return;
  }
  if (selectedMode() === "ai" && !dom.consent.checked) {
    showAlert("warn", "使用 AI 问答前，请确认发送资料片段。");
    return;
  }
  const topK = Number(dom.topK.value);
  const minScore = Number(dom.minScore.value);
  const started = performance.now();
  setBusy("正在检索原文…");
  try {
    await paint();
    const hits = state.retriever.search(question, topK, minScore);
    let answer = null;
    let error = null;
    if (selectedMode() === "ai") {
      setBusy("正在根据原文组织回答…");
      try {
        answer = await generateAnswer(question, hits, {
          apiKey: dom.apiKey.value,
          baseUrl: dom.baseUrl.value,
          model: dom.modelName.value,
        });
      } catch (issue) {
        error = issue instanceof ModelError ? issue.message : "模型请求失败，请稍后重试。";
      }
    }
    state.result = {
      question,
      hits,
      answer,
      error,
      seconds: (performance.now() - started) / 1000,
      mode: modeLabel(),
      corpusHash: state.corpusHash,
    };
    state.resultContext = currentContext();
  } finally {
    setBusy(null);
    updateStatus();
  }
  renderResult();
  renderFeedback();
}

function renderChips() {
  dom.sampleQuestions.replaceChildren();
  const corpus = state.corpora.get(selectedSource());
  const questions = corpus?.questions || [];
  for (const question of questions) {
    dom.sampleQuestions.append(h("button", {
      class: "chip",
      type: "button",
      text: question,
      onclick: () => {
        dom.question.value = question;
        dom.question.focus();
      },
    }));
  }
}

function syncSourceControls() {
  const source = selectedSource();
  const upload = source === "upload";
  dom.uploadArea.hidden = !upload;
  dom.fileInput.disabled = !upload;
  dom.corpusHint.textContent = upload
    ? "只接受 UTF-8 的 .txt / .md；文件不会离开这个浏览器。"
    : (state.corpora.get(source)?.description || "正在准备内置语料…");
}

function bindEvents() {
  for (const radio of dom.sourceRadios) {
    radio.addEventListener("change", async () => {
      syncSourceControls();
      renderChips();
      await loadCorpusFromUI();
    });
  }
  dom.fileInput.addEventListener("change", async () => {
    const files = [...dom.fileInput.files];
    dom.fileList.textContent = files.length === 0
      ? "尚未选择文件。"
      : files.map((file) => `${file.name}（${formatBytes(file.size)}）`).join("、");
    await loadCorpusFromUI();
  });
  for (const radio of dom.modeRadios) {
    radio.addEventListener("change", () => {
      dom.aiSettings.hidden = selectedMode() !== "ai";
      syncContext();
    });
  }
  for (const input of [dom.topK, dom.minScore, dom.baseUrl, dom.modelName, dom.apiKey]) {
    input.addEventListener("input", syncContext);
  }
  dom.topK.addEventListener("input", () => { dom.topKValue.textContent = dom.topK.value; });
  dom.minScore.addEventListener("input", () => { dom.minScoreValue.textContent = dom.minScore.value; });
  dom.consent.addEventListener("change", syncContext);
  dom.form.addEventListener("submit", handleSubmit);
  dom.clearButton.addEventListener("click", () => {
    state.result = null;
    state.feedback = [];
    clearAlerts();
    renderResult();
    renderFeedback();
  });
}

async function loadCorpora() {
  const encoder = new TextEncoder();
  try {
    const response = await fetch("samples.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    for (const item of data.corpora || []) {
      if (typeof item?.id !== "string" || !Array.isArray(item.files)) continue;
      const files = item.files
        .filter((file) => typeof file?.name === "string" && typeof file?.text === "string")
        .map((file) => ({ name: file.name, bytes: encoder.encode(file.text) }));
      if (files.length === 0) continue;
      state.corpora.set(item.id, {
        label: item.label || item.id,
        description: item.description || "",
        questions: Array.isArray(item.questions) ? item.questions : [],
        files,
      });
    }
    if (state.corpora.size === 0) throw new Error("内置语料为空");
    // 配置里没有的语料（例如旧版构建）就不要出现在选择项里，避免点开是空的。
    for (const radio of dom.sourceRadios) {
      if (radio.value !== "upload" && !state.corpora.has(radio.value)) {
        radio.disabled = true;
        radio.checked = false;
      }
    }
    if (!dom.sourceRadios.some((radio) => radio.checked)) {
      dom.sourceRadios.find((radio) => radio.value === "upload").checked = true;
    }
  } catch {
    state.corpora.clear();
    for (const radio of dom.sourceRadios) {
      if (radio.value !== "upload") {
        radio.disabled = true;
        radio.checked = false;
      }
    }
    dom.sourceRadios.find((radio) => radio.value === "upload").checked = true;
    showAlert("warn", "内置语料没有加载成功（本地预览时请先运行 scripts/build_site.py）。请上传自己的 .md / .txt 笔记。");
  }
}

async function init() {
  dom.aiSettings.hidden = true;
  bindEvents();
  await loadCorpora();
  syncSourceControls();
  renderChips();
  await loadCorpusFromUI();
}

init();
