/**
 * 校验静态版的模型接口限制与回答审核；全部使用本地桩，不发出真实请求。
 */
import test from "node:test";
import assert from "node:assert/strict";

import {
  ModelError,
  canonicalModelBaseUrl,
  generateAnswer,
  validateAnswer,
  validateModelBaseUrl,
} from "../../web/assets/model.js";

const HIT = { chunk: { id: "D1-1", text: "SC 与 SCL 的区别在于是否保留多条候选路径。" }, score: 0.5 };
const CONFIG = { apiKey: "sk-test", baseUrl: "https://api.deepseek.com", model: "deepseek-chat" };

function stubFetch(handler) {
  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    return handler(url, init);
  };
  return {
    calls,
    restore() {
      globalThis.fetch = original;
    },
  };
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function answerBody(content, finish = "stop") {
  return jsonResponse({
    choices: [{ finish_reason: finish, message: { content: JSON.stringify(content) } }],
  });
}

test("默认接口地址会被规范化", () => {
  assert.equal(canonicalModelBaseUrl("https://api.deepseek.com/"), "https://api.deepseek.com");
  assert.equal(canonicalModelBaseUrl("https://api.openai.com/v1/"), "https://api.openai.com/v1");
  assert.equal(canonicalModelBaseUrl("  https://API.DeepSeek.com  "), "https://api.deepseek.com");
  assert.equal(validateModelBaseUrl("https://api.deepseek.com"), "https://api.deepseek.com");
});

test("拒绝本地地址、端口、凭据和可疑路径", () => {
  const cases = [
    ["http://api.deepseek.com", /HTTPS 公共域名/],
    ["https://127.0.0.1", /IP 地址/],
    ["https://api.deepseek.com:8443", /HTTPS 公共域名/],
    ["https://api.deepseek.com:443", /HTTPS 公共域名/],
    ["https://user:pass@api.deepseek.com", /HTTPS 公共域名/],
    ["https://api.deepseek.com/?token=1", /查询参数/],
    ["https://api.deepseek.com/#x", /查询参数或片段/],
    ["https://localhost/v1", /公共服务商域名/],
    ["https://api.internal/v1", /公共服务商域名/],
    ["https://api.deepseek.com/v1/../x", /路径不正确/],
    ["https://api.deepseek.com//v1", /路径不正确/],
  ];
  for (const [value, pattern] of cases) {
    assert.throws(() => canonicalModelBaseUrl(value), pattern, value);
  }
});

test("只允许白名单内的服务商", () => {
  assert.throws(() => validateModelBaseUrl("https://example.com"), /未获允许/);
  assert.throws(() => validateModelBaseUrl("https://api.deepseek.com/extra"), /未获允许/);
});

test("回答审核规则与 Python 基线一致", () => {
  const allowed = new Set(["D1-1", "D1-2"]);
  assert.deepEqual(validateAnswer({ answer: " 答案 ", citations: ["D1-1", "D1-1"], insufficient: false }, allowed), {
    answer: "答案",
    citations: ["D1-1"],
    insufficient: false,
  });
  assert.deepEqual(validateAnswer({ answer: "资料不足", citations: [], insufficient: true }, allowed), {
    answer: "当前资料不足以回答，请补充笔记或缩小问题范围。",
    citations: [],
    insufficient: true,
  });
  assert.throws(() => validateAnswer({ answer: "答案", citations: ["D9-9"], insufficient: false }, allowed), /不存在的引用/);
  assert.throws(() => validateAnswer({ answer: "答案", citations: [], insufficient: false }, allowed), /没有引用依据/);
  assert.throws(() => validateAnswer({ answer: "答案", citations: ["D1-1"], insufficient: "false" }, allowed), /字段类型/);
  assert.throws(() => validateAnswer(null, allowed), /结构不正确/);
});

test("没有命中段落时不调用模型", async () => {
  const stub = stubFetch(() => {
    throw new Error("不应该发出请求");
  });
  try {
    const result = await generateAnswer("问题", [], CONFIG);
    assert.equal(result.insufficient, true);
    assert.equal(stub.calls.length, 0);
  } finally {
    stub.restore();
  }
});

test("缺少密钥时给出明确提示且不发出请求", async () => {
  const stub = stubFetch(() => {
    throw new Error("不应该发出请求");
  });
  try {
    await assert.rejects(
      () => generateAnswer("问题", [HIT], { ...CONFIG, apiKey: "  " }),
      (error) => error instanceof ModelError && /API Key 和模型名称/.test(error.message),
    );
    assert.equal(stub.calls.length, 0);
  } finally {
    stub.restore();
  }
});

test("请求结构与回答校验通过", async () => {
  const stub = stubFetch(() => answerBody({ answer: "按原文回答。", citations: ["D1-1"], insufficient: false }));
  try {
    const result = await generateAnswer("SC 和 SCL 有什么区别？", [HIT], CONFIG);
    assert.deepEqual(result, { answer: "按原文回答。", citations: ["D1-1"], insufficient: false });
    const [call] = stub.calls;
    assert.equal(call.url, "https://api.deepseek.com/chat/completions");
    assert.equal(call.init.method, "POST");
    assert.equal(call.init.redirect, "error");
    assert.equal(call.init.headers.Authorization, "Bearer sk-test");
    const payload = JSON.parse(call.init.body);
    assert.equal(payload.model, "deepseek-chat");
    assert.deepEqual(payload.response_format, { type: "json_object" });
    assert.equal(payload.messages[0].role, "system");
    assert.deepEqual(JSON.parse(payload.messages[1].content), {
      question: "SC 和 SCL 有什么区别？",
      evidence: [{ id: "D1-1", text: HIT.chunk.text }],
    });
  } finally {
    stub.restore();
  }
});

test("服务商错误、截断和不可信引用都会被拦截", async () => {
  const scenarios = [
    [() => jsonResponse({ error: "boom" }, 500), /HTTP 500/],
    [() => answerBody({ answer: "长答案", citations: ["D1-1"], insufficient: false }, "length"), /被截断/],
    [() => answerBody({ answer: "编造", citations: ["D9-9"], insufficient: false }), /不存在的引用/],
    [() => new Response("<html>", { status: 200 }), /预期的 JSON/],
  ];
  for (const [handler, pattern] of scenarios) {
    const stub = stubFetch(handler);
    try {
      await assert.rejects(() => generateAnswer("问题", [HIT], CONFIG), pattern);
    } finally {
      stub.restore();
    }
  }
});

test("网络错误给出跨域与网络提示", async () => {
  const stub = stubFetch(() => {
    throw new TypeError("Failed to fetch");
  });
  try {
    await assert.rejects(() => generateAnswer("问题", [HIT], CONFIG), /跨域（CORS）/);
  } finally {
    stub.restore();
  }
});
