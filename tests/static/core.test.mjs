/**
 * 校验浏览器实现与 Python 基线一致：分段结果、检索排序与分数。
 * 对照数据由 scripts/export_web_fixtures.py 生成，CI 会检查它是否过期。
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadDocuments, Retriever, parseDocument } from "../../web/assets/core.js";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, "fixtures", "retrieval.json"), "utf8"));
const encoder = new TextEncoder();

function recordsOf(files) {
  return files.map((file) => ({ name: file.name, bytes: encoder.encode(file.text) }));
}

function viewChunk(chunk) {
  return {
    id: chunk.id,
    source: chunk.source,
    section: chunk.section,
    startLine: chunk.startLine,
    endLine: chunk.endLine,
    text: chunk.text,
  };
}

test("对照数据包含可用的样例和查询", () => {
  assert.ok(fixture.cases.length >= 2);
  assert.ok(fixture.cases.every((item) => item.chunks.length > 0 && item.queries.length > 0));
});

for (const item of fixture.cases) {
  test(`分段与 Python 基线一致：${item.name}`, () => {
    const chunks = loadDocuments(recordsOf(item.files)).map(viewChunk);
    assert.deepEqual(chunks, item.chunks.map((chunk) => ({
      id: chunk.id,
      source: chunk.source,
      section: chunk.section,
      startLine: chunk.start_line,
      endLine: chunk.end_line,
      text: chunk.text,
    })));
  });

  test(`检索排序与分数与 Python 基线一致：${item.name}`, () => {
    const retriever = new Retriever(loadDocuments(recordsOf(item.files)));
    for (const query of item.queries) {
      const hits = retriever.search(query.question, query.top_k, query.min_score);
      assert.deepEqual(
        hits.map((hit) => hit.chunk.id),
        query.hits.map((hit) => hit.id),
        `问题「${query.question}」的命中段落不一致`,
      );
      hits.forEach((hit, index) => {
        const expected = query.hits[index].score;
        assert.ok(
          Math.abs(hit.score - expected) <= 1e-9,
          `问题「${query.question}」第 ${index + 1} 段分数 ${hit.score} 与基线 ${expected} 不一致`,
        );
      });
    }
  });
}

test("阈值与 Top-K 的过滤顺序与基线一致", () => {
  const retriever = new Retriever(loadDocuments(recordsOf(fixture.cases[0].files)));
  assert.deepEqual(retriever.search("SC 和 SCL 译码有什么区别？", 3, 0.99), []);
  // 过滤发生在取 Top-K 之后：即使放宽阈值，结果数量也不会超过 Top-K。
  assert.ok(retriever.search("SC 和 SCL 译码有什么区别？", 1, 0).length <= 1);
});

test("拒绝不支持的格式、超限文件和非法编码", () => {
  assert.throws(() => parseDocument("笔记.pdf", encoder.encode("文字"), "D1"), /仅支持 \.txt 和 \.md/);
  assert.throws(
    () => parseDocument("大文件.md", new Uint8Array(512 * 1024 + 1).fill(97), "D1"),
    /512 KB/,
  );
  assert.throws(() => parseDocument("坏的.md", Uint8Array.from([0xff, 0xfe]), "D1"), /UTF-8/);
  assert.throws(() => parseDocument("空.md", encoder.encode("  \n\t"), "D1"), /为空或不是正常文本/);
  assert.throws(() => parseDocument("空.md", encoder.encode("a\u0000b"), "D1"), /为空或不是正常文本/);
});

test("文件数量与总大小限制与 Python 基线一致", () => {
  const many = Array.from({ length: 11 }, (_, index) => ({
    name: `笔记${index}.md`,
    bytes: encoder.encode("内容"),
  }));
  assert.throws(() => loadDocuments(many), /1～10 份笔记/);
  assert.throws(() => loadDocuments([]), /1～10 份笔记/);

  const big = Array.from({ length: 5 }, (_, index) => ({
    name: `大${index}.md`,
    bytes: new Uint8Array(500 * 1024).fill(97),
  }));
  assert.throws(() => loadDocuments(big), /2 MB/);
});

test("UTF-8 BOM 会被去掉，标题仍然生效", () => {
  const chunks = parseDocument("bom.md", encoder.encode("\ufeff# 标题\n正文内容。\n"), "D1");
  assert.equal(chunks[0].section, "标题");
  assert.ok(!chunks[0].text.includes("\ufeff"));
});

test("代码块内的井号不当作标题切分", () => {
  const text = [
    "# 顶部",
    "```md",
    "## 代码里的标题",
    "```",
    "## 真正的标题",
    "内容。",
    "",
  ].join("\n");
  const chunks = parseDocument("code.md", encoder.encode(text), "D1");
  assert.deepEqual(chunks.map((chunk) => chunk.section), ["顶部", "真正的标题"]);
});

test("长小节按码点切分，不切断 emoji 代理对", () => {
  const text = `${"😀".repeat(600)}\n结束\n`;
  const chunks = parseDocument("emoji.md", encoder.encode(text), "D1");
  assert.ok(chunks.length >= 2);
  for (const chunk of chunks) {
    assert.ok(Array.from(chunk.text).length <= 500);
    assert.doesNotMatch(chunk.text, /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/);
  }
  assert.equal(chunks[0].startLine, 1);
  assert.equal(chunks.at(-1).endLine, 2);
});

test("空问题和空白问题返回空结果", () => {
  const retriever = new Retriever(loadDocuments(recordsOf(fixture.cases[0].files)));
  assert.deepEqual(retriever.search("", 3, 0), []);
  assert.deepEqual(retriever.search("   ", 3, 0), []);
});
