/**
 * 浏览器端检索内核。
 *
 * 行为与 studylens/documents.py、studylens/retrieval.py 保持一致：
 * - 只支持 UTF-8 的 .txt / .md，按 Markdown 标题分段，超长小节用滑动窗口；
 * - 检索使用字符 TF-IDF（单字与连续 2～3 字），L2 归一化后点积即余弦相似度；
 * - 所有偏移量按 Unicode 码点计算，与 Python 的字符串切片一致。
 *
 * 这里不依赖任何第三方库、网络或服务端，产物可以直接放在 GitHub Pages 上。
 */

export const MAX_FILE_BYTES = 512 * 1024;
export const MAX_TOTAL_BYTES = 2 * 1024 * 1024;
export const MAX_FILES = 10;
export const CHUNK_SIZE = 500;
export const OVERLAP = 80;
export const MAX_FEATURES = 50000;
export const SUPPORTED_SUFFIXES = [".txt", ".md"];

/** Python str.splitlines(keepends=True) 认可的换行边界。 */
const LINE_BREAK = /\r\n|[\n\r\v\f\x1c\x1d\x1e\u0085\u2028\u2029]/g;

export function splitLinesKeepEnds(text) {
  const lines = [];
  let start = 0;
  LINE_BREAK.lastIndex = 0;
  let match = LINE_BREAK.exec(text);
  while (match !== null) {
    const end = match.index + match[0].length;
    lines.push(text.slice(start, end));
    start = end;
    match = LINE_BREAK.exec(text);
  }
  if (start < text.length) lines.push(text.slice(start));
  return lines;
}

/** 与 pathlib.Path(name).suffix.lower() 对齐：隐藏文件名没有后缀。 */
export function fileSuffix(name) {
  const dot = name.lastIndexOf(".");
  if (dot <= 0 || dot === name.length - 1) return "";
  return name.slice(dot).toLowerCase();
}

export function isSupportedTextFile(name) {
  return SUPPORTED_SUFFIXES.includes(fileSuffix(name));
}

function matchHeading(line) {
  if (!line.startsWith("#")) return null;
  // Python 的 $ 允许尾部换行，这里先去掉行尾空白再整体匹配，结果相同。
  const candidate = line.replace(/\s+$/u, "");
  const match = /^#{1,6}\s+(.+?)\s*$/u.exec(candidate);
  return match ? match[1] : null;
}

function countNewlines(text) {
  let total = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] === "\n") total += 1;
  }
  return total;
}

function flushBlock(chunks, source, section, block, blockStart) {
  if (block.length === 0) return;
  const text = block.join("");
  // 仅当文本含代理对（如 emoji）时按码点拆开，普通笔记保持零额外开销。
  const units = /[\uD800-\uDBFF]/.test(text) ? Array.from(text) : null;
  const length = units ? units.length : text.length;
  const slice = (start, end) => (units ? units.slice(start, end).join("") : text.slice(start, end));
  let offset = 0;
  let scanned = 0;
  let newlines = 0;
  while (offset < length) {
    const end = Math.min(offset + CHUNK_SIZE, length);
    const fragment = slice(offset, end);
    if (fragment.trim()) {
      while (scanned < offset) {
        if ((units ? units[scanned] : text[scanned]) === "\n") newlines += 1;
        scanned += 1;
      }
      const first = blockStart + newlines;
      const last = first + countNewlines(fragment.replace(/[\r\n]+$/, ""));
      chunks.push({
        id: `${source.id}-${chunks.length + 1}`,
        source: source.name,
        section,
        startLine: first,
        endLine: last,
        text: fragment,
      });
    }
    if (end === length) break;
    offset = end - OVERLAP;
  }
}

/**
 * 解析一份笔记，返回带原文行号的段落。
 * @param {string} name 文件名（含后缀）
 * @param {Uint8Array} bytes 原始字节
 * @param {string} docId 文档编号，例如 D1
 */
export function parseDocument(name, bytes, docId) {
  const suffix = fileSuffix(name);
  if (!SUPPORTED_SUFFIXES.includes(suffix)) {
    throw new Error("仅支持 .txt 和 .md 文件。");
  }
  if (bytes.length > MAX_FILE_BYTES) {
    throw new Error(`${name} 超过 512 KB，请先缩小文件。`);
  }
  let content;
  try {
    content = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw new Error(`${name} 不是 UTF-8，请在编辑器中另存为 UTF-8。`);
  }
  if (!content.trim() || content.includes("\x00")) {
    throw new Error(`${name} 为空或不是正常文本。`);
  }

  const chunks = [];
  let section = "正文";
  let block = [];
  let blockStart = 1;
  let inCode = false;
  const lines = splitLinesKeepEnds(content);
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const number = index + 1;
    const heading = inCode || suffix !== ".md" ? null : matchHeading(line);
    if (heading !== null) {
      flushBlock(chunks, { id: docId, name }, section, block, blockStart);
      block = [];
      section = heading;
      blockStart = number;
    }
    if (block.length === 0) blockStart = number;
    block.push(line);
    // 代码块里的 # 只是注释，不应当作标题切分。
    const stripped = line.replace(/^\s+/u, "");
    if (stripped.startsWith("```") || stripped.startsWith("~~~")) inCode = !inCode;
  }
  flushBlock(chunks, { id: docId, name }, section, block, blockStart);
  return chunks;
}

/** 与 documents.load_documents 相同的校验顺序与提示语。 */
export function loadDocuments(files) {
  if (!Array.isArray(files) || files.length === 0 || files.length > MAX_FILES) {
    throw new Error("请选择 1～10 份笔记。");
  }
  let total = 0;
  for (const file of files) total += file.bytes.length;
  if (total > MAX_TOTAL_BYTES) throw new Error("文件总大小不能超过 2 MB。");
  const chunks = [];
  files.forEach((file, index) => {
    chunks.push(...parseDocument(file.name, file.bytes, `D${index + 1}`));
  });
  return chunks;
}

/**
 * sklearn 1.7 的 _char_ngrams 先执行 re.sub(r"\s\s+", " ", text)。
 * 这里列出 Python \s 认可的空白字符，避免把单个换行误折叠。
 */
const PY_WHITESPACE = "\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WHITESPACE_RUN = new RegExp(`[${PY_WHITESPACE}]{2,}`, "g");

function normalizeText(text) {
  // 顺序与 sklearn 一致：先 lower()，再由分词器折叠连续空白。
  return text.toLowerCase().replace(WHITESPACE_RUN, " ");
}

/** 统计单字与连续 2～3 字片段，键是原始字符串片段。 */
export function tokenize(text) {
  const lowered = normalizeText(text);
  const counts = new Map();
  const starts = [];
  let index = 0;
  while (index < lowered.length) {
    const code = lowered.charCodeAt(index);
    const width = code >= 0xd800 && code <= 0xdbff ? 2 : 1;
    const end = index + width;
    starts.push(index);
    for (let size = 1; size <= 3 && size <= starts.length; size += 1) {
      const start = starts[starts.length - size];
      const gram = lowered.slice(start, end);
      counts.set(gram, (counts.get(gram) || 0) + 1);
    }
    index = end;
  }
  return counts;
}

function buildWeights(counts, idf) {
  const weights = new Map();
  let sumSquares = 0;
  for (const [term, count] of counts) {
    const inverse = idf.get(term);
    if (inverse === undefined) continue;
    // sublinear_tf=True：1 + log(tf)。
    const value = (1 + Math.log(count)) * inverse;
    weights.set(term, value);
    sumSquares += value * value;
  }
  const norm = Math.sqrt(sumSquares);
  if (norm > 0) {
    for (const [term, value] of weights) weights.set(term, value / norm);
  }
  return weights;
}

/**
 * 与 sklearn TfidfVectorizer(analyzer="char", ngram_range=(1,3),
 * sublinear_tf=True, norm="l2", max_features=50000) 对齐的索引。
 */
export class Retriever {
  constructor(chunks) {
    if (!chunks || chunks.length === 0) {
      throw new Error("没有可索引的文本段落。");
    }
    this.chunks = chunks;
    const docFrequency = new Map();
    const termFrequency = new Map();
    const docTokens = chunks.map((chunk) => {
      const counts = tokenize(chunk.text);
      for (const [term, count] of counts) {
        docFrequency.set(term, (docFrequency.get(term) || 0) + 1);
        termFrequency.set(term, (termFrequency.get(term) || 0) + count);
      }
      return counts;
    });

    let vocabulary = [...docFrequency.keys()];
    if (vocabulary.length > MAX_FEATURES) {
      // sklearn 按全库词频截断；词频相同时的顺序由排序算法决定，这里固定一个确定顺序。
      vocabulary.sort((left, right) => {
        const delta = termFrequency.get(right) - termFrequency.get(left);
        if (delta !== 0) return delta;
        return left < right ? -1 : left > right ? 1 : 0;
      });
      vocabulary = vocabulary.slice(0, MAX_FEATURES);
    }

    const total = chunks.length;
    this.idf = new Map();
    for (const term of vocabulary) {
      const df = docFrequency.get(term);
      // smooth_idf=True：ln((1+n)/(1+df)) + 1。
      this.idf.set(term, Math.log((1 + total) / (1 + df)) + 1);
    }
    this.vectors = docTokens.map((counts) => buildWeights(counts, this.idf));
  }

  search(question, topK = 3, minScore = 0.08) {
    if (!question || !question.trim()) return [];
    if (!Number.isInteger(topK) || topK < 1 || topK > 10) {
      throw new Error("检索参数超出范围。");
    }
    if (typeof minScore !== "number" || Number.isNaN(minScore) || minScore < 0 || minScore > 1) {
      throw new Error("检索参数超出范围。");
    }
    const query = buildWeights(tokenize(question.trim()), this.idf);
    const scores = this.vectors.map((vector) => {
      const [small, large] = query.size <= vector.size ? [query, vector] : [vector, query];
      let score = 0;
      for (const [term, weight] of small) {
        const other = large.get(term);
        if (other !== undefined) score += weight * other;
      }
      return score;
    });
    const ranked = scores
      .map((score, index) => ({ score, index }))
      // 与 Python 的 key=lambda i: (-scores[i], i) 一致：分数降序，同分按段落顺序。
      .sort((left, right) => right.score - left.score || left.index - right.index)
      .slice(0, topK);
    return ranked
      .filter((item) => scores[item.index] > 0 && scores[item.index] >= minScore)
      .map((item) => ({ chunk: this.chunks[item.index], score: item.score }));
  }
}
