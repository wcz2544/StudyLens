"""按标题构建父块，再用相邻语义变化构建检索子块。"""
from dataclasses import dataclass, field
import hashlib
import math
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import validate_vectors
from .types import IndexedChunk


@dataclass(frozen=True)
class ChunkingConfig:
    parent_size: int = 1800
    child_size: int = 450
    semantic_unit_size: int = 140
    min_child_size: int = 100
    semantic_similarity_threshold: float = 0.58
    max_units: int = 4096
    max_chunks: int = 2000
    # 进入已有的索引策略指纹，避免旧分块结果与新策略被当作同一版本。
    format_version: str = field(default="semantic_hierarchy_v2", init=False)

    def __post_init__(self):
        if not 20 <= self.semantic_unit_size <= self.child_size <= self.parent_size <= 8000:
            raise ValueError("父子分块大小配置无效。")
        if not 0 <= self.min_child_size <= self.child_size:
            raise ValueError("语义分块最小长度配置无效。")
        if not -1 <= self.semantic_similarity_threshold <= 1:
            raise ValueError("语义断点阈值必须在 -1～1。")
        if not 1 <= self.max_units <= 4096 or not 1 <= self.max_chunks <= 2000:
            raise ValueError("语义单元或切片数量上限超出允许范围。")


def _cosine(a: list[float], b: list[float]) -> float:
    denominator = math.sqrt(sum(v * v for v in a) * sum(v * v for v in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0.0


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Block:
    text: str
    protected: bool = False
    level: int = 0
    title: str = ""


def _blocks(text: str) -> list[_Block]:
    """只解释ATX标题、围栏代码和独立$$公式，逐字保留其余文本。

    这不是完整Markdown/LaTeX解析器；不识别的环境仍视为普通正文。
    使用原始行而非strip后的行组装内容，保留缩进、空行和换行方式。
    """
    output, plain, protected = [], [], []
    fence = None
    in_math = False

    def flush_plain():
        if plain:
            output.append(_Block("".join(plain)))
            plain.clear()

    for line in text.splitlines(keepends=True):
        if fence is not None:
            protected.append(line)
            marker = re.escape(fence[0]) + "{" + str(len(fence)) + ",}"
            if re.fullmatch(r" {0,3}" + marker + r"[ \t]*(?:\r?\n)?", line):
                output.append(_Block("".join(protected), protected=True))
                protected.clear()
                fence = None
            continue
        if in_math:
            protected.append(line)
            if line.strip() == "$$":
                output.append(_Block("".join(protected), protected=True))
                protected.clear()
                in_math = False
            continue
        opener = re.match(r"^ {0,3}(`{3,}|~{3,})[^\r\n]*(?:\r?\n)?$", line)
        if opener:
            flush_plain()
            fence = opener.group(1)
            protected.append(line)
            continue
        if line.strip() == "$$":
            flush_plain()
            in_math = True
            protected.append(line)
            continue
        if line.strip().startswith("$$") and line.strip().endswith("$$"):
            flush_plain()
            output.append(_Block(line, protected=True))
            continue
        heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)[ \t]*|[ \t]*)(?:\r?\n)?$", line)
        if heading:
            flush_plain()
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group(2) or "").strip()
            output.append(_Block(line, level=len(heading.group(1)), title=title))
        else:
            plain.append(line)
    if fence is not None or in_math:
        raise ValueError("代码围栏或独立 $$ 公式块未闭合，请补齐结束标记后再导入。")
    flush_plain()
    return output


def _sections(blocks: list[_Block]) -> list[tuple[Document, list[_Block]]]:
    output, current, headings = [], [], {}
    has_body = False

    def emit():
        if current:
            path = [headings[level] for level in sorted(headings)]
            output.append((Document(page_content="".join(block.text for block in current),
                                    metadata={"heading_path": path}), current.copy()))

    for block in blocks:
        if block.level:
            # 相邻标题无需产生只含标题的噪声块，正文出现后再开始新的章节。
            if has_body:
                emit()
                current.clear()
                has_body = False
            headings = {level: title for level, title in headings.items() if level < block.level}
            headings[block.level] = block.title
        elif block.text.strip():
            has_body = True
        current.append(block)
    emit()
    return output


def _split_plain(text: str, splitter: RecursiveCharacterTextSplitter) -> list[str]:
    parts = splitter.split_text(text)
    # 库升级也不能静默改写证据；保真失败应在任何Embedding付费调用前明确中止。
    if "".join(parts) != text:
        raise ValueError("正文分块未能完整保留文字，请检查资料格式。")
    return parts


class SemanticHierarchySplitter:
    """父块保存提取文字；围栏代码与独立公式不拆开，子块用于召回重排。"""

    def __init__(self, embeddings: Embeddings, config: ChunkingConfig | None = None):
        self.embeddings = embeddings
        self.config = config or ChunkingConfig()

    def split(self, kb_id: str, source_id: str, revision: str, source: str,
              text: str, metadata: dict | None = None) -> list[IndexedChunk]:
        sections = _sections(_blocks(text))
        # 层级边界优先；父块仍有上限，避免回填整个长文导致模型上下文爆炸。
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.parent_size, chunk_overlap=0,
            separators=["\n\n", "\n", "。", " ", ""], keep_separator=True,
            strip_whitespace=False,
        )
        unit_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.semantic_unit_size, chunk_overlap=0,
            separators=["\n\n", "。", "！", "？", ". ", "\n", " ", ""],
            keep_separator=True, strip_whitespace=False,
        )
        parents, units_by_parent = [], []
        for section, blocks in sections:
            current, current_length = [], 0

            def emit_parent():
                if not current:
                    return
                parents.append(Document(page_content="".join(block.text for block in current),
                                        metadata=dict(section.metadata)))
                units_by_parent.append([unit for block in current
                                        for unit in ([block.text] if block.protected
                                                     else _split_plain(block.text, unit_splitter))])

            for block in blocks:
                if block.protected and len(block.text) > self.config.parent_size:
                    raise ValueError("代码或独立公式块超过父块长度限制，请人工整理后再导入；系统不会截断该块。")
                pieces = [block] if block.protected else [
                    _Block(part) for part in _split_plain(block.text, parent_splitter)]
                for piece in pieces:
                    if current and current_length + len(piece.text) > self.config.parent_size:
                        emit_parent()
                        current, current_length = [], 0
                    current.append(piece)
                    current_length += len(piece.text)
            emit_parent()
        # 空白留在父子文本中，但不单独购买无语义的Embedding。
        units = [unit for items in units_by_parent for unit in items if unit.strip()]
        if len(units) > self.config.max_units:
            raise ValueError("文档语义单元过多，请拆分文件后再导入。")
        if not units:
            raise ValueError("没有可建立索引的文字。")
        # 用真实 Embedding 的相邻相似度判断主题变化，不把字符分割冒充语义分块。
        vectors = validate_vectors(self.embeddings.embed_documents(units), len(units))
        chunks, vector_offset = [], 0
        for parent_number, (parent, parent_units) in enumerate(zip(parents, units_by_parent)):
            headings = parent.metadata["heading_path"]
            section = headings[-1] if headings else "正文"
            parent_id = _digest(f"{kb_id}|{source_id}|{revision}|parent|{parent_number}")
            page = next((int(match.group(1)) for heading in headings
                         if (match := re.fullmatch(r"第\s*(\d+)\s*页", heading))), None)
            safe_metadata = {**(metadata or {}), "heading_path": headings, "page": page,
                             "parent_id": parent_id, "chunking": self.config.format_version}
            groups, current = [], ""
            previous_vector = None
            for unit in parent_units:
                if not unit.strip():
                    current += unit
                    continue
                vector = vectors[vector_offset]
                vector_offset += 1
                topic_changed = (previous_vector is not None
                                 and len(current) >= self.config.min_child_size
                                 and _cosine(previous_vector, vector)
                                 < self.config.semantic_similarity_threshold)
                size_exceeded = len(current) + len(unit) > self.config.child_size
                if current and (topic_changed or size_exceeded):
                    groups.append(current)
                    current = ""
                current += unit
                previous_vector = vector
            if current.strip():
                groups.append(current)
            for child_number, body in enumerate(groups):
                if body.strip():
                    chunk_id = _digest(f"{parent_id}|child|{child_number}")
                    chunks.append(IndexedChunk(
                        chunk_id, kb_id, source_id, revision, parent_id,
                        source, section, body, parent.page_content, safe_metadata,
                    ))
            if len(chunks) > self.config.max_chunks:
                raise ValueError("文档切片过多，请拆分文件后再导入。")
        return chunks
