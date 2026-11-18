# 北港学院 · 灯塔调度导论（开放虚构语料）

本目录是 StudyLens 的**补充开发集**：虚构课程讲义 + 配对题目，用于练习检索、引用、拒答与澄清，不依赖真实教材版权。

## 虚构与许可声明

- 内容为原创虚构教学文本，不对应真实学校、港口、产品或人员。
- 联系方式仅使用 `.invalid` 等明显不可达示例，不接收真实邮件。
- 本目录文本以 **CC0 1.0 Universal** 意愿发布：可复制、修改、再分发，无需署名；作者不提供品质保证。仓库根目录若尚未选择整体 LICENSE，以本目录声明为准覆盖本目录文件。

## 目录

| 路径 | 说明 |
| --- | --- |
| `notes/` | 5 份 Markdown 讲义，二级标题为可检索小节 |
| `questions.csv` | 36 题检索配对题：`single` / `multi` / `unanswerable` |
| `rag_cases.jsonl` | 12 条独立 RAG 场景（含更新、删除、冲突、闲聊、跨语言等） |

## 在网页中试用

1. 启动 StudyLens 临时笔记或持久知识库模式。
2. 上传 `notes/` 下全部 Markdown。
3. 按 `questions.csv` 中的问题提问，对照 `expected_sections` 检查是否命中对应小节。

静态版（GitHub Pages）与 Streamlit 版都已内置本语料：前者在“笔记来源”中选“开放虚构语料 · 灯塔调度导论（5 份）”，后者选“使用开放虚构语料”，无需手动上传。

## 离线检索配对题（不改默认基线）

默认 `evaluate.py` 仍使用 `data/sample_notes` 与 `evaluation/questions.csv`，**请勿**用本语料覆盖 CI 基线。

人工核对时：将讲义导入后，按 CSV 的 `kind` 分别统计单段命中、多段全覆盖、无答案题是否空检索或后续拒答。

## 独立场景评测

```powershell
python -B evaluate_rag.py --backend baseline --cases data/open_corpus/rag_cases.jsonl --top-k 5 --output evaluation/generated/open_corpus_rag_baseline.json
```

该命令不调用真实问答模型。结果是补充开发集分数，**不能**与原 18 题 / 30 题基线直接相减当作提升。

## 题型意图（摘要）

- 直查术语与参数、口语同义、跨文件多证据
- 歧义澄清、缺证拒答、版本更新、删除后不可召回
- 冲突通知、提示注入诱饵、表格列对齐、中英跨语言答疑

详细评测口径见仓库 [`docs/评测说明.md`](../../docs/评测说明.md) 中的「开放虚构语料」小节。
