# 公开资料语料（public notes）

本目录是网页“内置示例笔记”的一部分，全部来自可再分发的公开文档，用 `scripts/prepare_public_notes.py` 抓取并整理为 Markdown。抓取日期：2026-11-18。

## 来源与许可

| 文件 | 来源 | 许可 |
| --- | --- | --- |
| `01_Rust_所有权.md` | 《Rust 程序设计语言》中文版（[rust-lang-cn/book-cn](https://github.com/rust-lang-cn/book-cn)）第 4.1 节 | MIT OR Apache-2.0 |
| `02_Rust_借用与引用.md` | 同上，第 4.2 节 | MIT OR Apache-2.0 |
| `03_Rust_切片.md` | 同上，第 4.3 节 | MIT OR Apache-2.0 |
| `04_Python_数据结构.md` | [Python 官方文档中文版教程](https://docs.python.org/zh-cn/3/tutorial/datastructures.html) 第 5 章 | Python Software Foundation License 2.0 |
| `05_Python_控制流.md` | [Python 官方文档中文版教程](https://docs.python.org/zh-cn/3/tutorial/controlflow.html) 第 4 章 | Python Software Foundation License 2.0 |

每份文件开头都保留了来源、许可证与抓取日期；正文为原文档的节选整理，措辞与编号以原文档为准。分发本目录内容时请一并保留这些声明，并遵守对应许可证（PSF-2.0 要求在副本中保留版权与许可声明）。

## 配套题目

`evaluation/public_notes_questions.csv` 有 31 道题：26 道单段题、2 道多段题、3 道无答案题（主题不在语料内）。题目与期望小节由项目维护者编写，用于离线检索评测，不涉及模型调用。

## 重新生成

```powershell
python scripts/prepare_public_notes.py
```

脚本优先读取 `var/public_notes_cache/` 的本地缓存（该目录被 Git 忽略），因此没有网络时也能复现相同结果。网络抓取失败会保留已有文件并以非零状态退出。

## 为什么不直接复制整本书

- 单文件 512 KiB、总计 2 MB 的上限是为了让浏览器端索引保持即时；
- 节选范围覆盖“术语定义、规则、用法、对比”等可出题的段落，足够支撑检索与引用演示；
- 完整书籍请查看上游仓库，本目录不替代原文档。