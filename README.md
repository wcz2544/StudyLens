# StudyLens：能更新、能核对依据的知识库问答

[![Checks](https://github.com/wcz2544/StudyLens/actions/workflows/ci.yml/badge.svg)](https://github.com/wcz2544/StudyLens/actions/workflows/ci.yml)

[打开在线网页](https://studylens-mqwruhmf4vdeystbecevds.streamlit.app/) · [非技术流程讲解](docs/项目流程讲解_非技术版.md) · [部署与运维](docs/部署与运维.md) · [验证记录](docs/升级验证记录.md)

面向学生与小型团队的资料检索问答项目。支持 TXT、Markdown、图片、PDF、DOCX。新增 LangChain + Milvus 持久知识库、混合检索、重排和受限 Agent，并保留原有无需 API 的临时笔记检索。

这是针对单组织、单服务进程设计的工程版本。代码与自动化测试不能替代真实模型质量、并发容量和正式环境验收；当前验证范围见验证记录。

## 两种使用方式

- **临时笔记**：资料保存在当前会话，字符 TF-IDF 检索不调用模型；AI 回答使用用户填写的、部署者允许的服务商接口。
- **持久知识库**：部署者连接知识库服务后，用户输入只读或管理凭据。支持资料新增、替换、删除；问答采用关键词与向量双路召回、重排、证据审核。该模式会使用部署者配置的模型额度。

在线网页是否已连接知识库服务，以页面提示为准。更新 GitHub 不会自动替你创建 Milvus 服务器或取得模型密钥。

## 新版处理流程

资料提取 → 标题层级与语义父子分块 → 新版本准备 → 原子发布。

问题 → 判断是否澄清/检索 → BM25 + Milvus 向量召回 → RRF 合并 → Rerank 过滤 → 父块上下文回填 → 必要时改写再查 → 逐项引文检查 → 模型审核 → 答案、引用与处理记录。

- 更新完成后才发布；相同内容重复导入不重复建库。
- 关键词与向量使用同一活动版本；删除后立即停止召回，物理清理可重试。
- Agent 最多四次模型任务、两轮检索、三个检索词，不开放任意网址访问或代码执行。
- 精确引文必须出现在本次证据中，语义审核不通过则拒答；仍不保证消灭幻觉。
- API 区分管理/只读凭据，限制请求大小、速度和并发任务，日志不记录原始笔记或密钥。

## 快速运行基础网页

Python 3.10 或 3.11，建议使用项目虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

打开终端提示的本地地址。文字版文件无需 OCR 系统组件；图片和扫描 PDF 需要另装 Tesseract 及 `chi_sim`、`eng` 语言包。仅安装 pytesseract 不等于安装了识别程序。容器和 Streamlit Cloud 的系统依赖已声明。

上传限制：最多10份；文字512KiB/份、图片8MiB/份、PDF/DOCX 12MiB/份，原文件合计24MiB；提取文字512KiB/份、合计2MiB。持久知识库服务每次导入一份，使用同一格式限制。

## 运行完整知识库

完整步骤、环境变量含义、HTTPS、备份与回滚见 [部署与运维](docs/部署与运维.md)。服务依赖独立保存在 `requirements-rag.txt`，原网页安装不强制携带 Milvus 客户端。

容器路径：复制 `.env.rag.example` 为 `.env.rag`，填好访问凭据、Milvus 和三个模型服务配置后，在仓库根目录运行：

```powershell
docker compose --env-file .env.rag -f deploy/compose.yml up -d --build
```

然后访问 `http://localhost:8501`，选择“持久知识库”。没有 Docker 的 Windows 环境可使用远程 Milvus/Zilliz 与本机 API，见部署文档。项目不会在缺少外部服务时用模拟答案充当生产结果。

## 评测与自动化

原30题开发集：三份自带笔记，默认 Top-3、阈值0.08，2026-10-31 实测单段命中15/15、多段全部证据覆盖10/10、无答案空检索0/5。最后一项不是模型拒答准确率。

新增18题开发集覆盖同义、跨语言、多证据、歧义、缺证、冲突、更新、删除、闲聊。基础检索在其中12道有答案题的 Recall@5 为10/12（83.33%）。真实混合检索提升和模型回答准确率需要外部服务与人工标注，未用模拟结果代替。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B evaluate.py --output evaluation/results.csv
.\.venv\Scripts\python.exe -B evaluate_rag.py --backend baseline
```

GitHub Actions 包含基础回归、离线 Agent/检索/API 测试和真实 Milvus 集成；集成使用确定性测试向量，不代表真实语义质量。评测口径与人工标注见 [评测说明](docs/评测说明.md)。

## 边界与协作

当前支持一个工作区、一个 API worker；不宣称多租户高可用、无限并发或完整多轮记忆。没有自动监听电脑目录，资料变动需重新导入或调用更新接口。复杂排版、手写、公式和图表需要核对提取结果。源文件请自行保留，知识库保存的是提取后的文本与索引。

问题反馈请提供：脱敏复现步骤、预期与实际行为、版本、请求编号；不要附 API Key、私密原文或未授权资料。提交前运行对应测试；涉及服务契约变更时同步文档和 CI。项目目前尚未选择 LICENSE，公开仓库不等于已经授予任意再分发许可。

实现由项目负责人驱动、AI Agent 辅助完成。适合通过阅读非技术流程说明、测试和评测记录学习；简历中应只报告自己能解释、能复现的实现和指标。
