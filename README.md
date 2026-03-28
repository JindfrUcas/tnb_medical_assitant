# Dia-Agent

Dia-Agent 是一个面向糖尿病场景的 Guardrail-First 专病智能体。它把结构化红线约束、指南检索、问诊推理和结果审计串成一条完整链路，生成安全、可追溯的诊疗建议。

核心思路：**先查红线、再查指南、最后给建议**。Reasoner 采用受控 ReAct 模式，AI 只决定"下一步查什么"，不直接编写数据库查询；Auditor 对输出执行二次审计，违规自动回流重新生成。

## 技术栈

- **Python 3.10+**
- **LangChain / LangGraph** — LLM 编排与 ReAct Agent
- **Neo4j** — 药物禁忌/疾病排除/剂量调整知识图谱
- **ChromaDB + HuggingFace Embeddings** — 指南 RAG 向量检索
- **FastAPI** — HTTP 服务接口
- **Streamlit** — 前端演示界面
- **Pydantic / Pydantic-Settings** — 数据模型与配置管理
- **PyMuPDF** — 指南 PDF 解析
