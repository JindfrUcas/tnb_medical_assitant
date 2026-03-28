# Dia-Agent

Dia-Agent 是一个面向糖尿病场景的 Guardrail-First 专病智能体。以 ReAct 自主推理为主控循环，以 GraphRAG 图遍历为核心证据引擎，生成安全、可追溯的诊疗建议。

## 架构

```
START → Perception（输入标准化）→ ReAct Agent（主控循环）→ END
                                      ↑
                           7 个受控工具：
                           guardrail_check    — 确定性红线检查
                           graph_traverse     — 1-2跳图遍历（实体→chunk→兄弟chunk / 共享指标→关联药物）
                           graph_expand       — 多实体子图提取（共同chunk + 实体间关系路径）
                           vector_search      — 向量相似度检索补充
                           get_dosage_info    — 剂量策略 + 相关指南chunk
                           audit_self         — 自审计（禁药检查）
                           get_chapter_context — 章节层级回溯 + 相邻chunk上下文
```

ReAct Agent 自主决定调用顺序：先查红线、再图遍历获取证据、不足时向量补充、生成建议后自审计、未通过则修正重试。

## 技术栈

- **Python 3.10+**
- **LangGraph** — ReAct Agent 主控循环（`create_react_agent v2`）
- **LangChain** — LLM 编排与工具封装
- **Neo4j** — 知识图谱（红线规则图 + 指南证据图），支持多跳遍历和子图提取
- **ChromaDB + HuggingFace Embeddings** — 指南 RAG 向量检索
- **FastAPI** — HTTP 服务接口
- **Streamlit** — 前端演示界面
- **Pydantic / Pydantic-Settings** — 数据模型与配置管理
- **PyMuPDF** — 指南 PDF 解析
