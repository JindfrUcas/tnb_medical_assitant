# Dia-Agent (Guardrail-First)

Dia-Agent 是一个面向糖尿病场景的 `Guardrail-First` 智能体项目。它把结构化红线约束、指南检索、问诊推理和结果审计串成一条完整链路，用来生成更安全、更可追溯的诊疗建议。

详细使用说明见：`docs/详细使用文档.md`

## 1. 项目功能介绍

这个项目要做的事情，不是简单地“让大模型回答医疗问题”，而是构建一个带安全约束的专病 Agent。

核心能力包括：

- **患者状态结构化**：把文本、JSON 等输入整理成统一的患者状态。
- **红线约束检索**：在生成建议之前，先从图谱或本地结构化数据中检索禁忌、阈值和剂量调整规则。
- **指南 RAG 检索**：从糖尿病相关指南 PDF 中检索与当前问题相关的参考片段。
- **诊疗建议生成**：融合患者信息、红线约束和指南片段，输出最终建议。
- **结果审计回流**：如果建议违反安全约束，审计节点会拦截并触发重新生成。
- **多种使用方式**：支持命令行问诊、FastAPI 接口和 Streamlit 前端演示。

一句话概括：**我们做的是一个“先查红线、再查指南、最后给建议”的糖尿病专病智能体。**

## 2. 项目结构

```text
.
├── dataset/
│   ├── graph.json
│   ├── 中国1型糖尿病诊治指南（2021版）.pdf
│   └── dc26sint.pdf
├── dia_agent/
│   ├── api/                 # FastAPI 服务
│   ├── evaluation/          # 陷阱案例生成与评测
│   ├── graph/               # Neo4j/JSON 图谱访问层
│   ├── nodes/               # Perception / Guardrail / Reasoner / Auditor
│   ├── rag/                 # PDF 解析、切片、向量索引与检索
│   ├── sft/                 # SFT 数据构造
│   ├── ui/                  # Streamlit 演示界面
│   ├── workflow/            # LangGraph 工作流
│   └── pipeline.py          # 总装配入口
├── scripts/
│   ├── import_graph_to_neo4j.py
│   ├── build_rag_index.py
│   ├── run_consultation.py
│   ├── run_api.py
│   ├── run_evaluation.py
│   └── build_sft_dataset.py
└── plan.md
```

目录说明：

- `dia_agent/`：项目主代码，包含工作流、节点、RAG、API 和 UI。
- `dataset/`：结构化图谱数据、指南 PDF、评测数据集。
- `scripts/`：索引构建、图谱导入、问诊运行、API 启动、评测等脚本入口。
- `docs/`：更详细的使用文档。

## 3. 环境准备

推荐先进入你的 Conda 环境：

```bash
conda activate tnb_medical_assitant
cd ~/tnb_medical_assitant
```

如果你不用 Conda，也可以使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果已经在 Conda 环境中，直接安装依赖即可：

```bash
pip install -r requirements.txt
```

建议复制示例配置文件：

```bash
cp .env.example .env
```

可选环境变量（OpenAI 兼容接口 + Neo4j）：

```bash
export DIA_AGENT_NEO4J_URI="bolt://localhost:7687"
export DIA_AGENT_NEO4J_USER="neo4j"
export DIA_AGENT_NEO4J_PASSWORD="your-password"
export DIA_AGENT_NEO4J_DATABASE="neo4j"

export DIA_AGENT_LLM_BASE_URL="https://your-llm-endpoint/v1"
export DIA_AGENT_LLM_API_KEY="your-api-key"
export DIA_AGENT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export DIA_AGENT_LLM_WIRE_API="chat_completions"  # 若中转站走 /v1/responses，则改为 responses

# 可选：视觉感知模型（化验单图片抽取）
export DIA_AGENT_VISION_BASE_URL="https://your-vision-endpoint/v1"
export DIA_AGENT_VISION_API_KEY="your-vision-api-key"
export DIA_AGENT_VISION_MODEL="Qwen/Qwen2-VL-7B-Instruct"
export DIA_AGENT_VISION_WIRE_API="chat_completions"
```

可复制 `.env.example` 为 `.env` 后填写自己的配置。

如果不配置 Neo4j 凭据，系统会自动回退到 `dataset/graph.json` 作为结构化约束源。

## 4. 构建流程

### 4.1 导入图谱到 Neo4j

```bash
python scripts/import_graph_to_neo4j.py \
  --input dataset/graph.json \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password your-password

# 如需清库后重建（危险操作，会删除库中现有图数据）
python -c 'from neo4j import GraphDatabase; d=GraphDatabase.driver("bolt://localhost:7687",auth=("neo4j","your-password")); s=d.session(database="neo4j"); s.run("MATCH (n) DETACH DELETE n").consume(); s.close(); d.close()'
python scripts/import_graph_to_neo4j.py --input dataset/graph.json --uri bolt://localhost:7687 --user neo4j --password your-password --database neo4j --skip-constraints
```

### 4.2 构建 RAG 向量索引

```bash
python scripts/build_rag_index.py --reset
```

### 4.3 命令行问诊

```bash
python scripts/run_consultation.py \
  --raw-input '{"indicators":{"eGFR":25,"HbA1c":9.3},"diseases":["糖尿病酮症酸中毒"],"current_drugs":["盐酸二甲双胍片 (普通片/缓释片)"]}' \
  --rag-query "1型糖尿病 肾功能不全 用药"

# 多模态输入（文本 + 图片）
python scripts/run_consultation.py \
  --raw-input "患者eGFR:25，HbA1c:9.3" \
  --image data/uploads/lab_001.png \
  --rag-query "1型糖尿病 肾功能不全 用药"
```

### 4.4 启动 API

```bash
python scripts/run_api.py
```

接口：

- `GET /health`
- `POST /v1/consult`

请求示例：

```json
{
  "raw_input": {
    "indicators": {"eGFR": 25, "HbA1c": 9.3},
    "diseases": ["糖尿病酮症酸中毒"],
    "current_drugs": ["盐酸二甲双胍片 (普通片/缓释片)"]
  },
  "rag_query": "1型糖尿病 肾功能不全 用药",
  "history_text": ""
}
```

多模态输入示例（文本 + 图片路径）：

```json
{
  "raw_input": {
    "text": "患者 eGFR 25, HbA1c 9.3，伴糖尿病酮症酸中毒，当前用药为二甲双胍。",
    "image_paths": ["data/uploads/lab_001.png"],
    "history_text": "病史: 糖尿病酮症酸中毒"
  },
  "rag_query": "1型糖尿病 肾功能不全 用药"
}
```

### 4.5 启动 Streamlit

```bash
streamlit run dia_agent/ui/streamlit_app.py
```

### 4.6 自动化评测

```bash
python scripts/run_evaluation.py --rebuild-cases --limit 50

# 可选：启用 Ragas 指标（需已配置可用 LLM/Embedding 访问能力）
python scripts/run_evaluation.py --use-ragas
```

### 4.7 生成 SFT 数据集（可选）

```bash
python scripts/build_sft_dataset.py --size 2000 --output dataset/sft_guardrail_2k.jsonl
```

输出 JSONL 可直接用于 LLaMA-Factory 的 SFT/偏好对齐流程。

## 5. 关键约束原则

- 先 Guardrail，后 Reasoner。
- 指南建议与红线冲突时，红线绝对优先。
- Auditor 对输出执行二次审计，违规自动回流。
