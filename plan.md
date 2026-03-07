Dia-Agent (2026版) 全周期工程落地蓝图
0. 项目核心定义
定位：基于“神经符号逻辑（Neuro-Symbolic）”的专病智能体。
杀手锏属性：Guardrail-First（红线优先）。系统在生成任何建议前，必须先强制检索 Neo4j 结构化知识库，生成硬性临床约束，而非仅仅依赖大模型的概率预测。
第一阶段：医学数据资产化（底层根基构建）
1.1 结构化图谱构建 (Neo4j)
数据源：已有的 graph.json（含300+核心药品的禁忌、阈值、调整逻辑）。
Schema 设计：
节点：Drug (名称), Indicator (指标名, 单位), Disease (病名), DosageStrategy (策略描述)。
关系：
(Drug)-[:CONTRAINDICATED_BY {operator, threshold, reason}]->(Indicator)
(Disease)-[:EXCLUDES]->(Drug)
(Drug)-[:ADJUST_DOSAGE {condition}]->(DosageStrategy)
任务：编写 Python 脚本解析 graph.json 并使用 py2neo 或 neo4j 驱动批量执行 MERGE 操作。
1.2 混合 RAG 引擎构建 (ChromaDB)
数据源：中国1型糖尿病诊治指南（2021版）.pdf 及 dc26sint.pdf (ADA 2025/2026)。
处理流：
解析：使用 Marker 库或 PyMuPDF 将 PDF 转为高精度 Markdown（保留表格逻辑）。
切片：采用语义切片（RecursiveCharacterTextSplitter），Chunk Size 800, Overlap 100。
向量化：使用 BGE-M3 或 m3e-base 模型。
存储：存入 ChromaDB，并为不同指南设置元数据标签（source: "CHN", "ADA"）。
第二阶段：感知层与状态标准化（Node A）
2.1 多模态感知节点 (Perception Node)
输入：化验单图片、描述性文本、历史病例。
逻辑：
调用 Qwen2-VL-7B 或 GPT-4o 提取关键数值。
Schema 强制对齐：输出必须符合 PatientState Pydantic 模型：
code
Python
{
  "indicators": {"eGFR": float, "HbA1c": float, "ALT": float, ...},
  "diseases": [list of strings],
  "current_drugs": [list of strings]
}
第三阶段：双逻辑推理引擎设计（核心大脑）
3.1 安全守卫节点 (Guardrail Node - 核心创新)
任务：“生成法律文件”。
逻辑：
接收 PatientState。
自动生成 Cypher 语句查询 Neo4j。
查询示例：MATCH (d:Drug)-[r:CONTRAINDICATED_BY]->(i:Indicator) WHERE i.name='eGFR' AND 25 < r.threshold RETURN d.name, r.reason。
输出：一段结构化的《本次问诊安全约束白皮书》，明确列出当前患者禁止使用的药物及必须调整的剂量建议。
3.2 仲裁推理节点 (Reasoner Node)
Prompt 逻辑：
输入：[感知到的患者状态] + [Neo4j 输出的红线约束] + [RAG 检索的指南片段]。
系统指令：要求模型在“红线约束”的绝对范围内，参考“指南建议”给出诊疗方案。
冲突处理：明确规定：若指南建议与红线冲突，必须优先遵守红线。
第四阶段：基于 LangGraph 的 Agent 工作流编排
4.1 拓扑结构定义
流程图：START -> Perception -> Guardrail -> Reasoner -> Auditor -> END。
反思循环 (Reflexion Loop)：
Auditor 节点：独立审查 Reasoner 的输出。
核对项：检查输出内容是否包含了 Guardrail 禁止的药物。
逻辑跳转：若违规，生成错误日志并触发 RETRY 连线回到 Reasoner。限制最大重试次数为 3。
第五阶段：指令微调与偏好对齐（可选进阶）
5.1 数据集构造
使用 graph.json 构造 2k 条 SFT 数据。
正样本：严谨引用红线且符合指南的回复。
负样本：只看指南而忽略了严重禁忌（如 eGFR < 30 仍开二甲双胍）的回复。
训练方案：使用 LLaMA-Factory 进行 QLoRA 微调，强化模型对安全红线的服从度。
第六阶段：工程化展示与评测（演示闭环）
6.1 前端交互 (Streamlit)
透明思维链展示：
左侧显示实时检索到的 Neo4j 知识点。
右侧显示 RAG 检索的指南原文。
中间显示 Agent 的思考流（Thinking Process）。
结果对比：设计“标准 LLM”与“Dia-Agent”对比模式，演示系统如何成功拦截高危处方。
6.2 自动化评测 (Ragas)
构建 50 个“陷阱案例”（例如：指标处于禁忌临界点的患者）。
核心 KPI：红线拦截率 (Interception Rate) 必须 100%；医学指南符合率 > 90%。
🛠️ 技术栈清单 (Tech Stack)
语言：Python 3.10+
大模型编排：LangGraph, LangChain
图数据库：Neo4j (Cypher)
向量数据库：ChromaDB
PDF 解析：Marker / PyPDFLoader
模型选型：Qwen2.5-7B-Instruct (推理), Qwen2-VL (感知)
API 框架：FastAPI
