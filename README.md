# 多模态客服 Agent

这是一个证据约束的客服 RAG 系统。当前运行入口统一走轻量版 `AgentGraph`，旧的“直接检索后生成”流程不再作为 `/chat` 或离线评测入口使用。

## 架构

```text
POST /chat
  -> ChatService
  -> AgentGraph
     -> ContextResolverNode
     -> PlannerNode
     -> ProductRouterNode
     -> RetrievalNode
     -> RerankNode
     -> EvidenceJudgeNode
     -> RetryNode
     -> FactExtractorNode
     -> ImageBinderNode
     -> AnswerGeneratorNode
     -> AnswerVerifierNode
     -> FinalResponseNode
```

核心原则：

- 先规划问题，再限制产品和手册范围。
- 检索和 rerank 后必须经过 `EvidenceJudgeNode`。
- 只有 `accepted_evidence` 能进入事实抽取和答案生成。
- 图片只来自固定 intent 图片或 accepted evidence 绑定图片。
- LLM 可用于 planner / evidence judge / fact extraction / verifier，但不能绕过证据自由回答。
- 每次请求都会写 trace，方便定位错误发生在哪个节点。

## 主要目录

```text
app/
  api/          # FastAPI /chat 接口
  core/         # AgentState、AgentGraph、配置、trace
  nodes/        # AgentGraph 节点
  prompts/      # LLM planner / judge / verifier prompt
  schemas/      # 请求和响应 schema
  services/     # 知识库、检索、reranker、LLM client、生成器、会话记忆
  evaluation/   # regression 测试和结果对比工具

scripts/
  build_index.py
  evaluate.py
  evaluate_rerank_agent.sh
  serve_reranker.py
```

## 构建索引

```bash
python3 scripts/build_index.py
```

生成：

- `data/parsed/manuals.jsonl`
- `data/chunks/manual_chunks.jsonl`
- `data/chunks/images.jsonl`
- `data/chunks/retrieval_corpus.jsonl`

## 启动 API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

请求：

```json
{
  "question": "如何开启空调的节能制冷模式？",
  "images": [],
  "session_id": "kf_session_demo"
}
```

响应：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "...",
    "session_id": "kf_session_demo",
    "timestamp": 1710000000,
    "references": [],
    "related_images": []
  }
}
```

如果配置 `KAFU_API_TOKEN`，接口需要：

```text
Authorization: Bearer <token>
```

## 评测

不启用 LLM / dense / reranker：

```bash
LLM_ENABLED=0 DENSE_ENABLED=0 RERANK_ENABLED=0 python3 scripts/evaluate.py --output-prefix local_graph
```

启用 reranker：

```bash
bash scripts/evaluate_rerank_agent.sh --output-prefix local_graph_rerank
```

查看答案：

```bash
python3 scripts/show_eval_answers.py data/eval/diagnostics_local_graph_rerank.csv --ids 79 123 172
```

## 可选 LLM

LLM 默认不直接生成最终答案。推荐只用于受控节点：

```bash
export LLM_ENABLED=1
export LLM_BASE_URL=http://127.0.0.1:8001/v1
export LLM_MODEL=qwen2.5-7b-instruct
export LLM_API_KEY=EMPTY

export LLM_PLANNER_ENABLED=1
export LLM_EVIDENCE_JUDGE_ENABLED=1
```

不建议开启自由最终生成；最终答案仍应来自 accepted evidence 或 high-confidence direct intent。

## Regression

历史错题只作为 regression case，不写入生产逻辑题号分支。

```bash
PYTHONDONTWRITEBYTECODE=1 TRACE_LOG_DIR=/tmp/agent_traces python3 app/evaluation/run_regression.py
```
