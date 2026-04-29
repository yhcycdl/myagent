# 多模态客服智能体 MVP


## 目前能力

- 解析 [KownledgeBase/手册](/home/cyh/agent/KownledgeBase/手册) 下的说明书文本
- 识别 `<PIC>` 并把图片 ID 绑定到 chunk
- 生成结构化 JSONL：
  - `data/parsed/manuals.jsonl`
  - `data/chunks/manual_chunks.jsonl`
  - `data/chunks/images.jsonl`
- 基于纯 Python 的混合检索：
  - BM25 风格关键词检索
  - 字符 n-gram 相似度检索
  - 会话上下文加权
- 提供 `POST /chat` API
- 支持 `session_id`
- 对图片输入做基础校验和占位式多模态入口，后续可以替换成真正的视觉模型

## 目录

```text
app/
  api/
  core/
  schemas/
  services/
scripts/
  parse_manuals.py
  build_index.py
  evaluate.py
data/
  parsed/
  chunks/
  index/
```

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/build_index.py
uvicorn app.main:app --reload
```

### 可选：接入本地 LLM 生成

当前系统默认仍可使用规则生成。如果你已经有一个 OpenAI 兼容的本地推理服务，例如 `vLLM` 或其他兼容 `/v1/chat/completions` 的服务，可以通过环境变量开启：

```bash
export LLM_ENABLED=1
export LLM_BASE_URL=http://127.0.0.1:8001/v1
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export LLM_API_KEY=EMPTY
export LLM_TIMEOUT_SECONDS=45
export LLM_MAX_TOKENS=512
export LLM_TEMPERATURE=0.2
```

开启后，系统会优先使用：

- 当前检索到的 top-k 证据
- 本地 LLM 生成客服式回答
- 如果模型调用失败，会自动回退到现有规则生成，不影响接口可用性

## API

### `POST /chat`

请求体：

```json
{
  "question": "如何给空调遥控器安装电池？",
  "images": [],
  "session_id": "kf_session_demo",
  "stream": false
}
```

响应体：

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "answer": "您好，根据当前检索到的说明书内容：...",
    "session_id": "kf_session_demo",
    "timestamp": 1710000000,
    "references": [],
    "related_images": []
  }
}
```

如果设置了环境变量 `KAFU_API_TOKEN`，接口会校验 `Authorization: Bearer <token>`。

## 构建知识库

只解析原始手册：

```bash
python3 scripts/parse_manuals.py
```

解析并生成 chunk、图片映射和元数据：

```bash
python3 scripts/build_index.py
```

## 离线评测

```bash
python3 scripts/evaluate.py
```

输出文件：

- `data/eval/predictions.csv`

## 当前局限

- 图片输入目前只做了接入和校验，没有接入真正的视觉理解模型
- `page` 字段暂时为空，因为原始手册文本没有稳定页码信息
- 即使已经支持可选本地 LLM 生成，当前默认检索仍是纯 Python 混合检索，还没有向量召回和 reranker

## 后续建议

- 接入视觉模型，把图片识别结果转成检索 query
- 接入 reranker 提高 top-k 证据质量
- 接入大模型做更自然的客服化表达，但继续保留证据约束
- 针对公开题补一套 case 级错误分析
