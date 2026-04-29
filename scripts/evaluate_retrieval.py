from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation against retriever_train.jsonl candidates.")
    parser.add_argument("--data-file", default=str(ROOT / "data" / "train" / "retriever_train.jsonl"))
    parser.add_argument("--model", required=True, help="Retriever checkpoint path")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional limit for quick checks")
    return parser.parse_args()


def format_doc(doc: dict) -> str:
    return "\n".join(
        part
        for part in (
            doc.get("manual_name", ""),
            doc.get("section_title", ""),
            doc.get("text_summary", ""),
        )
        if part
    )


def choose_query(payload: dict) -> str:
    variants = payload.get("query_variants", {}) or {}
    return (
        variants.get("query_mix")
        or variants.get("query_zh")
        or variants.get("query_en")
        or payload.get("sub_question")
        or payload.get("question")
        or ""
    )


def read_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            negatives = payload.get("negatives") or []
            if not negatives:
                continue
            rows.append(
                {
                    "id": str(payload["id"]),
                    "query": choose_query(payload),
                    "positive": format_doc(payload["positive"]),
                    "negatives": [format_doc(doc) for doc in negatives],
                }
            )
            if limit is not None and len(rows) >= max(limit, 0):
                break
    return rows


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled


def encode_texts(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: torch.device) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            result = model(**tokens)
            pooled = mean_pool(result.last_hidden_state, tokens["attention_mask"])
            outputs.append(F.normalize(pooled, p=2, dim=1).cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty(0, 0)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.data_file), args.limit)
    if not rows:
        raise SystemExit("no rows to evaluate")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)
    model.eval()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: list[float] = []

    for row in rows:
        docs = [row["positive"], *row["negatives"]]
        query_embedding = encode_texts(model, tokenizer, [row["query"]], args.batch_size, args.max_length, device)
        doc_embeddings = encode_texts(model, tokenizer, docs, args.batch_size, args.max_length, device)
        scores = torch.matmul(query_embedding, doc_embeddings.T).squeeze(0)
        ranking = scores.argsort(descending=True).tolist()
        positive_rank = ranking.index(0) + 1
        reciprocal_ranks.append(1.0 / positive_rank)
        if positive_rank <= 1:
            hits_at_1 += 1
        if positive_rank <= 3:
            hits_at_3 += 1
        if positive_rank <= 5:
            hits_at_5 += 1

    total = len(rows)
    print(f"examples={total}")
    print(f"recall@1={hits_at_1 / total:.4f}")
    print(f"recall@3={hits_at_3 / total:.4f}")
    print(f"recall@5={hits_at_5 / total:.4f}")
    print(f"mrr={sum(reciprocal_ranks) / total:.4f}")


if __name__ == "__main__":
    main()
