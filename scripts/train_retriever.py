from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight dual-encoder retriever from retriever_train.jsonl.")
    parser.add_argument("--train-file", default=str(ROOT / "data" / "train" / "retriever_train.jsonl"))
    parser.add_argument("--model", required=True, help="Base embedding model path, e.g. ~/models/bge-m3")
    parser.add_argument("--output-dir", required=True, help="Directory to save the fine-tuned retriever")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-train-examples", type=int, help="Optional cap for quick experiments.")
    return parser.parse_args()


@dataclass(slots=True)
class RetrieverExample:
    question_id: str
    query: str
    positive: str
    negatives: list[str]


class RetrieverDataset(Dataset):
    def __init__(self, rows: list[RetrieverExample]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RetrieverExample:
        return self.rows[index]


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


def read_examples(path: Path) -> list[RetrieverExample]:
    rows: list[RetrieverExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            negatives = [format_doc(doc) for doc in payload.get("negatives", []) if doc]
            if not negatives:
                continue
            rows.append(
                RetrieverExample(
                    question_id=str(payload["id"]),
                    query=choose_query(payload),
                    positive=format_doc(payload["positive"]),
                    negatives=negatives,
                )
            )
    return rows


def split_examples(
    examples: list[RetrieverExample],
    val_ratio: float,
    seed: int,
) -> tuple[list[RetrieverExample], list[RetrieverExample]]:
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    val_size = max(1, int(len(rows) * val_ratio)) if len(rows) >= 6 else 0
    if val_size == 0:
        return rows, []
    return rows[val_size:], rows[:val_size]


def collate_fn(tokenizer, max_length: int):
    def _collate(batch: list[RetrieverExample]) -> dict[str, object]:
        queries = [item.query for item in batch]
        positives = [item.positive for item in batch]
        negatives = [item.negatives for item in batch]
        max_negatives = max(len(items) for items in negatives)
        padded_negatives = [items + [""] * (max_negatives - len(items)) for items in negatives]
        flat_negatives = [text for items in padded_negatives for text in items]

        query_tokens = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        positive_tokens = tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        negative_tokens = tokenizer(
            flat_negatives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "query_tokens": query_tokens,
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "negative_counts": [len(items) for items in negatives],
            "max_negatives": max_negatives,
        }

    return _collate


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        "query_tokens": {key: value.to(device) for key, value in batch["query_tokens"].items()},
        "positive_tokens": {key: value.to(device) for key, value in batch["positive_tokens"].items()},
        "negative_tokens": {key: value.to(device) for key, value in batch["negative_tokens"].items()},
        "negative_counts": batch["negative_counts"],
        "max_negatives": batch["max_negatives"],
    }


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled


def encode_texts(model, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(**tokens)
    pooled = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
    return F.normalize(pooled, p=2, dim=1)


def compute_batch_loss(model, batch: dict[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    query_embeddings = encode_texts(model, batch["query_tokens"])
    positive_embeddings = encode_texts(model, batch["positive_tokens"])
    negative_embeddings = encode_texts(model, batch["negative_tokens"])

    batch_size = query_embeddings.shape[0]
    max_negatives = int(batch["max_negatives"])
    negative_embeddings = negative_embeddings.view(batch_size, max_negatives, -1)

    positive_scores = (query_embeddings * positive_embeddings).sum(dim=1, keepdim=True)
    negative_scores = torch.einsum("bd,bnd->bn", query_embeddings, negative_embeddings)

    mask = torch.zeros_like(negative_scores, dtype=torch.bool)
    for index, count in enumerate(batch["negative_counts"]):
        if count < max_negatives:
            mask[index, count:] = True
    negative_scores = negative_scores.masked_fill(mask, -1e4)

    logits = torch.cat([positive_scores, negative_scores], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=query_embeddings.device)
    loss = F.cross_entropy(logits, labels)
    return loss, logits


def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    total = 0

    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            loss, logits = compute_batch_loss(model, batch)
            losses.append(loss.item())
            rankings = logits.argsort(dim=1, descending=True)
            for row in rankings:
                ordered = row.tolist()
                if 0 in ordered[:1]:
                    hits_at_1 += 1
                if 0 in ordered[:3]:
                    hits_at_3 += 1
                if 0 in ordered[:5]:
                    hits_at_5 += 1
                total += 1

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    metrics = {
        "recall@1": hits_at_1 / total if total else 0.0,
        "recall@3": hits_at_3 / total if total else 0.0,
        "recall@5": hits_at_5 / total if total else 0.0,
    }
    return avg_loss, metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_examples(train_file)
    if args.max_train_examples is not None:
        examples = examples[: max(args.max_train_examples, 0)]
    if not examples:
        raise SystemExit("no training examples found")

    train_rows, val_rows = split_examples(examples, args.val_ratio, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)

    train_loader = DataLoader(
        RetrieverDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        RetrieverDataset(val_rows),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, args.max_length),
    ) if val_rows else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)) * max(1, args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_recall_at_1 = -1.0
    best_path = output_dir / "best"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(raw_batch, device)
            loss, _ = compute_batch_loss(model, batch)
            loss = loss / max(1, args.grad_accum)
            loss.backward()
            running_loss += loss.item()

            if step % max(1, args.grad_accum) == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = running_loss / max(1, len(train_loader))
        if val_loader is not None:
            val_loss, metrics = evaluate(model, val_loader, device)
        else:
            val_loss, metrics = 0.0, {"recall@1": 0.0, "recall@3": 0.0, "recall@5": 0.0}

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} recall@1={metrics['recall@1']:.4f} "
            f"recall@3={metrics['recall@3']:.4f} recall@5={metrics['recall@5']:.4f} "
            f"steps={global_step}"
        )

        current_score = metrics["recall@1"]
        if val_loader is None or current_score >= best_recall_at_1:
            best_recall_at_1 = current_score
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    model.save_pretrained(output_dir / "last")
    tokenizer.save_pretrained(output_dir / "last")
    print(f"saved_best={best_path}")
    print(f"saved_last={output_dir / 'last'}")


if __name__ == "__main__":
    main()
