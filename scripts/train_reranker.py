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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight cross-encoder reranker from reranker_train.jsonl.")
    parser.add_argument("--train-file", default=str(ROOT / "data" / "train" / "reranker_train.jsonl"))
    parser.add_argument("--model", required=True, help="Base reranker model path, e.g. ~/models/bge-reranker-v2-m3")
    parser.add_argument("--output-dir", required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3)
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
    parser.add_argument("--max-train-pairs", type=int, help="Optional cap for quick experiments.")
    return parser.parse_args()


@dataclass(slots=True)
class PairExample:
    query: str
    positive: str
    negative: str


class PairDataset(Dataset):
    def __init__(self, rows: list[PairExample]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> PairExample:
        return self.rows[index]


def read_examples(path: Path) -> list[PairExample]:
    examples: list[PairExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            query = payload["query"]
            positive = format_doc(payload["positive"])
            negative = format_doc(payload["negative"])
            examples.append(PairExample(query=query, positive=positive, negative=negative))
    return examples


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


def split_examples(examples: list[PairExample], val_ratio: float, seed: int) -> tuple[list[PairExample], list[PairExample]]:
    rows = list(examples)
    random.Random(seed).shuffle(rows)
    val_size = max(1, int(len(rows) * val_ratio)) if len(rows) >= 5 else 0
    if val_size == 0:
        return rows, []
    return rows[val_size:], rows[:val_size]


def collate_fn(tokenizer, max_length: int):
    def _collate(batch: list[PairExample]) -> dict[str, torch.Tensor]:
        queries = [item.query for item in batch]
        positives = [item.positive for item in batch]
        negatives = [item.negative for item in batch]

        pos = tokenizer(
            queries,
            positives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        neg = tokenizer(
            queries,
            negatives,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
        }

    return _collate


def compute_scores(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    if logits.ndim == 2 and logits.shape[-1] == 1:
        return logits.squeeze(-1)
    if logits.ndim == 2:
        return logits[:, -1]
    return logits


def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)

            pos_scores = compute_scores(model, pos_input_ids, pos_attention_mask)
            neg_scores = compute_scores(model, neg_input_ids, neg_attention_mask)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            losses.append(loss.item())
            correct += int((pos_scores > neg_scores).sum().item())
            total += pos_scores.numel()

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = read_examples(train_file)
    if args.max_train_pairs is not None:
        examples = examples[: max(args.max_train_pairs, 0)]
    if not examples:
        raise SystemExit("no training examples found")

    train_rows, val_rows = split_examples(examples, args.val_ratio, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)

    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        PairDataset(val_rows),
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

    best_val_loss = float("inf")
    best_path = output_dir / "best"

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_attention_mask = batch["pos_attention_mask"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_attention_mask = batch["neg_attention_mask"].to(device)

            pos_scores = compute_scores(model, pos_input_ids, pos_attention_mask)
            neg_scores = compute_scores(model, neg_input_ids, neg_attention_mask)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
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
            val_loss, val_acc = evaluate(model, val_loader, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} steps={global_step}"
        )

        if val_loader is None or val_loss <= best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)

    model.save_pretrained(output_dir / "last")
    tokenizer.save_pretrained(output_dir / "last")
    print(f"saved_best={best_path}")
    print(f"saved_last={output_dir / 'last'}")


if __name__ == "__main__":
    main()
