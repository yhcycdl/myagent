from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "best_answer_bank.csv"


FIELDNAMES = [
    "id",
    "status",
    "question",
    "best_ret",
    "reason",
    "required_terms",
    "forbidden_terms",
    "min_pic_count",
    "language",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or update a best-answer bank from diagnostics.csv.")
    parser.add_argument("--diagnostics", required=True, help="Diagnostics CSV containing id, question, answer.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output answer bank CSV.")
    parser.add_argument("--existing", help="Existing answer bank to update instead of replacing from scratch.")
    parser.add_argument("--ids", nargs="*", help="Optional ids to include/update. Defaults to all diagnostics rows.")
    parser.add_argument("--status", default="locked", choices=["locked", "prefer"], help="Bank status for included rows.")
    parser.add_argument("--reason", default="", help="Reason/version label stored for every included row.")
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="When --existing is provided, keep rows not present in the selected diagnostics.",
    )
    return parser.parse_args()


def pic_count(text: str) -> int:
    return text.count("<PIC>")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def language_from_row(row: dict[str, str]) -> str:
    english = row.get("english_dominant", "").strip().lower()
    if english in {"true", "1", "yes"}:
        return "en"
    if english in {"false", "0", "no"}:
        return "zh"
    question = row.get("question", "")
    ascii_letters = sum(1 for ch in question if ("a" <= ch.lower() <= "z"))
    cjk = sum(1 for ch in question if "\u4e00" <= ch <= "\u9fff")
    return "en" if ascii_letters > cjk else "zh"


def normalize_existing_row(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in FIELDNAMES}


def main() -> None:
    args = parse_args()
    diagnostics_path = Path(args.diagnostics)
    output_path = Path(args.output)
    selected_ids = set(args.ids or [])

    bank: dict[str, dict[str, str]] = {}
    if args.existing:
        existing_path = Path(args.existing)
        if existing_path.exists():
            for row in load_csv(existing_path):
                row_id = row.get("id", "").strip()
                if row_id:
                    bank[row_id] = normalize_existing_row(row)

    if not args.preserve_existing and not args.existing:
        bank.clear()

    updated = 0
    for row in load_csv(diagnostics_path):
        row_id = row.get("id", "").strip()
        if not row_id or (selected_ids and row_id not in selected_ids):
            continue
        answer = row.get("answer") or row.get("ret") or ""
        if not answer.strip():
            continue
        bank[row_id] = {
            "id": row_id,
            "status": args.status,
            "question": row.get("question", ""),
            "best_ret": answer,
            "reason": args.reason,
            "required_terms": "",
            "forbidden_terms": "",
            "min_pic_count": str(pic_count(answer)),
            "language": language_from_row(row),
        }
        updated += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row_id in sorted(bank, key=lambda value: int(value) if value.isdigit() else value):
            writer.writerow(bank[row_id])

    print(f"bank_rows={len(bank)}")
    print(f"updated_rows={updated}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
