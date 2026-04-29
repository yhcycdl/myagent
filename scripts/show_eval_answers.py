#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print selected answers from an evaluation diagnostics CSV.")
    parser.add_argument("diagnostics_csv", type=Path)
    parser.add_argument("--ids", nargs="*", default=[], help="Question ids to print. If omitted, print all rows.")
    parser.add_argument("--chars", type=int, default=1200)
    args = parser.parse_args()

    selected = set(args.ids)
    with args.diagnostics_csv.open(encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        for row in rows:
            if selected and row.get("id") not in selected:
                continue
            answer = (row.get("answer") or "").replace("\n", " ")
            question = row.get("question") or ""
            status = row.get("coverage_gap_status") or ""
            reason = row.get("refusal_reason") or ""
            print(f'{row.get("id")} {status} {reason} {question}')
            print(answer[: args.chars])
            print("-" * 80)


if __name__ == "__main__":
    main()
