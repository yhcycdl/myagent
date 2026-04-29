#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print compact generation candidates from diagnose_queries output.")
    parser.add_argument("diagnostics_json", type=Path)
    parser.add_argument("--ids", nargs="*", default=[], help="Question ids to print. If omitted, print all.")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    selected = set(args.ids)
    data = json.loads(args.diagnostics_json.read_text(encoding="utf-8"))
    for item in data:
        qid = str(item.get("question_id", ""))
        if selected and qid not in selected:
            continue
        print(f"\nID {qid} {item.get('question', '')}")
        for trace in item.get("retrieval", []):
            rows = []
            for result in trace.get("generation_top_k", [])[: args.top]:
                rows.append((result.get("chunk_id"), (result.get("section_title") or "")[:80]))
            print(rows)


if __name__ == "__main__":
    main()
