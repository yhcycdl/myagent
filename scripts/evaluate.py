from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.chat_service import ChatService


QUESTION_FILE = ROOT / "question_public.csv"
OUTPUT_FILE = ROOT / "data" / "eval" / "predictions.csv"
DIAGNOSTIC_FILE = ROOT / "data" / "eval" / "diagnostics.csv"
SUMMARY_FILE = ROOT / "data" / "eval" / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full or filtered evaluation over question_public.csv.")
    parser.add_argument("--ids", nargs="*", help="Only evaluate these question ids, e.g. 280 284 293")
    parser.add_argument(
        "--coverage-gap-status",
        dest="coverage_gap_statuses",
        nargs="*",
        help="Reuse ids from the previous diagnostics.csv by coverage_gap_status, e.g. retrieval_failure",
    )
    parser.add_argument(
        "--refusal-reason",
        dest="refusal_reasons",
        nargs="*",
        help="Reuse ids from the previous diagnostics.csv by refusal_reason",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Restrict to rows marked english_dominant=true in the previous diagnostics.csv",
    )
    parser.add_argument(
        "--contains",
        nargs="*",
        help="Question text must contain one of these substrings (case-insensitive), e.g. camera jetski",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional max number of selected questions after filtering.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Write filtered outputs to predictions_<prefix>.csv, diagnostics_<prefix>.csv, summary_<prefix>.json",
    )
    return parser.parse_args()


def iter_questions(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            question_id = row[0].strip()
            question_parts = [part.strip() for part in row[1:] if part.strip()]
            question = "\n".join(question_parts)
            if question_id and question:
                yield question_id, question


def load_prior_diagnostics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def build_output_paths(prefix: str | None) -> tuple[Path, Path, Path]:
    if not prefix:
        return OUTPUT_FILE, DIAGNOSTIC_FILE, SUMMARY_FILE
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in prefix).strip("_")
    safe = safe or "subset"
    base = OUTPUT_FILE.parent
    return (
        base / f"predictions_{safe}.csv",
        base / f"diagnostics_{safe}.csv",
        base / f"summary_{safe}.json",
    )


def select_questions(args: argparse.Namespace) -> list[tuple[str, str]]:
    questions = list(iter_questions(QUESTION_FILE))
    if not any(
        [
            args.ids,
            args.coverage_gap_statuses,
            args.refusal_reasons,
            args.english_only,
            args.contains,
            args.limit,
        ]
    ):
        return questions

    selected_ids: set[str] = {question_id for question_id, _ in questions}
    prior = load_prior_diagnostics(DIAGNOSTIC_FILE)

    if args.ids:
        selected_ids &= {question_id for question_id in args.ids}

    if args.coverage_gap_statuses:
        wanted = set(args.coverage_gap_statuses)
        selected_ids &= {
            question_id
            for question_id, row in prior.items()
            if row.get("coverage_gap_status") in wanted
        }

    if args.refusal_reasons:
        wanted = set(args.refusal_reasons)
        selected_ids &= {
            question_id
            for question_id, row in prior.items()
            if row.get("refusal_reason") in wanted
        }

    if args.english_only:
        selected_ids &= {
            question_id
            for question_id, row in prior.items()
            if row.get("english_dominant") == "true"
        }

    lowered_contains = [token.lower() for token in (args.contains or []) if token.strip()]
    filtered: list[tuple[str, str]] = []
    for question_id, question in questions:
        if question_id not in selected_ids:
            continue
        if lowered_contains and not any(token in question.lower() for token in lowered_contains):
            continue
        filtered.append((question_id, question))

    if args.limit is not None:
        filtered = filtered[: max(args.limit, 0)]
    return filtered


def main() -> None:
    args = parse_args()
    output_file, diagnostic_file, summary_file = build_output_paths(args.output_prefix)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    service = ChatService()
    selected_questions = select_questions(args)

    rows: list[dict[str, str]] = []
    diagnostic_rows: list[dict[str, str]] = []
    coverage_counter: Counter[str] = Counter()
    refusal_counter: Counter[str] = Counter()
    for question_id, question in selected_questions:
        diagnostic = service.diagnose(question=question, session_id=f"eval_{question_id}")
        answer = diagnostic["generation"]["answer"]
        refusal = diagnostic["refusal"]
        coverage_gap = diagnostic.get("coverage_gap", {"status": "unknown", "reason": "missing"})

        rows.append({"id": question_id, "ret": answer})
        diagnostic_rows.append(
            {
                "id": question_id,
                "question": question,
                "english_dominant": str(diagnostic.get("english_dominant", False)).lower(),
                "answer": answer,
                "refusal": str(refusal.get("is_refusal", False)).lower(),
                "refusal_reason": refusal.get("reason", "none"),
                "coverage_gap_status": coverage_gap.get("status", "unknown"),
                "coverage_gap_reason": coverage_gap.get("reason", "unknown"),
            }
        )
        coverage_counter[coverage_gap.get("status", "unknown")] += 1
        refusal_counter[refusal.get("reason", "none")] += 1

    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(rows)

    with diagnostic_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "question",
                "english_dominant",
                "answer",
                "refusal",
                "refusal_reason",
                "coverage_gap_status",
                "coverage_gap_reason",
            ],
        )
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    summary = {
        "filters": {
            "ids": args.ids or [],
            "coverage_gap_statuses": args.coverage_gap_statuses or [],
            "refusal_reasons": args.refusal_reasons or [],
            "english_only": bool(args.english_only),
            "contains": args.contains or [],
            "limit": args.limit,
            "output_prefix": args.output_prefix,
        },
        "question_count": len(rows),
        "coverage_gap_counts": dict(sorted(coverage_counter.items())),
        "refusal_reason_counts": dict(sorted(refusal_counter.items())),
        "files": {
            "predictions": str(output_file),
            "diagnostics": str(diagnostic_file),
            "summary": str(summary_file),
        },
    }
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"selected questions: {len(selected_questions)}")
    print(f"saved predictions: {len(rows)}")
    print(f"output: {output_file}")
    print(f"diagnostics: {diagnostic_file}")
    print(f"summary: {summary_file}")
    print("coverage_gap_counts:")
    for status, count in sorted(coverage_counter.items()):
        print(f"  {status}: {count}")
    print("refusal_reason_counts:")
    for reason, count in sorted(refusal_counter.items()):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
