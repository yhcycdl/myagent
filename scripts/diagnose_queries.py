from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.chat_service import ChatService


QUESTION_FILE = ROOT / "question_public.csv"


def load_questions(path: Path) -> dict[str, str]:
    questions: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            question_id = row[0].strip()
            question_parts = [part.strip() for part in row[1:] if part.strip()]
            question = "\n".join(question_parts).strip().strip('"')
            if question_id and question:
                questions[question_id] = question
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose retrieval and generation traces for selected questions.")
    parser.add_argument("--ids", nargs="*", help="Question IDs from question_public.csv, e.g. 242 246 350")
    parser.add_argument("--question", help="A raw question string to diagnose directly")
    parser.add_argument("--session-id", default="diag_cli", help="Session ID prefix for diagnosis runs")
    parser.add_argument("--output", help="Optional JSON output file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ids and not args.question:
        raise SystemExit("Provide either --ids or --question")

    service = ChatService()
    diagnostics: list[dict] = []

    if args.question:
        diagnostics.append(
            {
                "question_id": None,
                **service.diagnose(question=args.question, session_id=f"{args.session_id}_raw"),
            }
        )

    if args.ids:
        questions = load_questions(QUESTION_FILE)
        for question_id in args.ids:
            question = questions.get(question_id)
            if not question:
                diagnostics.append(
                    {
                        "question_id": question_id,
                        "error": "question_id_not_found",
                    }
                )
                continue
            diagnostics.append(
                {
                    "question_id": question_id,
                    **service.diagnose(question=question, session_id=f"{args.session_id}_{question_id}"),
                }
            )

    payload = diagnostics if len(diagnostics) > 1 else diagnostics[0]
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
