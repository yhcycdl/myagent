from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.chat_service import ChatService


def main() -> int:
    cases_path = Path(__file__).with_name("regression_cases.json")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    service = ChatService()
    failures: list[dict] = []
    for case in cases:
        diagnostic = service.diagnose(case["question"], session_id=f"reg_{case['case_name']}")
        answer = diagnostic["generation"]["answer"]
        notes = diagnostic.get("notes", {})
        failed_reasons: list[str] = []
        for term in case.get("must_terms", []):
            if term.lower() not in answer.lower() and term not in json.dumps(diagnostic, ensure_ascii=False):
                failed_reasons.append(f"missing_must_term:{term}")
        for term in case.get("exclude_terms", []):
            if term.lower() in answer.lower():
                failed_reasons.append(f"answer_contains_exclude_term:{term}")
        if failed_reasons:
            failures.append(
                {
                    "case_name": case["case_name"],
                    "reasons": failed_reasons,
                    "answer": answer,
                    "trace_nodes": notes.get("trace_nodes", []),
                    "coverage_gap": diagnostic.get("coverage_gap", {}),
                }
            )
    print(json.dumps({"passed": len(cases) - len(failures), "failed": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
