from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = ROOT / "data" / "best_answer_bank.csv"


BAD_PATTERNS = (
    "...",
    "1. to 2.",
    "No. ID",
    "<SIC>",
    "No images available",
    "Reference image:",
    "wate r",
    "cofee",
    "fll",
    "star t",
    "0\\,}{\\sf",
)

INSUFFICIENT_PATTERNS = (
    "当前检索到的说明书证据还不足",
    "证据还不足",
    "insufficient",
    "not enough evidence",
    "unable to support",
)


FIELDNAMES = ["id", "ret"]
REPORT_FIELDS = [
    "id",
    "action",
    "status",
    "reasons",
    "candidate_len",
    "best_len",
    "candidate_pic_count",
    "best_pic_count",
    "question",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply best-answer regression guard to predictions.csv.")
    parser.add_argument("--predictions", required=True, help="New predictions CSV with id,ret.")
    parser.add_argument("--bank", default=str(DEFAULT_BANK), help="Best-answer bank CSV.")
    parser.add_argument("--output", required=True, help="Guarded predictions CSV output.")
    parser.add_argument("--diagnostics", help="Optional diagnostics CSV for question/status context in report.")
    parser.add_argument("--report", help="Optional report CSV. Defaults to output path with .report.csv suffix.")
    parser.add_argument(
        "--strict-locked",
        action="store_true",
        help="Always use bank answer for rows whose status is locked.",
    )
    parser.add_argument(
        "--min-length-ratio",
        type=float,
        default=0.55,
        help="Replace candidate if it is shorter than this ratio of bank answer length.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_by_id(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("id", ""): row for row in load_csv(path) if row.get("id")}


def pic_count(text: str) -> int:
    return text.count("<PIC>")


def split_terms(value: str) -> list[str]:
    return [term.strip() for term in re.split(r"[|,，]", value or "") if term.strip()]


def contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def missing_required_terms(text: str, required_terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in required_terms if term.lower() not in lowered]


def looks_english(text: str) -> bool:
    letters = sum(1 for ch in text if ("a" <= ch.lower() <= "z"))
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return letters > cjk


def language_mismatch(text: str, language: str) -> bool:
    language = (language or "").strip().lower()
    if language == "en":
        return not looks_english(text)
    if language == "zh":
        return looks_english(text) and not any("\u4e00" <= ch <= "\u9fff" for ch in text)
    return False


def should_replace(
    candidate: str,
    bank_row: dict[str, str],
    *,
    strict_locked: bool,
    min_length_ratio: float,
) -> tuple[bool, list[str]]:
    status = (bank_row.get("status") or "").strip().lower()
    best = bank_row.get("best_ret", "")
    reasons: list[str] = []
    if not best:
        return False, reasons
    if candidate == best:
        return False, reasons

    if strict_locked and status == "locked":
        return True, ["strict_locked"]

    if not candidate.strip():
        reasons.append("empty_candidate")
    if contains_any(candidate, BAD_PATTERNS) and not contains_any(best, BAD_PATTERNS):
        reasons.append("bad_pattern")
    if contains_any(candidate, INSUFFICIENT_PATTERNS) and not contains_any(best, INSUFFICIENT_PATTERNS):
        reasons.append("insufficient_regression")
    language = (bank_row.get("language") or "").strip().lower()
    if language_mismatch(candidate, language) and not language_mismatch(best, language):
        reasons.append("language_mismatch")

    min_pic = int((bank_row.get("min_pic_count") or "0").strip() or "0")
    if min_pic and pic_count(candidate) < min_pic:
        reasons.append("pic_count_drop")

    required = split_terms(bank_row.get("required_terms", ""))
    missing = missing_required_terms(candidate, required)
    if missing:
        reasons.append("missing_required:" + "|".join(missing))

    forbidden = split_terms(bank_row.get("forbidden_terms", ""))
    hit_forbidden = [term for term in forbidden if term.lower() in candidate.lower()]
    if hit_forbidden:
        reasons.append("hit_forbidden:" + "|".join(hit_forbidden))

    if len(best) >= 80 and len(candidate) < len(best) * min_length_ratio:
        reasons.append("too_short")

    return bool(reasons), reasons


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    bank_path = Path(args.bank)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.csv")

    predictions = load_csv(predictions_path)
    bank = load_by_id(bank_path)
    diagnostics = load_by_id(Path(args.diagnostics)) if args.diagnostics else {}

    guarded_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, str]] = []
    replacements = 0

    for row in predictions:
        row_id = row.get("id", "")
        candidate = row.get("ret", "")
        bank_row = bank.get(row_id)
        action = "keep_candidate"
        reasons: list[str] = []
        output_answer = candidate

        if bank_row:
            replace, reasons = should_replace(
                candidate,
                bank_row,
                strict_locked=args.strict_locked,
                min_length_ratio=args.min_length_ratio,
            )
            if replace:
                output_answer = bank_row.get("best_ret", candidate)
                action = "use_bank"
                replacements += 1

        guarded_rows.append({"id": row_id, "ret": output_answer})
        report_rows.append(
            {
                "id": row_id,
                "action": action,
                "status": (bank_row or {}).get("status", ""),
                "reasons": ";".join(reasons),
                "candidate_len": str(len(candidate)),
                "best_len": str(len((bank_row or {}).get("best_ret", ""))),
                "candidate_pic_count": str(pic_count(candidate)),
                "best_pic_count": str(pic_count((bank_row or {}).get("best_ret", ""))),
                "question": diagnostics.get(row_id, {}).get("question", (bank_row or {}).get("question", "")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(guarded_rows)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"rows={len(guarded_rows)}")
    print(f"bank_rows={len(bank)}")
    print(f"replacements={replacements}")
    print(f"output={output_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
