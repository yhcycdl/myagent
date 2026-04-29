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
DEFAULT_DIAGNOSTICS = ROOT / "data" / "eval" / "diagnostics.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retriever/reranker training data from current diagnostics and live diagnose runs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for generated jsonl files.")
    parser.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS), help="Existing diagnostics.csv used for filtering.")
    parser.add_argument("--ids", nargs="*", help="Only include these question ids.")
    parser.add_argument("--coverage-gap-status", dest="coverage_gap_statuses", nargs="*", help="Filter by existing diagnostics coverage_gap_status.")
    parser.add_argument("--english-only", action="store_true", help="Restrict to english_dominant=true rows in diagnostics.")
    parser.add_argument("--limit", type=int, help="Optional limit after filtering.")
    parser.add_argument("--max-negatives", type=int, default=4, help="Max hard negatives per sub-question.")
    parser.add_argument("--min-positive-score", type=float, default=0.25, help="Skip answered examples whose selected positive score is lower than this.")
    parser.add_argument(
        "--allowed-positive-types",
        nargs="*",
        default=["general", "step", "list", "component", "menu", "warranty"],
        help="Allowed positive chunk types.",
    )
    parser.add_argument(
        "--require-manual-alignment",
        action="store_true",
        help="Require the positive manual to align with product hint / boosted manuals when available.",
    )
    parser.add_argument(
        "--min-score-gap",
        type=float,
        default=0.03,
        help="Minimum positive-minus-negative score gap for exported examples.",
    )
    return parser.parse_args()


def iter_questions(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
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
                rows.append((question_id, question))
    return rows


def load_diagnostics(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def select_questions(args: argparse.Namespace) -> list[tuple[str, str]]:
    questions = iter_questions(QUESTION_FILE)
    selected_ids: set[str] = {question_id for question_id, _ in questions}
    diagnostics = load_diagnostics(Path(args.diagnostics))

    if args.ids:
        selected_ids &= set(args.ids)

    if args.coverage_gap_statuses:
        wanted = set(args.coverage_gap_statuses)
        selected_ids &= {
            question_id
            for question_id, row in diagnostics.items()
            if row.get("coverage_gap_status") in wanted
        }

    if args.english_only:
        selected_ids &= {
            question_id
            for question_id, row in diagnostics.items()
            if row.get("english_dominant") == "true"
        }

    filtered = [(question_id, question) for question_id, question in questions if question_id in selected_ids]
    if args.limit is not None:
        filtered = filtered[: max(args.limit, 0)]
    return filtered


def choose_positive_chunk(sub_question: str, retrieval_trace: dict, evidence_preview: dict) -> dict | None:
    rerank = retrieval_trace.get("rerank_top_k") or []
    fusion = retrieval_trace.get("fusion_top_k") or []
    evidence = evidence_preview.get("evidence") or []

    if evidence:
        return evidence[0]
    if rerank:
        return rerank[0]
    if fusion:
        return fusion[0]
    return None


def choose_negative_chunks(
    positive_chunk_id: str,
    retrieval_trace: dict,
    max_negatives: int,
) -> list[dict]:
    negatives: list[dict] = []
    seen: set[str] = {positive_chunk_id}

    for bucket_name in ("rerank_top_k", "fusion_top_k"):
        for item in retrieval_trace.get(bucket_name) or []:
            chunk_id = item.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            negatives.append(item)
            if len(negatives) >= max_negatives:
                return negatives
    return negatives


def infer_product_context(
    service: ChatService,
    diagnostic: dict,
    question: str,
    sub_question: str,
    retrieval_trace: dict,
) -> tuple[str | None, set[str]]:
    query_variants = retrieval_trace.get("query_variants", {})
    query_rewrites = diagnostic.get("query_rewrite", [])
    multimodal_hint = diagnostic.get("multimodal_insight", {}).get("product_hint")

    texts = [
        question,
        sub_question,
        query_variants.get("query_en", ""),
        query_variants.get("query_zh", ""),
        query_variants.get("query_mix", ""),
    ]
    for rewrite in query_rewrites:
        texts.extend(
            [
                rewrite.get("query_en", ""),
                rewrite.get("query_zh", ""),
                rewrite.get("query_mix", ""),
                " ".join(rewrite.get("search_keywords", []) or []),
            ]
        )

    inferred_product = multimodal_hint or service._infer_product_hint_from_text(texts)
    expected_manuals: set[str] = set(retrieval_trace.get("boost_manuals") or [])
    if inferred_product:
        expected_manuals.update(service._resolve_manual_hints_from_product(inferred_product))
    return inferred_product, expected_manuals


def positive_is_trustworthy(
    positive: dict,
    negatives: list[dict],
    expected_manuals: set[str],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if positive.get("chunk_type") not in set(args.allowed_positive_types):
        return False, "disallowed_positive_type"

    if args.require_manual_alignment:
        if expected_manuals and positive.get("manual_name") not in expected_manuals:
            return False, "manual_mismatch"

    if negatives:
        best_negative = max(item.get("score", 0.0) for item in negatives)
        if positive.get("score", 0.0) - best_negative < args.min_score_gap:
            return False, "weak_margin"

    title = str(positive.get("section_title", ""))
    if any(token in title for token in ("目录", "前言", "目标", "重要信息", "概述")):
        return False, "generic_positive_title"

    return True, ""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever_path = output_dir / "retriever_train.jsonl"
    reranker_path = output_dir / "reranker_train.jsonl"
    gap_path = output_dir / "gap_cases.jsonl"
    summary_path = output_dir / "summary.json"

    service = ChatService()
    selected_questions = select_questions(args)

    retriever_rows: list[dict] = []
    reranker_rows: list[dict] = []
    gap_rows: list[dict] = []
    counters: Counter[str] = Counter()

    for question_id, question in selected_questions:
        diagnostic = service.diagnose(question=question, session_id=f"train_{question_id}")
        coverage = diagnostic["coverage_gap"]
        counters[f"coverage::{coverage['status']}"] += 1

        if coverage["status"] != "none":
            gap_rows.append(
                {
                    "id": question_id,
                    "question": question,
                    "coverage_gap_status": coverage["status"],
                    "coverage_gap_reason": coverage["reason"],
                    "query_rewrite": diagnostic.get("query_rewrite", []),
                }
            )
            continue

        retrieval_traces = diagnostic.get("retrieval", [])
        evidence_previews = diagnostic.get("generation", {}).get("evidence_preview", [])
        preview_by_subq = {item["sub_question"]: item for item in evidence_previews}

        for retrieval_trace in retrieval_traces:
            sub_question = retrieval_trace["sub_question"]
            evidence_preview = preview_by_subq.get(sub_question, {"evidence": []})
            inferred_product_hint, expected_manuals = infer_product_context(
                service=service,
                diagnostic=diagnostic,
                question=question,
                sub_question=sub_question,
                retrieval_trace=retrieval_trace,
            )
            positive = choose_positive_chunk(sub_question, retrieval_trace, evidence_preview)
            if positive is None:
                counters["skipped::no_positive"] += 1
                continue
            if positive.get("score", 0.0) < args.min_positive_score:
                counters["skipped::weak_positive"] += 1
                continue

            negatives = choose_negative_chunks(
                positive_chunk_id=positive["chunk_id"],
                retrieval_trace=retrieval_trace,
                max_negatives=args.max_negatives,
            )
            if not negatives:
                counters["skipped::no_negatives"] += 1
                continue

            trusted, reason = positive_is_trustworthy(
                positive=positive,
                negatives=negatives,
                expected_manuals=expected_manuals,
                args=args,
            )
            if not trusted:
                counters[f"skipped::{reason}"] += 1
                continue

            retriever_rows.append(
                {
                    "id": question_id,
                    "question": question,
                    "sub_question": sub_question,
                    "english_dominant": diagnostic.get("english_dominant", False),
                    "product_hint": inferred_product_hint,
                    "expected_manuals": sorted(expected_manuals),
                    "query_variants": retrieval_trace.get("query_variants", {}),
                    "positive": positive,
                    "negatives": negatives,
                }
            )

            for negative in negatives:
                reranker_rows.append(
                    {
                        "id": question_id,
                        "question": question,
                        "sub_question": sub_question,
                        "query": retrieval_trace.get("query_variants", {}).get("query_mix")
                        or retrieval_trace.get("query_variants", {}).get("query_zh")
                        or retrieval_trace.get("query_variants", {}).get("query_en")
                        or sub_question,
                        "positive": positive,
                        "negative": negative,
                    }
                )

            counters["exported::retriever_examples"] += 1
            counters["exported::reranker_pairs"] += len(negatives)

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(retriever_path, retriever_rows)
    write_jsonl(reranker_path, reranker_rows)
    write_jsonl(gap_path, gap_rows)

    summary = {
        "selected_questions": len(selected_questions),
        "retriever_examples": len(retriever_rows),
        "reranker_pairs": len(reranker_rows),
        "gap_cases": len(gap_rows),
        "counters": dict(sorted(counters.items())),
        "files": {
            "retriever_train": str(retriever_path),
            "reranker_train": str(reranker_path),
            "gap_cases": str(gap_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"selected_questions={len(selected_questions)}")
    print(f"retriever_examples={len(retriever_rows)}")
    print(f"reranker_pairs={len(reranker_rows)}")
    print(f"gap_cases={len(gap_rows)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
