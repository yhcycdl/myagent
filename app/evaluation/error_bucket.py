from __future__ import annotations


def bucket_answer(answer: str) -> list[str]:
    lowered = answer.lower()
    buckets: list[str] = []
    if "当前检索到的说明书证据" in answer:
        buckets.append("insufficient_evidence")
    if "<pic>" in lowered and "manual" not in lowered:
        buckets.append("pic_without_visible_id")
    if any(term in lowered for term in ("spark plug", "electrode")):
        buckets.append("maintenance_contamination")
    if any(term in answer for term in ("...", "1. to 2.", "No. ID", "<SIC>")):
        buckets.append("dirty_output")
    return buckets or ["unknown"]
