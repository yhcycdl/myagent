from __future__ import annotations

import re
from typing import Any


def normalize_terms(terms: list[str] | tuple[str, ...] | None, *, limit: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms or []:
        text = str(term).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def result_to_dict(result: Any) -> dict[str, Any]:
    chunk = result.chunk
    text = chunk.text.strip().replace("\n", " ")
    if len(text) > 220:
        text = text[:217].rstrip("，,；; ") + "..."
    return {
        "chunk_id": chunk.chunk_id,
        "manual_name": chunk.manual_name,
        "section_title": chunk.section_title,
        "chunk_type": chunk.chunk_type,
        "score": round(float(getattr(result, "score", 0.0)), 4),
        "bm25_score": round(float(getattr(result, "bm25_score", 0.0)), 4),
        "semantic_score": round(float(getattr(result, "semantic_score", 0.0)), 4),
        "image_ids": list(getattr(chunk, "image_ids", []) or []),
        "section_image_ids": list(getattr(chunk, "section_image_ids", []) or []),
        "nearby_image_ids": list(getattr(chunk, "nearby_image_ids", []) or []),
        "parent_section_id": getattr(chunk, "parent_section_id", ""),
        "text": text,
    }


def contains_any(text: str, terms: list[str] | tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms if term)


def coverage_ratio(text: str, terms: list[str]) -> float:
    terms = [term for term in terms if term]
    if not terms:
        return 1.0
    lowered = text.lower()
    hits = sum(1 for term in terms if term.lower() in lowered)
    return hits / max(len(terms), 1)


def split_search_terms(text: str) -> list[str]:
    raw = re.split(r"[\s,，;；/、|]+", text)
    return [item.strip() for item in raw if len(item.strip()) >= 2]
