from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    question: str
    session_id: str | None = None
    images: list[str] = field(default_factory=list)

    resolved_question: str = ""
    history_summary: str = ""
    image_understanding: str = ""
    image_product_hint: str | None = None
    image_query_terms: list[str] = field(default_factory=list)

    language: str = "zh"
    product: str | None = None
    manual_scope: list[str] = field(default_factory=list)
    intent: str | None = None
    question_type: str | None = None

    action_terms: list[str] = field(default_factory=list)
    object_terms: list[str] = field(default_factory=list)
    must_terms: list[str] = field(default_factory=list)
    should_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    answer_expectation: str = ""

    candidates: list[dict[str, Any]] = field(default_factory=list)
    reranked_candidates: list[dict[str, Any]] = field(default_factory=list)
    raw_candidates: list[Any] = field(default_factory=list, repr=False)
    raw_reranked_candidates: list[Any] = field(default_factory=list, repr=False)

    accepted_evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected_evidence: list[dict[str, Any]] = field(default_factory=list)
    raw_accepted_evidence: list[Any] = field(default_factory=list, repr=False)
    evidence_confidence: float = 0.0

    facts: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)

    image_ids: list[str] = field(default_factory=list)

    answer: str = ""
    references: list[dict[str, Any]] = field(default_factory=list)
    related_images: list[dict[str, Any]] = field(default_factory=list)
    used_manuals: list[str] = field(default_factory=list)
    used_sections: list[str] = field(default_factory=list)
    verifier_result: dict[str, Any] = field(default_factory=dict)

    retry_count: int = 0
    max_retry: int = 3
    fallback_reason: str | None = None

    trace: list[dict[str, Any]] = field(default_factory=list)
    final_response: dict[str, Any] = field(default_factory=dict)
