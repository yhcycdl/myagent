from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.nodes.utils import normalize_terms


class RetryNode(BaseNode):
    name = "retry"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        reject_terms: list[str] = []
        for rejected in state.rejected_evidence[:8]:
            reject_terms.extend(str(term) for term in rejected.get("exclude_terms_hit", []) if term)
            if rejected.get("reason") in {"insufficient_term_coverage", "low_confidence"}:
                text = " ".join(
                    str(value)
                    for value in (
                        rejected.get("section_title"),
                        rejected.get("text"),
                    )
                    if value
                )
                reject_terms.extend(self._weak_terms(text))

        state.exclude_terms = normalize_terms([*state.exclude_terms, *reject_terms], limit=20)
        expanded_query = " ".join(
            term
            for term in [state.resolved_question or state.question, *state.must_terms, *state.object_terms, *state.action_terms]
            if term
        )
        if expanded_query and expanded_query not in state.query_variants:
            state.query_variants.insert(0, expanded_query)
        state.query_variants = normalize_terms(state.query_variants, limit=6)
        self.log(
            state,
            {
                "retry_count": state.retry_count,
                "query_variants": state.query_variants,
                "exclude_terms": state.exclude_terms,
            },
        )
        return state

    def _weak_terms(self, text: str) -> list[str]:
        lowered = text.lower()
        terms: list[str] = []
        for term in ("spark plug", "electrode", "warranty", "fcc", "emission", "火花塞", "保修", "目录"):
            if term in lowered:
                terms.append(term)
        return terms
