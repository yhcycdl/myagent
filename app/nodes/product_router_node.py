from __future__ import annotations

import os
from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.nodes.utils import normalize_terms


class ProductRouterNode(BaseNode):
    name = "product_router"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        original_scope = list(state.manual_scope)
        resolved: list[str] = []
        if state.manual_scope:
            known_manuals = {chunk.manual_name for chunk in self.service.repository.get_chunks()}
            resolved = [manual for manual in state.manual_scope if manual in known_manuals]
        if not resolved and state.product:
            resolved = self.service._resolve_manual_hints_from_product(state.product)
        if not resolved:
            resolved = self.service._resolve_allowed_manuals(
                [state.resolved_question, " ".join(state.query_variants), " ".join(state.must_terms)],
                state.product,
            )
        if not self._hard_filter_enabled() and len(resolved) > 4:
            resolved = []
        state.manual_scope = normalize_terms(resolved, limit=4)
        self.log(
            state,
            {
                "input_manual_scope": original_scope,
                "product": state.product,
                "manual_scope": state.manual_scope,
                "hard_filter": self._hard_filter_enabled(),
            },
        )
        return state

    def _hard_filter_enabled(self) -> bool:
        return os.getenv("PRODUCT_SCOPE_HARD_FILTER", "1").strip().lower() in {"1", "true", "yes", "on"}

