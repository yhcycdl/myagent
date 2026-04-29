from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.nodes.utils import result_to_dict


class RerankNode(BaseNode):
    name = "rerank"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        if not state.raw_candidates:
            state.raw_reranked_candidates = []
            state.reranked_candidates = []
            self.log(state, {"reranked_count": 0, "mode": "empty"})
            return state
        # RetrievalNode already runs the legacy hybrid/reranker path per query.
        # This node is explicit so evidence cannot flow directly into generation.
        state.raw_reranked_candidates = state.raw_candidates[:20]
        state.reranked_candidates = [result_to_dict(result) for result in state.raw_reranked_candidates]
        self.log(
            state,
            {
                "reranked_count": len(state.raw_reranked_candidates),
                "top": [item["chunk_id"] for item in state.reranked_candidates[:5]],
                "mode": "legacy_rerank_passthrough",
            },
        )
        return state

