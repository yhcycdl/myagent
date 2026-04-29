from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.nodes.utils import result_to_dict
from app.services.preprocess import looks_english_dominant


class RetrievalNode(BaseNode):
    name = "retrieval"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        retriever = self.service._get_retriever()
        sidecar_retriever = self.service._get_sidecar_retriever()
        dense_retriever = self.service._get_dense_retriever()
        query_variants = state.query_variants or [state.resolved_question or state.question]
        boost_manuals = list(state.manual_scope)
        allowed_manuals = list(state.manual_scope) if state.manual_scope else None
        merged: dict[str, Any] = {}
        route_trace: list[dict[str, Any]] = []

        for query in query_variants[:5]:
            rewritten = self.service.query_rewriter.rewrite(query, state.product) if looks_english_dominant(query) else None
            results, variants, bm25_routes, vector_routes, hybrid_routes, fusion_results = self.service._run_retrieval(
                retriever,
                sidecar_retriever,
                dense_retriever,
                query,
                rewritten,
                boost_manuals,
                allowed_manuals=allowed_manuals,
                root_query=state.resolved_question or state.question,
            )
            for result in results:
                key = result.chunk.chunk_id
                current = merged.get(key)
                if current is None or result.score > current.score:
                    merged[key] = result
            route_trace.append(
                {
                    "query": query,
                    "variants": variants,
                    "allowed_manuals": allowed_manuals,
                    "bm25_routes": {name: [item.chunk.chunk_id for item in items[:5]] for name, items in bm25_routes.items()},
                    "vector_routes": {name: [item.chunk.chunk_id for item in items[:5]] for name, items in vector_routes.items()},
                    "hybrid_routes": {name: [item.chunk.chunk_id for item in items[:5]] for name, items in hybrid_routes.items()},
                    "fusion_top": [item.chunk.chunk_id for item in fusion_results[:8]],
                    "rerank_top": [item.chunk.chunk_id for item in results[:8]],
                }
            )

        state.raw_candidates = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        state.candidates = [result_to_dict(result) for result in state.raw_candidates[:20]]
        self.log(state, {"candidate_count": len(state.raw_candidates), "routes": route_trace})
        return state

