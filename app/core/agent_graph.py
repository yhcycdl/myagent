from __future__ import annotations

import os
from typing import Any

from app.core.agent_state import AgentState
from app.nodes.answer_generator_node import AnswerGeneratorNode
from app.nodes.answer_verifier_node import AnswerVerifierNode
from app.nodes.context_resolver_node import ContextResolverNode
from app.nodes.evidence_judge_node import EvidenceJudgeNode
from app.nodes.fact_extractor_node import FactExtractorNode
from app.nodes.fallback_node import FallbackNode
from app.nodes.final_response_node import FinalResponseNode
from app.nodes.image_binder_node import ImageBinderNode
from app.nodes.planner_node import PlannerNode
from app.nodes.product_router_node import ProductRouterNode
from app.nodes.rerank_node import RerankNode
from app.nodes.retrieval_node import RetrievalNode
from app.nodes.retry_node import RetryNode
from app.nodes.utils import result_to_dict
from app.services.preprocess import is_general_support_question, is_manual_access_question


class AgentGraph:
    def __init__(self, service: Any) -> None:
        self.service = service
        self.context_resolver = ContextResolverNode(service)
        self.planner = PlannerNode(service)
        self.product_router = ProductRouterNode(service)
        self.retriever = RetrievalNode(service)
        self.reranker = RerankNode(service)
        self.evidence_judge = EvidenceJudgeNode(service)
        self.retry = RetryNode(service)
        self.fact_extractor = FactExtractorNode(service)
        self.image_binder = ImageBinderNode(service)
        self.answer_generator = AnswerGeneratorNode(service)
        self.answer_verifier = AnswerVerifierNode(service)
        self.fallback = FallbackNode(service)
        self.final_response = FinalResponseNode(service)

    def run(self, question: str, session_id: str | None = None, images: list[str] | None = None) -> AgentState:
        state = AgentState(question=question, session_id=session_id, images=images or [])
        state.max_retry = self._max_retry()

        state = self.context_resolver.run(state)
        resolved = state.resolved_question or state.question
        if is_general_support_question(resolved) or is_manual_access_question(resolved):
            state = self.fallback.run(state)
            return self.final_response.run(state)

        state = self.planner.run(state)
        state = self.product_router.run(state)

        for attempt in range(state.max_retry):
            state.retry_count = attempt
            state = self.retriever.run(state)
            state = self.reranker.run(state)
            state = self.evidence_judge.run(state)
            if state.accepted_evidence:
                break
            if attempt < state.max_retry - 1:
                state = self.retry.run(state)

        direct_intent = bool(state.accepted_evidence and str(state.accepted_evidence[0].get("chunk_id", "")).startswith("direct:"))
        if not state.accepted_evidence and not direct_intent and self._weak_evidence_enabled(state):
            state = self._accept_weak_evidence(state)

        if not state.accepted_evidence and not direct_intent:
            state = self.fallback.run(state)
            return self.final_response.run(state)

        state = self.fact_extractor.run(state)
        state = self.image_binder.run(state)
        state = self.answer_generator.run(state)
        state = self.answer_verifier.run(state)
        if state.verifier_result.get("decision") != "pass":
            repaired = self._repair_answer(state)
            if repaired is not state:
                state = repaired
        return self.final_response.run(state)

    def _repair_answer(self, state: AgentState) -> AgentState:
        if state.raw_accepted_evidence:
            return state
        state.fallback_reason = state.verifier_result.get("feedback") or "verifier_failed"
        return self.fallback.run(state)

    def _weak_evidence_enabled(self, state: AgentState) -> bool:
        if not getattr(self.service.settings, "weak_evidence_fallback_enabled", True):
            return False
        question_type = state.question_type or ""
        return question_type.startswith("manual") or question_type == "troubleshooting" or bool(state.product)

    def _accept_weak_evidence(self, state: AgentState) -> AgentState:
        candidates = list(state.raw_reranked_candidates or state.raw_candidates or [])
        if not candidates:
            return state
        accepted = candidates[: min(3, len(candidates))]
        state.raw_accepted_evidence = accepted
        state.accepted_evidence = []
        for result in accepted:
            payload = result_to_dict(result)
            payload.update(
                {
                    "decision": "accept",
                    "reason": "weak_evidence_after_failed_judge",
                    "confidence": max(0.35, min(float(getattr(result, "score", 0.0)), 0.5)),
                }
            )
            state.accepted_evidence.append(payload)
        state.evidence_confidence = max((item.get("confidence", 0.0) for item in state.accepted_evidence), default=0.35)
        state.fallback_reason = "weak_evidence_after_failed_judge"
        state.trace.append(
            {
                "node": "weak_evidence",
                "data": {
                    "accepted": [item.get("chunk_id") for item in state.accepted_evidence],
                    "reason": state.fallback_reason,
                },
            }
        )
        return state

    def _max_retry(self) -> int:
        raw = os.getenv("RETRIEVAL_MAX_RETRY", "3")
        try:
            value = int(raw)
        except ValueError:
            return 3
        return max(1, min(value, 5))
