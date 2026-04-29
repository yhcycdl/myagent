from __future__ import annotations

import os
from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.nodes.utils import contains_any, coverage_ratio, result_to_dict


class EvidenceJudgeNode(BaseNode):
    name = "evidence_judge"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        accepted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        soft_fallbacks: list[tuple[Any, dict[str, Any]]] = []
        state.accepted_evidence = []
        state.rejected_evidence = []
        state.raw_accepted_evidence = []
        threshold = float(os.getenv("EVIDENCE_ACCEPT_THRESHOLD", "0.45"))
        hard_reject = self._flag("EXCLUDE_TERMS_HARD_REJECT", True)

        for result in state.raw_reranked_candidates or state.raw_candidates:
            text = self._combined_text(result)
            exclude_hits = [term for term in state.exclude_terms if term and term.lower() in text.lower()]
            product_match = self._product_match(state, result)
            must_coverage = coverage_ratio(text, state.must_terms)
            object_coverage = coverage_ratio(text, state.object_terms)
            action_coverage = coverage_ratio(text, state.action_terms)
            answers_question = (must_coverage >= 0.35) or (object_coverage >= 0.35 and action_coverage >= 0.10)
            confidence = self._confidence(product_match, must_coverage, object_coverage, action_coverage, result.score)

            decision = "accept" if product_match and answers_question and confidence >= threshold else "reject"
            reason = "accepted"
            if not product_match:
                reason = "product_or_manual_scope_mismatch"
                decision = "reject"
            elif hard_reject and exclude_hits:
                reason = "exclude_terms_hit"
                decision = "reject"
            elif not answers_question:
                reason = "insufficient_term_coverage"
            elif confidence < threshold:
                reason = "low_confidence"

            payload = result_to_dict(result)
            payload.update(
                {
                    "decision": decision,
                    "reason": reason,
                    "confidence": round(confidence, 3),
                    "product_match": product_match,
                    "must_coverage": round(must_coverage, 3),
                    "object_coverage": round(object_coverage, 3),
                    "action_coverage": round(action_coverage, 3),
                    "exclude_terms_hit": exclude_hits,
                }
            )
            if decision == "accept":
                accepted.append(result)
                state.accepted_evidence.append(payload)
            else:
                rejected.append(payload)
                if product_match and not exclude_hits and len(soft_fallbacks) < 4:
                    soft_payload = dict(payload)
                    soft_payload["decision"] = "accept"
                    soft_payload["reason"] = f"soft_accept_after_rule_reject:{reason}"
                    soft_payload["confidence"] = max(float(soft_payload.get("confidence", 0.0)), 0.35)
                    soft_fallbacks.append((result, soft_payload))

            if len(accepted) >= 5:
                break

        if not accepted and soft_fallbacks:
            for result, payload in soft_fallbacks:
                accepted.append(result)
                state.accepted_evidence.append(payload)
        elif not accepted and self._has_direct_answer(state):
            state.accepted_evidence.append(
                {
                    "chunk_id": f"direct:{state.intent}",
                    "manual_name": "",
                    "section_title": "",
                    "text": "",
                    "score": 1.0,
                    "decision": "accept",
                    "reason": "preplanned_direct_intent",
                    "confidence": 1.0,
                }
            )
            state.evidence_confidence = 1.0
        else:
            state.evidence_confidence = max((item.get("confidence", 0.0) for item in state.accepted_evidence), default=0.0)

        state.raw_accepted_evidence = accepted
        state.rejected_evidence = rejected[:20]
        self.log(
            state,
            {
                "accepted": [item.get("chunk_id") for item in state.accepted_evidence[:8]],
                "rejected": [
                    {
                        "chunk_id": item.get("chunk_id"),
                        "reason": item.get("reason"),
                        "confidence": item.get("confidence"),
                    }
                    for item in state.rejected_evidence[:8]
                ],
                "confidence": state.evidence_confidence,
            },
        )
        return state

    def _has_direct_answer(self, state: AgentState) -> bool:
        if state.intent in self.service.generator._preplanned_direct_intents():
            return True
        question = state.resolved_question or state.question
        try:
            plan = self.service.generator._build_evidence_plan(question)
            return bool(self.service.generator._select_planned_direct_answer(plan, [], question))
        except Exception:  # noqa: BLE001 - private compatibility path must not break retrieval
            return False

    def _combined_text(self, result: Any) -> str:
        chunk = result.chunk
        return f"{chunk.manual_name} {chunk.section_title} {chunk.text}"

    def _product_match(self, state: AgentState, result: Any) -> bool:
        if state.manual_scope:
            return result.chunk.manual_name in state.manual_scope
        if not state.product:
            return True
        manuals = self.service._resolve_manual_hints_from_product(state.product)
        if manuals:
            return result.chunk.manual_name in manuals
        return not contains_any(
            f"{result.chunk.manual_name} {result.chunk.section_title}",
            self._cross_product_terms(state.product),
        )

    def _confidence(
        self,
        product_match: bool,
        must_coverage: float,
        object_coverage: float,
        action_coverage: float,
        retriever_score: float,
    ) -> float:
        score = 0.0
        if product_match:
            score += 0.30
        score += min(must_coverage, 1.0) * 0.30
        score += min(object_coverage, 1.0) * 0.25
        score += min(action_coverage, 1.0) * 0.10
        score += min(max(retriever_score, 0.0), 1.0) * 0.05
        return min(score, 1.0)

    def _cross_product_terms(self, product: str) -> tuple[str, ...]:
        groups = {
            "snowmobile": ("spark plug", "electrode", "engine oil"),
            "motherboard": ("snowmobile", "boat", "grill", "camera"),
            "空调": ("洗碗机", "烤箱", "自动运行"),
            "发电机": ("保修", "warranty", "emission"),
            "电钻": ("运输电池", "火灾隐患"),
        }
        return groups.get(product, ())

    def _flag(self, name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}
