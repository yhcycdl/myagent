from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.services.preprocess import is_general_support_question, is_manual_access_question


class FallbackNode(BaseNode):
    name = "fallback"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        question = state.resolved_question or state.question
        if is_general_support_question(question) or is_manual_access_question(question):
            generated = self.service.generator.build_support_fallback(question)
            state.fallback_reason = "support_fallback"
        else:
            generated = self.service.generator.build_manual_fallback(question)
            state.fallback_reason = state.fallback_reason or "no_accepted_evidence"
        generated = self.service.guardrail.review(generated)
        state.answer = generated.answer
        state.references = generated.references
        state.related_images = generated.related_images
        state.used_manuals = generated.used_manuals
        state.used_sections = generated.used_sections
        self.log(state, {"fallback_reason": state.fallback_reason, "answer_preview": state.answer[:200]})
        return state
