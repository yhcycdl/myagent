from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.services.preprocess import looks_english_dominant


class AnswerVerifierNode(BaseNode):
    name = "answer_verifier"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        answer = state.answer or ""
        issues: list[str] = []
        if not answer.strip():
            issues.append("empty_answer")
        if state.exclude_terms and any(term and term.lower() in answer.lower() for term in state.exclude_terms):
            issues.append("answer_contains_exclude_terms")
        if looks_english_dominant(state.question) and self._looks_chinese(answer):
            issues.append("language_mismatch")
        if "<PIC>" in answer and not state.related_images and not state.image_ids:
            issues.append("pic_without_image")
        decision = "pass" if not issues else "fail"
        risk = "low" if not issues else ("medium" if len(issues) <= 2 else "high")
        state.verifier_result = {
            "decision": decision,
            "risk": risk,
            "feedback": ", ".join(issues),
        }
        self.log(state, state.verifier_result)
        return state

    def _looks_chinese(self, text: str) -> bool:
        chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
        ascii_alpha = sum(1 for char in text if char.isascii() and char.isalpha())
        return chinese > ascii_alpha
