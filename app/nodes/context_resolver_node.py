from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.services.preprocess import looks_english_dominant, normalize_question


class ContextResolverNode(BaseNode):
    name = "context_resolver"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        question = normalize_question(state.question)
        context = self.service.memory.get(state.session_id) if state.session_id else None
        resolved = question
        last_product = getattr(context, "last_product", None)
        if last_product and self._looks_followup(question):
            resolved = f"{last_product} {question}"
        state.resolved_question = resolved
        state.language = "en" if looks_english_dominant(resolved) else "zh"
        if context and context.turns:
            latest = context.turns[-1]
            state.history_summary = f"last_product={context.last_product}; last_sections={context.last_sections}; last_answer={latest.answer[:160]}"
        self.log(
            state,
            {
                "resolved_question": state.resolved_question,
                "language": state.language,
                "last_product": last_product,
                "history_used": bool(context and context.turns),
            },
        )
        return state

    def _looks_followup(self, question: str) -> bool:
        lowered = question.lower()
        markers = ("它", "这个", "那个", "刚才", "上面", "继续", "怎么关", "什么时候", "what about", "how about", "it ", "that ")
        return any(marker in lowered for marker in markers)

