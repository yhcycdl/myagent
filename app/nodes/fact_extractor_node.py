from __future__ import annotations

import re
from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode


class FactExtractorNode(BaseNode):
    name = "fact_extractor"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        facts: list[str] = []
        steps: list[str] = []
        warnings: list[str] = []
        conditions: list[str] = []

        for result in state.raw_accepted_evidence[:5]:
            snippet = self._snippet(state, result)
            if not snippet:
                continue
            for sentence in self._split_sentences(snippet):
                if self._is_warning(sentence):
                    warnings.append(sentence)
                elif self._is_step(sentence):
                    steps.append(self._clean_step(sentence))
                elif self._is_condition(sentence):
                    conditions.append(sentence)
                else:
                    facts.append(sentence)

        state.facts = self._dedupe(facts, limit=8)
        state.steps = self._dedupe(steps, limit=8)
        state.warnings = self._dedupe(warnings, limit=4)
        state.conditions = self._dedupe(conditions, limit=4)
        self.log(
            state,
            {
                "facts": state.facts,
                "steps": state.steps,
                "warnings": state.warnings,
                "conditions": state.conditions,
            },
        )
        return state

    def _snippet(self, state: AgentState, result: Any) -> str:
        try:
            snippet = self.service.generator._select_best_snippet(state.resolved_question or state.question, [result])
        except Exception:  # noqa: BLE001
            snippet = result.chunk.text
        return self._clean(snippet)

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？.!?])\s+|\n+|(?=\d+[.、]\s*)", text)
        return [self._clean(part) for part in parts if self._clean(part)]

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"(?<=\w)\s+(?=\w\b)", "", text)
        return text.strip(" ;；")

    def _is_step(self, text: str) -> bool:
        return bool(re.match(r"^\d+[.、]\s*", text)) or any(term in text.lower() for term in ("press", "remove", "connect", "install", "turn", "按", "取下", "连接", "安装", "打开"))

    def _is_warning(self, text: str) -> bool:
        return any(term in text.lower() for term in ("warning", "caution", "do not", "注意", "警告", "切勿", "不要"))

    def _is_condition(self, text: str) -> bool:
        return any(term in text.lower() for term in ("if ", "when ", "如果", "若", "当"))

    def _clean_step(self, text: str) -> str:
        return re.sub(r"^\d+[.、]\s*", "", text).strip()

    def _dedupe(self, values: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            key = value.lower()
            if key in seen or len(value) < 3:
                continue
            seen.add(key)
            output.append(value)
            if len(output) >= limit:
                break
        return output
