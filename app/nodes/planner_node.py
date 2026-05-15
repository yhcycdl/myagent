from __future__ import annotations

import os
from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.nodes.utils import normalize_terms, split_search_terms


class PlannerNode(BaseNode):
    name = "planner"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        question = state.resolved_question or state.question
        plan = self.service.generator._build_evidence_plan(question)
        state.intent = plan.intent
        state.question_type = self._question_type(question)
        state.action_terms = normalize_terms(list(plan.secondary_terms), limit=8)
        state.object_terms = normalize_terms(list(plan.primary_terms), limit=10)
        state.must_terms = normalize_terms([*plan.primary_terms, *plan.secondary_terms][:8], limit=8)
        state.exclude_terms = normalize_terms([*plan.background_title_terms, *plan.background_body_terms], limit=12)
        state.answer_expectation = self._answer_expectation(state.intent, state.question_type)

        product = state.image_product_hint or self.service._infer_product_hint_from_text([question])
        state.product = product

        query_plan = None
        if self._llm_planner_enabled() and plan.intent not in self.service.generator._preplanned_direct_intents():
            try:
                query_plan = self.service.llm_query_planner.plan(
                    question=question,
                    product=product,
                    language=state.language,
                    candidates=[],
                )
            except Exception as exc:  # noqa: BLE001 - planner must never break retrieval
                self.log(state, {"llm_planner_error": str(exc)})
                query_plan = None

        if query_plan and query_plan.confidence >= 0.55:
            state.product = query_plan.product or state.product
            state.intent = query_plan.intent or state.intent
            state.action_terms = normalize_terms([*state.action_terms, *query_plan.action_terms], limit=10)
            state.object_terms = normalize_terms([*state.object_terms, *query_plan.object_terms], limit=12)
            state.must_terms = normalize_terms([*state.must_terms, *query_plan.must_terms], limit=12)
            state.should_terms = normalize_terms([*state.should_terms, *query_plan.should_terms], limit=12)
            state.exclude_terms = normalize_terms([*state.exclude_terms, *query_plan.exclude_terms], limit=16)
            state.query_variants = normalize_terms(query_plan.query_variants, limit=6)
            state.manual_scope = normalize_terms(query_plan.manual_scope, limit=4)

        if not state.query_variants:
            state.query_variants = self._default_query_variants(question, state)
        if state.image_query_terms:
            state.object_terms = normalize_terms([*state.object_terms, *state.image_query_terms], limit=14)
            state.must_terms = normalize_terms([*state.must_terms, *state.image_query_terms[:6]], limit=12)
            visual_query = " ".join([question, *state.image_query_terms[:10]])
            state.query_variants = normalize_terms([visual_query, *state.query_variants], limit=6)

        self.log(
            state,
            {
                "product": state.product,
                "intent": state.intent,
                "question_type": state.question_type,
                "manual_scope": state.manual_scope,
                "must_terms": state.must_terms,
                "exclude_terms": state.exclude_terms,
                "query_variants": state.query_variants,
                "llm_plan_used": bool(query_plan and query_plan.confidence >= 0.55),
            },
        )
        return state

    def _llm_planner_enabled(self) -> bool:
        return os.getenv("LLM_PLANNER_ENABLED", os.getenv("AGENT_LLM_QUERY_PLANNER_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _question_type(self, question: str) -> str:
        lowered = question.lower()
        if any(term in lowered for term in ("how", "step", "步骤", "如何", "怎么", "安装", "清洁", "设置", "use", "replace")):
            return "manual_instruction"
        if any(term in lowered for term in ("warning", "error", "故障", "报错", "警告", "无法")):
            return "troubleshooting"
        if any(term in lowered for term in ("what", "which", "哪些", "什么", "含义", "meaning")):
            return "manual_qa"
        return "general"

    def _answer_expectation(self, intent: str | None, question_type: str | None) -> str:
        if question_type == "manual_instruction":
            return "answer with concise actionable steps and warnings from evidence"
        if question_type == "troubleshooting":
            return "answer with causes, checks, and safe next steps from evidence"
        return f"answer the user question using accepted evidence for intent={intent or 'unknown'}"

    def _default_query_variants(self, question: str, state: AgentState) -> list[str]:
        terms = " ".join([*state.must_terms[:4], *state.should_terms[:4]])
        variants = [question]
        if terms:
            variants.append(f"{question} {terms}")
        object_terms = " ".join(split_search_terms(" ".join(state.object_terms))[:6])
        if object_terms and object_terms not in variants:
            variants.append(object_terms)
        return normalize_terms(variants, limit=4)
