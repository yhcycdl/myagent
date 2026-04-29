from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.services.generator import AnswerContext


class AnswerGeneratorNode(BaseNode):
    name = "answer_generator"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        question = state.resolved_question or state.question
        evidence = state.raw_accepted_evidence or []
        if not evidence:
            direct_images = self._direct_images(state)
            direct_answer = self._direct_answer(question, direct_images)
            if direct_answer:
                state.answer = direct_answer
                state.references = []
                state.related_images = direct_images
                state.used_manuals = []
                state.used_sections = []
                self.log(
                    state,
                    {
                        "answer_preview": state.answer[:240],
                        "references": 0,
                        "related_images": [image.get("image_id") for image in state.related_images],
                        "mode": "direct_intent",
                    },
                )
                return state
        generated = self.service.generator.generate(
            question,
            [(question, evidence)],
            self.service.repository.image_index(),
            product_hint=state.product,
        )
        generated = self.service.guardrail.review(generated)
        if "当前检索到的说明书证据" in generated.answer:
            direct_images = self._direct_images(state)
            direct_answer = self._direct_answer(question, direct_images)
            if direct_answer:
                state.answer = direct_answer
                state.references = []
                state.related_images = direct_images
                state.used_manuals = []
                state.used_sections = []
                self.log(
                    state,
                    {
                        "answer_preview": state.answer[:240],
                        "references": 0,
                        "related_images": [image.get("image_id") for image in state.related_images],
                        "mode": "direct_intent_after_insufficient",
                    },
                )
                return state
        state.answer = generated.answer
        state.references = generated.references
        state.related_images = generated.related_images
        state.used_manuals = generated.used_manuals
        state.used_sections = generated.used_sections
        self.log(
            state,
            {
                "answer_preview": state.answer[:240],
                "references": len(state.references),
                "related_images": [image.get("image_id") for image in state.related_images],
            },
        )
        return state

    def _direct_answer(self, question: str, related_images: list[dict]) -> str | None:
        try:
            plan = self.service.generator._build_evidence_plan(question)
            answer = self.service.generator._select_planned_direct_answer(plan, [], question)
            if not answer:
                return None
            context = AnswerContext(
                references=[],
                related_images=related_images,
                used_manuals=[],
                used_sections=[],
            )
            return self.service.generator._finalize_answer_text(answer, context)
        except Exception:  # noqa: BLE001
            return None

    def _direct_images(self, state: AgentState) -> list[dict]:
        image_index = self.service.repository.image_index()
        image_ids: list[str] = []
        if state.intent:
            image_ids.extend(self.service.generator._planned_image_ids(state.intent))
        if not image_ids:
            image_ids.extend(image_id for image_id in state.image_ids if image_id not in image_ids)
        related: list[dict] = []
        for image_id in image_ids[:3]:
            image = image_index.get(image_id)
            if image is None:
                continue
            related.append(
                {
                    "image_id": image.image_id,
                    "manual_name": image.manual_name,
                    "caption": image.caption,
                    "image_path": image.image_path,
                }
            )
        return related
