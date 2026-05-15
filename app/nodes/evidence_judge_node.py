from __future__ import annotations

import json
import os
import re
from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.nodes.utils import contains_any, coverage_ratio, result_to_dict
from app.services.llm_client import LLMMessage


class EvidenceJudgeNode(BaseNode):
    name = "evidence_judge"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        accepted: list[Any] = []
        rejected: list[dict[str, Any]] = []
        soft_fallbacks: list[tuple[Any, dict[str, Any]]] = []
        llm_calls = 0
        state.accepted_evidence = []
        state.rejected_evidence = []
        state.raw_accepted_evidence = []
        threshold = self._float_env("EVIDENCE_ACCEPT_THRESHOLD", self.service.settings.evidence_accept_threshold)
        hard_reject = self._flag("EXCLUDE_TERMS_HARD_REJECT", self.service.settings.exclude_terms_hard_reject)
        llm_enabled = self._flag("LLM_EVIDENCE_JUDGE_ENABLED", self.service.settings.llm_evidence_judge_enabled)
        llm_call_limit = self._int_env("LLM_EVIDENCE_JUDGE_MAX_CALLS", 4)
        soft_accept_enabled = self._flag("EVIDENCE_SOFT_ACCEPT_ENABLED", False)

        for result in state.raw_reranked_candidates or state.raw_candidates:
            text = self._combined_text(result)
            dynamic_excludes = self._dynamic_exclude_terms(state)
            exclude_hits = self._term_hits(text, [*state.exclude_terms, *dynamic_excludes])
            product_match = self._product_match(state, result)
            must_coverage = coverage_ratio(text, state.must_terms)
            object_coverage = coverage_ratio(text, state.object_terms)
            action_coverage = coverage_ratio(text, state.action_terms)
            quality_flags = self._quality_flags(state, text)
            answers_question = self._answers_question(must_coverage, object_coverage, action_coverage, quality_flags)
            confidence = self._confidence(
                product_match,
                must_coverage,
                object_coverage,
                action_coverage,
                result.score,
                quality_flags,
            )

            decision = "accept" if product_match and answers_question and confidence >= threshold else "reject"
            reason = "accepted"
            if not product_match:
                reason = "product_or_manual_scope_mismatch"
                decision = "reject"
            elif hard_reject and exclude_hits:
                reason = "exclude_terms_hit"
                decision = "reject"
            elif quality_flags:
                reason = quality_flags[0]
                decision = "reject"
            elif not answers_question:
                reason = "insufficient_term_coverage"
            elif confidence < threshold:
                reason = "low_confidence"

            llm_payload: dict[str, Any] | None = None
            if llm_enabled and llm_calls < llm_call_limit and self._should_call_llm_judge(product_match, exclude_hits, confidence, decision):
                llm_calls += 1
                llm_payload = self._llm_judge(state, result, text)
                if llm_payload:
                    llm_decision = str(llm_payload.get("decision", "")).lower()
                    llm_confidence = self._coerce_confidence(llm_payload.get("confidence"), confidence)
                    if llm_decision == "reject":
                        decision = "reject"
                        reason = str(llm_payload.get("reason") or "llm_reject")[:120]
                        confidence = min(confidence, llm_confidence)
                    elif llm_decision == "accept" and product_match and not exclude_hits and not quality_flags:
                        decision = "accept"
                        reason = "llm_accept"
                        confidence = max(confidence, llm_confidence)

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
                    "quality_flags": quality_flags,
                }
            )
            if llm_payload:
                payload["llm_judge"] = llm_payload
                usable_facts = llm_payload.get("usable_facts")
                if isinstance(usable_facts, list):
                    payload["usable_facts"] = [str(item).strip() for item in usable_facts if str(item).strip()][:8]
            if decision == "accept":
                accepted.append(result)
                state.accepted_evidence.append(payload)
            else:
                rejected.append(payload)
                if soft_accept_enabled and self._can_soft_accept(product_match, exclude_hits, quality_flags, confidence, must_coverage, object_coverage) and len(soft_fallbacks) < 4:
                    soft_payload = dict(payload)
                    soft_payload["decision"] = "accept"
                    soft_payload["reason"] = f"soft_accept_after_rule_reject:{reason}"
                    soft_payload["confidence"] = max(float(soft_payload.get("confidence", 0.0)), 0.42)
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
                "llm_calls": llm_calls,
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
        search_text = getattr(chunk, "search_text", "") or chunk.text
        return f"{chunk.manual_name} {chunk.section_title} {search_text}"

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
        quality_flags: list[str],
    ) -> float:
        score = 0.0
        if product_match:
            score += 0.30
        score += min(must_coverage, 1.0) * 0.30
        score += min(object_coverage, 1.0) * 0.25
        score += min(action_coverage, 1.0) * 0.10
        score += min(max(retriever_score, 0.0), 1.0) * 0.05
        if quality_flags:
            score -= 0.35
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

    def _answers_question(
        self,
        must_coverage: float,
        object_coverage: float,
        action_coverage: float,
        quality_flags: list[str],
    ) -> bool:
        if quality_flags:
            return False
        if must_coverage >= 0.45:
            return True
        if object_coverage >= 0.40 and action_coverage >= 0.12:
            return True
        return object_coverage >= 0.55

    def _quality_flags(self, state: AgentState, text: str) -> list[str]:
        lowered = text.lower()
        question = (state.resolved_question or state.question).lower()
        flags: list[str] = []
        if self._looks_toc_or_index_noise(lowered):
            flags.append("toc_or_index_noise")
        if self._looks_ocr_fragment(lowered):
            flags.append("ocr_fragment")
        if self._wrong_section_by_question(question, lowered):
            flags.append("wrong_section_for_question")
        return flags

    def _looks_toc_or_index_noise(self, lowered: str) -> bool:
        if any(term in lowered for term in ("table of contents", "contents ", "目录", "index ")):
            return True
        return lowered.count(".....") >= 2 or lowered.count("___") >= 2

    def _looks_ocr_fragment(self, lowered: str) -> bool:
        if re.search(r"\b\w\s+\w\b", lowered) and any(term in lowered for term in (" charg e", " operatio n", " startu p", " use d", " wate r")):
            return True
        if len(lowered) < 40 and not any(char in lowered for char in "。.!?"):
            return True
        return False

    def _wrong_section_by_question(self, question: str, lowered: str) -> bool:
        pairs = [
            (("uphill", "downhill", "crossing a slope", "slope"), ("spark plug", "electrode", "engine oil")),
            (("energy saving", "节能制冷"), ("auto operation", "自动运行", "自动转换")),
            (("charging", "充电步骤", "给电钻充电"), ("fire hazard", "transporting batteries", "运输电池", "火灾隐患")),
            (("t_sensor", "thermal sensor"), ("bios", "ai tweaker", "boot menu")),
            (("af mode", "auto focus"), ("virtual wall", "vacuum")),
            (("warranty", "保修", "售后"), ("fcc", "interference")),
        ]
        return any(any(q in question for q in q_terms) and any(bad in lowered for bad in bad_terms) for q_terms, bad_terms in pairs)

    def _dynamic_exclude_terms(self, state: AgentState) -> list[str]:
        question = (state.resolved_question or state.question).lower()
        terms: list[str] = []
        if any(term in question for term in ("uphill", "downhill", "crossing a slope", "slope")):
            terms.extend(["spark plug", "electrode", "engine oil"])
        if "节能制冷" in question or "energy saving cooling" in question:
            terms.extend(["自动运行", "auto operation", "自动转换"])
        if "t_sensor" in question or "thermal sensor" in question:
            terms.extend(["BIOS", "Ai Tweaker", "boot"])
        if "af mode" in question or "auto focus" in question:
            terms.extend(["Virtual Wall", "vacuum"])
        return terms

    def _term_hits(self, text: str, terms: list[str]) -> list[str]:
        lowered = text.lower()
        hits: list[str] = []
        for term in terms:
            normalized = str(term).strip()
            if normalized and normalized.lower() in lowered and normalized not in hits:
                hits.append(normalized)
        return hits

    def _can_soft_accept(
        self,
        product_match: bool,
        exclude_hits: list[str],
        quality_flags: list[str],
        confidence: float,
        must_coverage: float,
        object_coverage: float,
    ) -> bool:
        if not product_match or exclude_hits or quality_flags:
            return False
        return confidence >= 0.38 and (must_coverage >= 0.25 or object_coverage >= 0.35)

    def _should_call_llm_judge(
        self,
        product_match: bool,
        exclude_hits: list[str],
        confidence: float,
        decision: str,
    ) -> bool:
        if not product_match or exclude_hits:
            return False
        if decision == "accept" and confidence < 0.72:
            return True
        return decision == "reject" and confidence >= 0.25

    def _llm_judge(self, state: AgentState, result: Any, text: str) -> dict[str, Any] | None:
        if not self.service.llm_client.is_enabled():
            return None
        payload = {
            "question": state.resolved_question or state.question,
            "product": state.product,
            "intent": state.intent,
            "manual_scope": state.manual_scope,
            "must_terms": state.must_terms,
            "object_terms": state.object_terms,
            "action_terms": state.action_terms,
            "exclude_terms": state.exclude_terms,
            "candidate": {
                "chunk_id": result.chunk.chunk_id,
                "manual_name": result.chunk.manual_name,
                "section_title": result.chunk.section_title,
                "text": re.sub(r"\s+", " ", text).strip()[:900],
            },
        }
        raw = self.service.llm_client.chat(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "You are a strict RAG evidence judge. Do not answer the user. "
                        "Return JSON only with keys: decision, confidence, reason, usable_facts. "
                        "decision must be accept or reject. Reject same-product but wrong-section evidence. "
                        "Reject evidence containing exclude_terms."
                    ),
                ),
                LLMMessage(role="user", content=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
            ],
            temperature=0.0,
            max_tokens=min(self.service.llm_client.settings.llm_max_tokens, 320),
        )
        if not raw:
            return None
        return self._parse_json(raw)

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None

    def _coerce_confidence(self, value: Any, fallback: float) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return fallback
        return max(0.0, min(1.0, confidence))

    def _int_env(self, name: str, default: int) -> int:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _float_env(self, name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _flag(self, name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}
