from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.services.llm_client import LLMClient, LLMMessage


LOGGER = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = (
    "You are a RAG retrieval planner. Do not answer the question. "
    "Return compact strict JSON only, no markdown. "
    "Keep product/model/button/feature terms unchanged. "
    "Use exclude_terms for wrong products/topics."
)

PLANNER_SCHEMA_HINT = {
    "product": "normalized product name or null",
    "manual_scope": ["exact manual names likely relevant"],
    "intent": "short snake_case intent",
    "answer_language": "zh|en",
    "action_terms": ["verbs/actions from the question"],
    "object_terms": ["objects/features/components from the question"],
    "must_terms": ["terms that should appear in good evidence"],
    "should_terms": ["helpful optional evidence terms"],
    "exclude_terms": ["terms indicating wrong product/topic evidence"],
    "query_variants": ["2-5 retrieval queries"],
    "confidence": 0.0,
}


@dataclass(slots=True)
class LLMQueryPlan:
    product: str | None = None
    manual_scope: list[str] = field(default_factory=list)
    intent: str = "unknown"
    answer_language: str = "zh"
    action_terms: list[str] = field(default_factory=list)
    object_terms: list[str] = field(default_factory=list)
    must_terms: list[str] = field(default_factory=list)
    should_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    query_variants: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def as_variants(self) -> dict[str, str]:
        variants: dict[str, str] = {}
        for index, query in enumerate(self.query_variants[:5], start=1):
            if query:
                variants[f"planner_{index}"] = query
        return variants


class LLMQueryPlanner:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client

    def plan(
        self,
        *,
        question: str,
        product: str | None,
        language: str,
        candidates: list[Any],
    ) -> LLMQueryPlan | None:
        if self.llm_client is None or not self.llm_client.is_enabled():
            return None
        messages = self._build_messages(question, product, language, candidates, candidate_limit=5, text_limit=140)
        raw = self.llm_client.chat(messages, temperature=0.0, max_tokens=min(self.llm_client.settings.llm_max_tokens, 384))
        if not raw:
            messages = self._build_messages(question, product, language, candidates, candidate_limit=3, text_limit=80)
            raw = self.llm_client.chat(messages, temperature=0.0, max_tokens=min(self.llm_client.settings.llm_max_tokens, 256))
        if not raw:
            return None
        payload = self._parse_json(raw)
        if payload is None:
            LOGGER.warning("LLM query planner returned non-JSON payload: %s", raw[:300])
            return None
        return self._coerce_plan(payload, question, language)

    def _build_messages(
        self,
        question: str,
        product: str | None,
        language: str,
        candidates: list[Any],
        candidate_limit: int,
        text_limit: int,
    ) -> list[LLMMessage]:
        candidate_payload = [self._candidate_summary(candidate, text_limit=text_limit) for candidate in candidates[:candidate_limit]]
        return [
            LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    "Return JSON keys exactly: product, manual_scope, intent, answer_language, "
                    "action_terms, object_terms, must_terms, should_terms, exclude_terms, query_variants, confidence.\n"
                    + json.dumps(
                        {
                            "question": question,
                            "rule_product": product,
                            "language": language,
                            "candidates": candidate_payload,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                ),
            ),
        ]

    def _candidate_summary(self, candidate: Any, *, text_limit: int) -> dict[str, str]:
        chunk = getattr(candidate, "chunk", candidate)
        text = getattr(chunk, "text", "") or ""
        return {
            "chunk_id": str(getattr(chunk, "chunk_id", "")),
            "manual_name": str(getattr(chunk, "manual_name", "")),
            "product_name": str(getattr(chunk, "product_name", "")),
            "section_title": str(getattr(chunk, "section_title", ""))[:90],
            "text": re.sub(r"\s+", " ", text).strip()[:text_limit],
        }

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _coerce_plan(self, payload: dict[str, Any], question: str, language: str) -> LLMQueryPlan:
        return LLMQueryPlan(
            product=self._clean_optional_string(payload.get("product")),
            manual_scope=self._clean_list(payload.get("manual_scope"), limit=5),
            intent=self._clean_optional_string(payload.get("intent")) or "unknown",
            answer_language=self._coerce_language(payload.get("answer_language"), language),
            action_terms=self._clean_list(payload.get("action_terms"), limit=8),
            object_terms=self._clean_list(payload.get("object_terms"), limit=10),
            must_terms=self._clean_list(payload.get("must_terms"), limit=10),
            should_terms=self._clean_list(payload.get("should_terms"), limit=12),
            exclude_terms=self._clean_list(payload.get("exclude_terms"), limit=14),
            query_variants=self._clean_queries(payload.get("query_variants"), question),
            confidence=self._coerce_confidence(payload.get("confidence")),
        )

    def _clean_optional_string(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned or cleaned.lower() in {"none", "null", "unknown", "无"}:
            return None
        return cleaned[:80]

    def _clean_list(self, value: Any, *, limit: int) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                continue
            normalized = re.sub(r"\s+", " ", item).strip()
            if len(normalized) < 2:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(normalized[:80])
            if len(cleaned) >= limit:
                break
        return cleaned

    def _clean_queries(self, value: Any, question: str) -> list[str]:
        queries = self._clean_list(value, limit=5)
        if question and question not in queries:
            queries.insert(0, question)
        return queries[:5]

    def _coerce_language(self, value: Any, fallback: str) -> str:
        if isinstance(value, str) and value.lower().strip() in {"en", "english"}:
            return "en"
        if isinstance(value, str) and value.lower().strip() in {"zh", "chinese", "中文"}:
            return "zh"
        return "en" if fallback == "en" else "zh"

    def _coerce_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))
