from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings
from app.services.llm_client import LLMClient
from app.services.preprocess import looks_like_follow_up
from app.services.session_memory import SessionContext


DATA_URL_RE = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE)


@dataclass(slots=True)
class ImageInsight:
    mime_type: str
    size_bytes: int
    sha1: str


@dataclass(slots=True)
class MultimodalInsight:
    manual_hints: list[str] = field(default_factory=list)
    product_hint: str | None = None
    visual_summary: str = ""
    visible_text: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)
    image_insights: list[ImageInsight] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MultimodalUnderstandingService:
    def __init__(self, llm_client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.llm_client = llm_client
        self.settings = settings or Settings()

    def analyze(
        self,
        question: str,
        images: list[str],
        session_context: SessionContext,
        alias_lookup: dict[str, str],
    ) -> MultimodalInsight:
        manual_hints: list[str] = []
        lowered_question = question.lower()
        for alias, manual_name in alias_lookup.items():
            if alias and alias in lowered_question and manual_name not in manual_hints:
                manual_hints.append(manual_name)

        if not manual_hints and looks_like_follow_up(question):
            manual_hints.extend(session_context.last_manuals)

        product_hint = None
        if manual_hints:
            product_hint = manual_hints[0].removesuffix("手册")
        elif session_context.last_product:
            product_hint = session_context.last_product

        image_insights: list[ImageInsight] = []
        warnings: list[str] = []
        for image in images[:3]:
            try:
                image_insights.append(self._inspect_image(image))
            except ValueError as exc:
                warnings.append(str(exc))

        visual_summary = ""
        visible_text: list[str] = []
        query_terms: list[str] = []
        if images and self.settings.llm_vision_enabled and self.llm_client and self.llm_client.is_enabled() and not warnings:
            payload = self._llm_visual_understanding(question, images)
            if payload:
                product_hint = payload.get("product") or product_hint
                observed_state = str(payload.get("observed_state") or "").strip()
                visual_summary = str(payload.get("visual_summary") or "").strip()
                if observed_state and observed_state not in visual_summary:
                    visual_summary = f"{visual_summary}；状态：{observed_state}".strip("；")
                visible_text = self._coerce_string_list(payload.get("visible_text"), limit=8)
                query_terms = self._coerce_string_list(payload.get("query_terms"), limit=12)
                for alias, manual_name in alias_lookup.items():
                    haystack = " ".join([product_hint or "", visual_summary, " ".join(visible_text), " ".join(query_terms)]).lower()
                    if alias and alias in haystack and manual_name not in manual_hints:
                        manual_hints.append(manual_name)

        return MultimodalInsight(
            manual_hints=manual_hints,
            product_hint=product_hint,
            visual_summary=visual_summary,
            visible_text=visible_text,
            query_terms=query_terms,
            image_insights=image_insights,
            warnings=warnings,
        )

    def _inspect_image(self, payload: str) -> ImageInsight:
        match = DATA_URL_RE.match(payload.strip())
        if not match:
            raise ValueError("Unsupported image payload. Expected data URL with base64 content.")

        mime_type = match.group("mime")
        image_bytes = base64.b64decode(match.group("data"), validate=True)
        return ImageInsight(
            mime_type=mime_type,
            size_bytes=len(image_bytes),
            sha1=hashlib.sha1(image_bytes).hexdigest(),
        )

    def _llm_visual_understanding(self, question: str, images: list[str]) -> dict[str, Any] | None:
        prompt = (
            "你是多模态客服 RAG 系统的图片理解器。不要回答用户问题，只把图片转成检索线索。\n"
            "请严格输出 JSON，不要输出 markdown。\n"
            "字段：product, visual_summary, visible_text, observed_state, query_terms。\n"
            "要求：product 用简短产品名；visible_text 是图片里可见文字/OCR；"
            "observed_state 是灯光、按钮、报错、损坏、安装状态等；query_terms 给适合检索说明书的中英文关键词。\n"
            f"用户问题：{question}"
        )
        raw = self.llm_client.chat_with_images(
            prompt=prompt,
            images=images,
            system_prompt="Only output valid JSON. Do not answer the customer question.",
            temperature=0.0,
            max_tokens=min(self.llm_client.settings.llm_max_tokens, 768),
        )
        if not raw:
            return None
        try:
            return json.loads(self._strip_json(raw))
        except json.JSONDecodeError:
            return None

    def _strip_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text

    def _coerce_string_list(self, value: Any, *, limit: int) -> list[str]:
        if isinstance(value, str):
            value = re.split(r"[,，;；、\n]+", value)
        if not isinstance(value, list):
            return []
        output: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(text)
            if len(output) >= limit:
                break
        return output
