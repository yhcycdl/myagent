from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field

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
    image_insights: list[ImageInsight] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MultimodalUnderstandingService:
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

        return MultimodalInsight(
            manual_hints=manual_hints,
            product_hint=product_hint,
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

