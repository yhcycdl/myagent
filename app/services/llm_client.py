from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings


LOGGER = logging.getLogger(__name__)

try:
    import httpx
except ImportError:  # pragma: no cover - optional dependency until installed
    httpx = None


@dataclass(slots=True)
class LLMMessage:
    role: str
    content: str


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def is_enabled(self) -> bool:
        return (
            self.settings.llm_enabled
            and bool(self.settings.llm_base_url)
            and bool(self.settings.llm_model)
            and httpx is not None
        )

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        if not self.is_enabled():
            return None

        endpoint = self._build_endpoint(self.settings.llm_base_url)
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": self.settings.llm_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }

        try:
            with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                LOGGER.warning("LLM request failed: %s body=%s", exc, exc.response.text[:500])
                return None
            LOGGER.warning("LLM request failed: %s", exc)
            return None

        answer = self._extract_answer(data)
        if not answer:
            LOGGER.warning("LLM response did not contain usable text")
            return None
        return answer

    def _build_endpoint(self, base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/chat/completions"
        return f"{normalized}/v1/chat/completions"

    def _extract_answer(self, payload: dict[str, Any]) -> str | None:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        first = choices[0]
        if not isinstance(first, dict):
            return None

        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            extracted = self._extract_content_text(content)
            if extracted:
                return extracted

        text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None

    def _extract_content_text(self, content: Any) -> str | None:
        if isinstance(content, str):
            return content.strip() or None
        if not isinstance(content, list):
            return None

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            nested = item.get("content")
            if isinstance(nested, str) and nested.strip():
                parts.append(nested.strip())
        if not parts:
            return None
        return "\n".join(parts)
