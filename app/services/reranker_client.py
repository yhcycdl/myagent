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
class RerankItem:
    index: int
    relevance_score: float


class RerankerClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def is_enabled(self) -> bool:
        return (
            self.settings.rerank_enabled
            and bool(self.settings.rerank_base_url)
            and bool(self.settings.rerank_model)
            and httpx is not None
        )

    def rerank(self, query: str, documents: list[str], *, top_n: int | None = None) -> list[RerankItem] | None:
        if not self.is_enabled():
            return None
        if not query or not documents:
            return []

        endpoint = self._build_endpoint(self.settings.rerank_base_url)
        headers = {"Content-Type": "application/json"}
        if self.settings.rerank_api_key:
            headers["Authorization"] = f"Bearer {self.settings.rerank_api_key}"

        payload = {
            "model": self.settings.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": self.settings.rerank_top_n if top_n is None else top_n,
        }

        try:
            with httpx.Client(timeout=self.settings.rerank_timeout_seconds) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Rerank request failed: %s", exc)
            return None

        try:
            return self._extract_results(data)
        except Exception as exc:  # pragma: no cover - malformed payload
            LOGGER.warning("Rerank response parse failed: %s", exc)
            return None

    def _build_endpoint(self, base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/rerank"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/rerank"
        return f"{normalized}/v1/rerank"

    def _extract_results(self, payload: dict[str, Any]) -> list[RerankItem]:
        data = payload.get("results")
        if not isinstance(data, list):
            raise ValueError("rerank response missing results list")
        results: list[RerankItem] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int):
                raise ValueError("rerank item missing integer index")
            if not isinstance(score, (int, float)):
                raise ValueError("rerank item missing numeric relevance_score")
            results.append(RerankItem(index=index, relevance_score=float(score)))
        return results
