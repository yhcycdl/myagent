from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings

LOGGER = logging.getLogger(__name__)

try:
    import httpx
except ImportError:  # pragma: no cover - optional dependency until installed
    httpx = None


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def is_enabled(self) -> bool:
        return (
            self.settings.dense_enabled
            and bool(self.settings.dense_base_url)
            and bool(self.settings.dense_model)
            and httpx is not None
        )

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.is_enabled():
            return None
        if not texts:
            return []

        endpoint = self._build_endpoint(self.settings.dense_base_url)
        headers = {"Content-Type": "application/json"}
        if self.settings.dense_api_key:
            headers["Authorization"] = f"Bearer {self.settings.dense_api_key}"

        vectors: list[list[float]] = []
        batch_size = max(1, self.settings.dense_batch_size)
        try:
            with httpx.Client(timeout=self.settings.dense_timeout_seconds) as client:
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    payload = {
                        "model": self.settings.dense_model,
                        "input": batch,
                        "encoding_format": "float",
                    }
                    response = client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    vectors.extend(self._extract_vectors(data, len(batch)))
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Embedding request failed: %s", exc)
            return None
        return vectors

    def _build_endpoint(self, base_url: str) -> str:
        normalized = base_url.rstrip('/')
        if normalized.endswith('/embeddings'):
            return normalized
        if normalized.endswith('/v1'):
            return f"{normalized}/embeddings"
        return f"{normalized}/v1/embeddings"

    def _extract_vectors(self, payload: dict[str, Any], expected: int) -> list[list[float]]:
        data = payload.get('data')
        if not isinstance(data, list):
            raise ValueError('embedding response missing data list')
        ordered = sorted(
            (item for item in data if isinstance(item, dict)),
            key=lambda item: int(item.get('index', 0)),
        )
        vectors: list[list[float]] = []
        for item in ordered:
            embedding = item.get('embedding')
            if not isinstance(embedding, list) or not embedding:
                raise ValueError('embedding response item missing embedding vector')
            vectors.append([float(value) for value in embedding])
        if len(vectors) != expected:
            raise ValueError(f'embedding response size mismatch: expected {expected}, got {len(vectors)}')
        return vectors
