from __future__ import annotations

from fastapi import FastAPI

from app.api.chat_api import router as chat_router
from app.core.config import Settings
from app.services.knowledge_base import KnowledgeBaseRepository


settings = Settings()
app = FastAPI(
    title="Multimodal Customer Agent",
    version="0.1.0",
    description="Evidence-grounded customer service agent for manual QA and conservative support fallback.",
)
app.include_router(chat_router)


@app.on_event("startup")
def warmup_knowledge_base() -> None:
    repository = KnowledgeBaseRepository(settings)
    repository.ensure_ready()


@app.get("/health")
def health() -> dict:
    repository = KnowledgeBaseRepository(settings)
    repository.ensure_ready()
    return {
        "status": "ok",
        "chunk_count": len(repository.get_chunks()),
        "image_count": len(repository.get_images()),
    }
