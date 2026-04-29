from __future__ import annotations

from pydantic import BaseModel, Field


class ChatData(BaseModel):
    answer: str
    session_id: str
    timestamp: int
    references: list[dict] = Field(default_factory=list)
    related_images: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "success"
    data: ChatData

