from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    images: list[str] = Field(default_factory=list, description="Base64 data URL images")
    session_id: str | None = Field(default=None, description="Short-term conversation id")
    stream: bool = Field(default=False, description="Reserved for future streaming support")

