from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings
from app.schemas.request import ChatRequest
from app.schemas.response import ChatData, ChatResponse
from app.services.chat_service import ChatService


router = APIRouter()
security = HTTPBearer(auto_error=False)
settings = Settings()
chat_service = ChatService(settings)


def _check_token(credentials: HTTPAuthorizationCredentials | None) -> None:
    if not settings.api_token:
        return
    if credentials is None or credentials.credentials != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
        )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> ChatResponse:
    _check_token(credentials)

    try:
        reply = chat_service.chat(
            question=request.question,
            images=request.images,
            session_id=request.session_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        data=ChatData(
            answer=reply.answer,
            session_id=reply.session_id,
            timestamp=reply.timestamp,
            references=reply.references,
            related_images=reply.related_images,
        )
    )

__all__ = ["router"]
