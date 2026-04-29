from __future__ import annotations

# Compatibility module for the AgentGraph layout. The active FastAPI route is
# still app.api.chat; this module gives the new architecture a stable import path.
from app.api.chat import router

__all__ = ["router"]
