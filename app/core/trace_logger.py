from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from app.core.agent_state import AgentState


class TraceLogger:
    def __init__(self, log_dir: str | Path | None = None) -> None:
        default_dir = Path("data") / "traces"
        self.log_dir = Path(log_dir or os.getenv("TRACE_LOG_DIR", default_dir))
        self.enabled = os.getenv("TRACE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}

    def write(self, state: AgentState) -> None:
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        session = state.session_id or "no_session"
        path = self.log_dir / f"{session}_{int(time.time() * 1000)}.json"
        payload: dict[str, Any] = {
            "question": state.question,
            "resolved_question": state.resolved_question,
            "session_id": state.session_id,
            "product": state.product,
            "intent": state.intent,
            "manual_scope": state.manual_scope,
            "evidence_confidence": state.evidence_confidence,
            "image_ids": state.image_ids,
            "answer": state.answer,
            "verifier_result": state.verifier_result,
            "fallback_reason": state.fallback_reason,
            "trace": state.trace,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

