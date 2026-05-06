from __future__ import annotations

import os
import time
from typing import Any
from uuid import uuid4

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode
from app.core.trace_logger import TraceLogger


class FinalResponseNode(BaseNode):
    name = "final_response"

    def __init__(self, service: Any) -> None:
        self.service = service
        self.trace_logger = TraceLogger(os.getenv("TRACE_LOG_DIR") or service.settings.base_dir / "data" / "traces")

    def run(self, state: AgentState) -> AgentState:
        state.session_id = state.session_id or f"kf_session_{uuid4().hex[:12]}"
        timestamp = int(time.time())
        if state.answer:
            self.service.memory.append_turn(
                session_id=state.session_id,
                question=state.resolved_question or state.question,
                answer=state.answer,
                manuals=state.used_manuals,
                product_name=state.product,
                section_titles=state.used_sections,
            )
        state.final_response = {
            "answer": state.answer,
            "session_id": state.session_id,
            "timestamp": timestamp,
            "references": state.references,
            "related_images": state.related_images,
        }
        self.log(state, {"timestamp": timestamp})
        self.trace_logger.write(state)
        return state
