from __future__ import annotations

import time
from typing import Any

from app.core.agent_state import AgentState


class BaseNode:
    name = "base"

    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError

    def log(self, state: AgentState, data: dict[str, Any]) -> None:
        state.trace.append(
            {
                "node": self.name,
                "timestamp": int(time.time()),
                "data": data,
            }
        )

