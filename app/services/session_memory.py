from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class SessionTurn:
    timestamp: int
    question: str
    answer: str
    manuals: list[str]
    product_name: str | None
    section_titles: list[str]


@dataclass(slots=True)
class SessionContext:
    session_id: str
    last_product: str | None = None
    last_manuals: list[str] = field(default_factory=list)
    last_sections: list[str] = field(default_factory=list)
    turns: deque[SessionTurn] = field(default_factory=deque)


class SessionMemory:
    def __init__(self, ttl_seconds: int = 1800, max_turns: int = 6) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._sessions: dict[str, SessionContext] = {}

    def _purge_expired(self) -> None:
        now = time.time()
        expired = []
        for session_id, context in self._sessions.items():
            if not context.turns:
                expired.append(session_id)
                continue
            latest = context.turns[-1].timestamp
            if now - latest > self.ttl_seconds:
                expired.append(session_id)
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def get(self, session_id: str) -> SessionContext:
        self._purge_expired()
        return self._sessions.setdefault(session_id, SessionContext(session_id=session_id))

    def append_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        manuals: list[str],
        product_name: str | None,
        section_titles: list[str],
    ) -> SessionContext:
        context = self.get(session_id)
        context.last_manuals = manuals[:3]
        context.last_product = product_name
        context.last_sections = section_titles[:4]
        context.turns.append(
            SessionTurn(
                timestamp=int(time.time()),
                question=question,
                answer=answer,
                manuals=manuals[:3],
                product_name=product_name,
                section_titles=section_titles[:4],
            )
        )
        while len(context.turns) > self.max_turns:
            context.turns.popleft()
        return context

