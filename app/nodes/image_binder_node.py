from __future__ import annotations

from typing import Any

from app.core.agent_state import AgentState
from app.core.node_base import BaseNode


class ImageBinderNode(BaseNode):
    name = "image_binder"

    def __init__(self, service: Any) -> None:
        self.service = service

    def run(self, state: AgentState) -> AgentState:
        image_index = self.service.repository.image_index()
        image_ids: list[str] = []
        if state.intent:
            image_ids.extend(self.service.generator._planned_image_ids(state.intent))
        for result in state.raw_accepted_evidence:
            for attr_name in ("image_ids", "nearby_image_ids", "section_image_ids"):
                for image_id in getattr(result.chunk, attr_name, []) or []:
                    image_ids.append(image_id)

        filtered: list[str] = []
        for image_id in image_ids:
            image = image_index.get(image_id)
            if image is None or image_id in filtered:
                continue
            if state.manual_scope and image.manual_name not in state.manual_scope:
                continue
            filtered.append(image_id)
            if len(filtered) >= 3:
                break
        state.image_ids = filtered
        self.log(state, {"image_ids": state.image_ids})
        return state
