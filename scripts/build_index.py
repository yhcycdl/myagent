from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.services.knowledge_base import build_knowledge_base


def main() -> None:
    settings = Settings()
    metadata = build_knowledge_base(settings)
    print(f"manual_count={metadata['manual_count']}")
    print(f"chunk_count={metadata['chunk_count']}")
    print(f"image_count={metadata['image_count']}")
    warnings = metadata.get("warnings", [])
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()

