from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.services.knowledge_base import build_retrieval_corpus


def main() -> None:
    settings = Settings()
    metadata = build_retrieval_corpus(settings)
    print(f"chunk_count={metadata['chunk_count']}")
    print(f"retrieval_record_count={metadata['retrieval_record_count']}")
    print(f"output: {settings.retrieval_corpus_path}")


if __name__ == "__main__":
    main()
