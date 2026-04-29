from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings
from app.services.embedding_client import EmbeddingClient
from app.services.knowledge_base import KnowledgeBaseRepository


def main() -> None:
    settings = Settings()
    repository = KnowledgeBaseRepository(settings)
    corpus = repository.get_retrieval_corpus()
    client = EmbeddingClient(settings)
    if not client.is_enabled():
        raise SystemExit(
            'dense embedding client is not enabled; set DENSE_ENABLED=1, DENSE_BASE_URL and DENSE_MODEL before building the index'
        )

    texts = [record.retrieval_text_mix or record.retrieval_text_en or record.title_zh for record in corpus]
    vectors = client.embed(texts)
    if vectors is None:
        raise SystemExit('failed to build dense index: embedding request returned no vectors')
    if len(vectors) != len(corpus):
        raise SystemExit(f'failed to build dense index: expected {len(corpus)} vectors, got {len(vectors)}')

    settings.ensure_data_dirs()
    with settings.dense_index_path.open('w', encoding='utf-8') as handle:
        for record, vector in zip(corpus, vectors):
            payload = {
                'chunk_id': record.chunk_id,
                'vector': vector,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')

    print(f'corpus_records={len(corpus)}')
    print(f'output: {settings.dense_index_path}')


if __name__ == '__main__':
    main()
