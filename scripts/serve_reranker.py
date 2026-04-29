from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible reranker server.")
    parser.add_argument("--model", required=True, help="Local reranker model path")
    parser.add_argument("--served-model-name", default="reranker-model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]
    top_n: int | None = None


@dataclass(slots=True)
class RerankEngine:
    model_path: str
    served_model_name: str
    device: str
    max_length: int
    batch_size: int
    trust_remote_code: bool
    tokenizer: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        self.model.to(self.device)

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict[str, float | int]]:
        scores: list[tuple[int, float]] = []
        for start in range(0, len(documents), self.batch_size):
            batch_docs = documents[start : start + self.batch_size]
            pairs = [[query, document] for document in batch_docs]
            encoded = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)
                logits = outputs.logits
                if logits.ndim == 2 and logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                elif logits.ndim == 2:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits
            for offset, score in enumerate(batch_scores.detach().cpu().tolist()):
                scores.append((start + offset, float(score)))

        scores.sort(key=lambda item: item[1], reverse=True)
        limit = len(scores) if top_n is None else max(0, min(top_n, len(scores)))
        return [
            {
                "index": index,
                "relevance_score": score,
            }
            for index, score in scores[:limit]
        ]


def build_app(engine: RerankEngine) -> FastAPI:
    app = FastAPI(title="Local Reranker Server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/rerank")
    def rerank(request: RerankRequest) -> dict[str, Any]:
        if request.model != engine.served_model_name:
            raise HTTPException(status_code=400, detail=f"unknown model: {request.model}")
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        if not request.documents:
            raise HTTPException(status_code=400, detail="documents must not be empty")
        results = engine.rerank(request.query, request.documents, request.top_n)
        return {
            "model": engine.served_model_name,
            "results": results,
        }

    return app


def main() -> None:
    args = parse_args()
    engine = RerankEngine(
        model_path=args.model,
        served_model_name=args.served_model_name,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        trust_remote_code=args.trust_remote_code,
    )
    app = build_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
