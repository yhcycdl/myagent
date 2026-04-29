from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible embedding server.")
    parser.add_argument("--model", required=True, help="Local model path, e.g. ~/models/bge-m3")
    parser.add_argument("--served-model-name", default="embedding-model", help="Model name returned by /v1/models")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


@dataclass(slots=True)
class EmbeddingEngine:
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
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()
        self.model.to(self.device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)
                last_hidden_state = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                masked = last_hidden_state * attention_mask
                pooled = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(normalized.cpu().tolist())
        return vectors


def build_app(engine: EmbeddingEngine) -> FastAPI:
    app = FastAPI(title="Local Embedding Server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelListResponse)
    def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[
                ModelCard(
                    id=engine.served_model_name,
                    created=int(time.time()),
                )
            ]
        )

    @app.post("/v1/embeddings")
    def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
        if request.model != engine.served_model_name:
            raise HTTPException(status_code=400, detail=f"unknown model: {request.model}")
        if request.encoding_format != "float":
            raise HTTPException(status_code=400, detail="only encoding_format=float is supported")

        inputs = request.input if isinstance(request.input, list) else [request.input]
        if not inputs:
            raise HTTPException(status_code=400, detail="input must not be empty")

        vectors = engine.embed(inputs)
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector,
                }
                for index, vector in enumerate(vectors)
            ],
            "model": engine.served_model_name,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
        }

    return app


def main() -> None:
    args = parse_args()
    engine = EmbeddingEngine(
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
