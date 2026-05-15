from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(name: str, fallback_name: str | None = None, default: str = "") -> str:
    value = os.getenv(name)
    if value is None and fallback_name:
        value = os.getenv(fallback_name)
    return (value if value is not None else default).strip()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = BASE_DIR
    raw_manual_dir: Path = BASE_DIR / "KownledgeBase" / "手册"
    image_dir: Path = BASE_DIR / "KownledgeBase" / "手册" / "插图"
    parsed_manual_path: Path = BASE_DIR / "data" / "parsed" / "manuals.jsonl"
    chunk_path: Path = BASE_DIR / "data" / "chunks" / "manual_chunks.jsonl"
    retrieval_corpus_path: Path = BASE_DIR / "data" / "chunks" / "retrieval_corpus.jsonl"
    dense_index_path: Path = BASE_DIR / "data" / "index" / "retrieval_dense_index.jsonl"
    image_path: Path = BASE_DIR / "data" / "chunks" / "images.jsonl"
    metadata_path: Path = BASE_DIR / "data" / "index" / "metadata.json"
    default_top_k: int = int(os.getenv("TOP_K", "5"))
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
    max_session_turns: int = int(os.getenv("MAX_SESSION_TURNS", "6"))
    api_token: str = os.getenv("KAFU_API_TOKEN", "").strip()
    llm_enabled: bool = _env_flag("LLM_ENABLED", _env_flag("AGENT_LLM_ENABLED", False))
    llm_base_url: str = _env_value("LLM_BASE_URL", "AGENT_LLM_BASE_URL")
    llm_api_key: str = _env_value("LLM_API_KEY", "AGENT_LLM_API_KEY")
    llm_model: str = _env_value("LLM_MODEL", "AGENT_LLM_MODEL")
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    dense_enabled: bool = _env_flag("DENSE_ENABLED", False)
    dense_base_url: str = os.getenv("DENSE_BASE_URL", "").strip()
    dense_api_key: str = os.getenv("DENSE_API_KEY", "").strip()
    dense_model: str = os.getenv("DENSE_MODEL", "").strip()
    dense_timeout_seconds: float = float(os.getenv("DENSE_TIMEOUT_SECONDS", "45"))
    dense_batch_size: int = int(os.getenv("DENSE_BATCH_SIZE", "32"))
    dense_top_k: int = int(os.getenv("DENSE_TOP_K", "8"))
    rerank_enabled: bool = _env_flag("RERANK_ENABLED", False)
    rerank_base_url: str = os.getenv("RERANK_BASE_URL", "").strip()
    rerank_api_key: str = os.getenv("RERANK_API_KEY", "").strip()
    rerank_model: str = os.getenv("RERANK_MODEL", "").strip()
    rerank_timeout_seconds: float = float(os.getenv("RERANK_TIMEOUT_SECONDS", "45"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "5"))
    rerank_max_candidates: int = int(os.getenv("RERANK_MAX_CANDIDATES", "24"))
    llm_planner_enabled: bool = _env_flag("LLM_PLANNER_ENABLED", _env_flag("AGENT_LLM_QUERY_PLANNER_ENABLED", False))
    llm_evidence_judge_enabled: bool = _env_flag("LLM_EVIDENCE_JUDGE_ENABLED", False)
    llm_fact_extractor_enabled: bool = _env_flag("LLM_FACT_EXTRACTOR_ENABLED", False)
    llm_answer_verifier_enabled: bool = _env_flag("LLM_ANSWER_VERIFIER_ENABLED", False)
    llm_vision_enabled: bool = _env_flag("LLM_VISION_ENABLED", _env_flag("AGENT_LLM_VISION_ENABLED", False))
    weak_evidence_fallback_enabled: bool = _env_flag("WEAK_EVIDENCE_FALLBACK_ENABLED", True)
    product_scope_hard_filter: bool = _env_flag("PRODUCT_SCOPE_HARD_FILTER", True)
    exclude_terms_hard_reject: bool = _env_flag("EXCLUDE_TERMS_HARD_REJECT", True)
    image_from_accepted_evidence_only: bool = _env_flag("IMAGE_FROM_ACCEPTED_EVIDENCE_ONLY", True)
    retrieval_max_retry: int = int(os.getenv("RETRIEVAL_MAX_RETRY", "3"))
    retrieval_top_k_per_query: int = int(os.getenv("RETRIEVAL_TOP_K_PER_QUERY", "20"))
    evidence_accept_threshold: float = float(os.getenv("EVIDENCE_ACCEPT_THRESHOLD", "0.45"))
    session_memory_enabled: bool = _env_flag("SESSION_MEMORY_ENABLED", True)
    trace_enabled: bool = _env_flag("TRACE_ENABLED", True)

    def ensure_data_dirs(self) -> None:
        self.parsed_manual_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval_corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self.dense_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
