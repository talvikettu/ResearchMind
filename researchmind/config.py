from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_", env_file=".env", extra="ignore", env_ignore_empty=True
    )
    data_dir: Path = Path("data")
    embeddings: str = "demo"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker: str = "lexical"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    qdrant_url: str = ""
    qdrant_path: Path | None = None
    llm_url: str = ""
    llm_key: str = ""
    llm_model: str = ""
    input_price: float | None = None
    output_price: float | None = None
    tavily_key: str = ""
    model_cache: Path = Path("data/models/huggingface/hub")
    retrieval_device: str = "cpu"
    embedding_batch: int = 24
    reranker_batch: int = 12
    answer_backend: str = "extractive"
    ollama_url: str = "http://127.0.0.1:11435"
    answer_model: str = "qwen2.5:3b"
    answer_context: int = 6144
    answer_max_tokens: int = 2000
    answer_timeout: int = 180
    chunk_words: int = 140
    search_pool: int = Field(default=20, ge=5, le=200)
    rerank_candidates: int = Field(default=40, ge=5, le=400)
    dense_weight: float = Field(default=0.5, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1)
    handbook_boost: float = 0.0
    title_weight: float = 0.0
    retrieval_k: int = Field(default=8, ge=1, le=40)
    evidence_limit: int = Field(default=10, ge=1, le=20)
    per_document: int = Field(default=3, ge=1, le=10)
    diversity_penalty: float = 0.15
    parent_max_chars: int = 5000
    collapse_parents: bool = False
    entity_routing: bool = False
    technical_expansion: bool = False
    strict_entity_evidence: bool = False
    answer_profile: Literal["legacy", "structured", "grounded"] = "legacy"
    teaching_precision: bool = False
    answer_temperature: float = Field(default=0.0, ge=0, le=2)
    answer_sources: int = Field(default=6, ge=1, le=12)
    answer_source_first: bool = False
    clean_evidence: bool = False
