from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Telegram ---
    bot_token: str = ""
    # Telegram user ids allowed to run /reindex (comma separated). Empty = nobody.
    admin_ids: Annotated[list[int], NoDecode] = Field(default_factory=list)

    # --- Postgres ---
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    # --- Crawler ---
    start_urls: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["https://osmi-it.ru/", "https://osmi-ai.ru/", "https://docs.osmi-ai.ru/"]
    )
    allowed_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["osmi-it.ru", "osmi-ai.ru", "docs.osmi-ai.ru"]
    )
    crawl_max_pages: int = 600
    crawl_concurrency: int = 6
    crawl_timeout: float = 30.0
    crawl_user_agent: str = "Mozilla/5.0 (compatible; osmi-rag-bot/0.1; knowledge-base indexer)"
    # Re-crawl the sites on bot startup when the knowledge base is empty.
    auto_ingest: bool = True

    # --- Chunking ---
    chunk_size: int = 1200  # characters
    chunk_overlap: int = 200

    # --- Embeddings ---
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    # e5-small: ~120M params, fast on CPU, solid Russian quality. For better recall at the
    # cost of ~2.5x slower indexing: intfloat/multilingual-e5-base (768) or -large (1024).
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    fastembed_cache_dir: str = "/models"

    # --- Retrieval ---
    top_k: int = 8
    candidates_k: int = 30  # per retriever, before RRF fusion
    rrf_k: int = 20  # tuned on eval/questions.json (60 -> 20: +2 hits@8)

    # --- LLM ---
    llm_provider: Literal["anthropic", "gemini", "openai", "none"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    # Claude effort level: low | medium | high | xhigh | max. "low" keeps chat replies fast.
    llm_effort: str = "low"
    # Backup model on the same provider, used when the main one is overloaded or out of quota.
    llm_fallback_model: str = ""
    llm_max_tokens: int = 4000
    llm_timeout: float = 90.0
    # Claude: retry a safety-classifier refusal on another model server-side.
    llm_refusal_fallback: bool = True

    anthropic_api_key: str = ""
    anthropic_base_url: str | None = None

    # Any OpenAI-compatible endpoint: OpenAI, OpenRouter, DeepSeek, Ollama, vLLM, YandexGPT...
    openai_api_key: str = ""
    # Google Gemini (key from https://aistudio.google.com/api-keys), via its OpenAI-compatible endpoint
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_base_url: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- Conversation memory ---
    history_turns: int = 4

    # --- Langfuse (optional tracing) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    log_level: str = "INFO"

    @field_validator("start_urls", "allowed_hosts", "admin_ids", mode="before")
    @classmethod
    def _split_csv(cls, value):
        return _csv(value) if isinstance(value, str) else value

    @field_validator("allowed_hosts")
    @classmethod
    def _lower_hosts(cls, value: list[str]) -> list[str]:
        return [h.lower() for h in value]


@lru_cache
def get_settings() -> Settings:
    return Settings()
