"""Application settings, loaded from environment variables and an optional `.env` file.

Every secret (database URLs, API tokens) is a `SecretStr`, so it never shows up in
logs, reprs, or tracebacks by accident. Call `.get_secret_value()` only at the point
where the value is handed to a driver or client.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DESCRIPTIONS_PATH = DATA_DIR / "descriptions.yaml"

LLMProvider = Literal["ollama", "hf", "groq"]


class Settings(BaseSettings):
    """All runtime configuration for TalkTable. Field names map to UPPER_CASE env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # `HF_TOKEN=` in .env means "unset", not ""
        extra="ignore",
    )

    # --- Databases ---------------------------------------------------------
    sqlite_path: Path = Path("data/bookstore.db")
    postgres_url: SecretStr | None = None
    mongo_url: SecretStr | None = None
    mongo_db: str = "bookstore"

    # Write-capable URLs, used only by data/seed/*. The app never touches them.
    postgres_seed_url: SecretStr | None = None
    mongo_seed_url: SecretStr | None = None

    # --- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = "ollama"
    llm_model: str = "qwen3:14b"
    # Optional second model just for writing SQL / Mongo pipelines (generation and
    # self-correction), e.g. a code model. Unset means LLM_MODEL does every step.
    llm_query_model: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    hf_token: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    # Local CPU/GPU inference can be slow; hosted APIs answer in a few seconds.
    llm_timeout_seconds: float = Field(default=180.0, gt=0)

    # --- Retrieval ---------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Messages less similar than this to every database are treated as small talk or
    # off-topic. Tuned for all-MiniLM-L6-v2; re-measure if you change the model.
    min_relevance: float = Field(default=0.38, ge=0, le=1)
    # Estimated tokens of schema in the query-writing prompt; bigger selections are shown
    # more briefly, then trimmed.
    schema_token_budget: int = Field(default=2000, ge=200)

    # --- Vector store ------------------------------------------------------
    # numpy: in memory, re-embedded at every start-up (fine for a few hundred tables).
    # qdrant: a Qdrant collection, embedded in QDRANT_PATH or on a server at QDRANT_URL;
    # vectors persist, so a restart only embeds what changed.
    # auto: qdrant when QDRANT_URL is set or the schemas have VECTOR_STORE_AUTO_TABLES
    # tables or more, numpy otherwise.
    vector_store: Literal["auto", "numpy", "qdrant"] = "auto"
    vector_store_auto_tables: int = Field(default=300, ge=1)
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_path: Path = Path("data/qdrant")  # embedded mode: one process at a time per folder
    qdrant_collection: str = "talktable_schema"

    # --- Answer cache --------------------------------------------------------
    # Repeated questions skip the LLM: the cached query is re-run on the current data.
    cache_enabled: bool = True
    cache_path: Path = Path("data/answer_cache.sqlite")
    cache_ttl_hours: float = Field(default=168.0, gt=0)
    # Final sanity check for reworded questions (they must also share their words and anchors).
    cache_similarity: float = Field(default=0.80, ge=0, le=1)

    # --- Limits ------------------------------------------------------------
    # The guardrails cap every query at this many rows. The spec's hard ceiling is 200.
    max_rows: int = Field(default=200, ge=1, le=200)
    query_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    log_level: str = "INFO"

    @property
    def cache_file(self) -> Path:
        path = self.cache_path.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def qdrant_dir(self) -> Path:
        path = self.qdrant_path.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path

    def vector_backend(self, table_count: int) -> str:
        """The vector store to use ("numpy" or "qdrant") for schemas of this size."""
        if self.vector_store != "auto":
            return self.vector_store
        return "qdrant" if self.qdrant_url or table_count >= self.vector_store_auto_tables else "numpy"

    @property
    def sqlite_file(self) -> Path:
        """SQLite path resolved against the project root, so the CLI works from any cwd."""
        path = self.sqlite_path.expanduser()
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()


def mask_url(url: str) -> str:
    """Hide the password in a connection URL so it is safe to print or log.

    >>> mask_url("postgresql://user:secret@host:5432/db")
    'postgresql://user:***@host:5432/db'
    """
    parts = urlsplit(url)
    if parts.password is None:
        return url
    netloc = parts.netloc.replace(f":{parts.password}@", ":***@", 1)
    return urlunsplit(parts._replace(netloc=netloc))
