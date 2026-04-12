"""
Central configuration for the RAG system.

All settings are read from environment variables with sensible defaults.
Copy .env.example to .env and set ANTHROPIC_API_KEY to run the system.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Claude API ---
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )
    # Generation model. Use claude-haiku-4-5-20251001 for cheaper experiments.
    model: str = field(
        default_factory=lambda: os.environ.get("RAG_MODEL", "claude-sonnet-4-6")
    )

    # --- Retrieval ---
    top_k: int = field(
        default_factory=lambda: int(os.environ.get("RAG_TOP_K", "5"))
    )
    # Minimum cosine similarity to consider a result valid.
    # Documents below this threshold are treated as "no result found".
    min_similarity: float = field(
        default_factory=lambda: float(os.environ.get("RAG_MIN_SIMILARITY", "0.10"))
    )

    # --- Chunking ---
    chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("RAG_CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
    )

    # --- Multi-hop ---
    max_hops: int = field(
        default_factory=lambda: int(os.environ.get("RAG_MAX_HOPS", "3"))
    )

    # --- Generation ---
    temperature: float = 0.1
    max_tokens: int = 1024

    # --- RAPTOR ---
    raptor_cluster_size: int = 5   # chunks per cluster leaf
    raptor_max_levels: int = 3     # how many summary levels to build

    def validate(self) -> None:
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it as an environment variable or add it to a .env file."
            )


# Module-level singleton — importable everywhere as `from src.config import config`
config = Config()
