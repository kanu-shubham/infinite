from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    model_path: Path
    redis_url: str | None
    feature_ttl_s: int
    user_cache_size: int
    ad_cache_size: int
    n_slots: int
    reserve_cpc: float
    max_candidates: int

    @classmethod
    def from_env(cls) -> "Config":
        default_model = Path(__file__).resolve().parents[1] / "models" / "artifacts" / "ranker.txt"
        return cls(
            model_path=Path(os.environ.get("RANKER_MODEL_PATH", str(default_model))),
            redis_url=os.environ.get("REDIS_URL") or None,
            feature_ttl_s=int(os.environ.get("FEATURE_TTL_S", "10")),
            user_cache_size=int(os.environ.get("USER_CACHE_SIZE", "50000")),
            ad_cache_size=int(os.environ.get("AD_CACHE_SIZE", "200000")),
            n_slots=int(os.environ.get("AD_SLOTS", "8")),
            reserve_cpc=float(os.environ.get("RESERVE_CPC", "0.05")),
            max_candidates=int(os.environ.get("MAX_CANDIDATES", "200")),
        )
