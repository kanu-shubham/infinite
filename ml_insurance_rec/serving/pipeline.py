"""
End-to-end serving pipeline — the production-facing entry point.

Mirrors the four-stage architecture from the diagram exactly:

  Stage 1  User Tower      user profile → 128-dim embedding        (DNN inference)
  Stage 2  ANN Index       embedding → top-K candidates             (FAISS lookup)
  Stage 3  Ranking Model   candidates → P(purchase) scores, top 20  (XGBoost)
  Stage 4  Business Rules  diversity + freshness + sponsored slots   (deterministic)

Usage
-----
  from ml_insurance_rec.serving.pipeline import RecommendationPipeline
  from ml_insurance_rec.data.interaction_log import UserProfile

  pipe = RecommendationPipeline.load("artifacts/")
  user = UserProfile(
      user_id="u1", age=32, trip_duration=14,
      max_price_per_day=15.0, min_coverage=100_000,
      preferred_type_idx=2,     # adventure
      secondary_type_idx=1,     # comprehensive
      destination_indices=[1,3,7],
      feature_indices=[1,2,5,0,10],
      style_indices=[0,7],
  )
  result = pipe.recommend(user, top_k_retrieval=500, top_n_final=20)
  for r in result.recommendations:
      print(r["rank"], r["product_id"], f"P(buy)={r['purchase_prob']:.3f}")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.catalogue import (
    InsuranceProduct, load_catalogue,
    FEATURES, DESTINATIONS, STYLES, TYPES,
)
from ..data.feature_engineering import build_ranking_row
from ..data.interaction_log import UserProfile
from ..models.ann_index import ANNIndex
from ..models.ranker import XGBoostRanker
from ..models.two_tower import TwoTowerModel

# ── Business rule constants ───────────────────────────────────────────────────
MAX_PER_PROVIDER    = 3
SPONSORED_POSITIONS = [0, 4]       # 0-indexed slots reserved for sponsored items
STALE_THRESHOLD     = 0.6          # freshnessScore below this → eligible for demotion
FINAL_CAP           = 20


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class StageMetrics:
    stage:       str
    description: str
    latency_ms:  float
    details:     Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    recommendations: List[Dict[str, Any]]   # final ranked list
    stages:          List[StageMetrics]
    total_latency_ms: float

    def print_summary(self) -> None:
        print(f"\n{'─'*60}")
        print(f"Pipeline total: {self.total_latency_ms:.2f} ms")
        for s in self.stages:
            print(f"  [{s.latency_ms:6.3f} ms]  {s.stage:<22} {s.description}")
        print(f"{'─'*60}")
        print(f"Top recommendations:")
        for r in self.recommendations[:5]:
            print(
                f"  #{r['rank']:<2} {r['name']:<35} "
                f"P(buy)={r['purchase_prob']:.3f}  "
                f"${r['price_per_day']:.2f}/day"
            )
        print()


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RecommendationPipeline:
    """
    Loads trained artefacts and serves recommendations.

    Parameters
    ----------
    two_tower  : trained TwoTowerModel
    ann_index  : built ANNIndex
    ranker     : trained XGBoostRanker
    products   : full InsuranceProduct catalogue (used to enrich results)
    """

    def __init__(
        self,
        two_tower: TwoTowerModel,
        ann_index: ANNIndex,
        ranker:    XGBoostRanker,
        products:  List[InsuranceProduct],
    ):
        self._tower   = two_tower
        self._index   = ann_index
        self._ranker  = ranker
        self._cat     = {p.id: p for p in products}

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, artifacts_dir: str | Path = "artifacts") -> "RecommendationPipeline":
        """Load all trained artefacts from a directory."""
        art = Path(artifacts_dir)
        print(f"Loading pipeline artefacts from {art} …")
        tower   = TwoTowerModel.load(art / "two_tower_best.pt")
        index   = ANNIndex.load(art / "ann_index.pkl")
        ranker  = XGBoostRanker.load(art / "ranker.pkl")
        products = load_catalogue(art / "catalogue.json")
        print("Pipeline ready.\n")
        return cls(tower, index, ranker, products)

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage1_user_embedding(self, user: UserProfile) -> Tuple[np.ndarray, StageMetrics]:
        t0  = time.perf_counter()
        emb = self._tower.embed_single_user(user.to_feature_vector())
        ms  = (time.perf_counter() - t0) * 1_000
        metrics = StageMetrics(
            stage       = "1. User Tower",
            description = f"'{user.user_id}' → {len(emb)}-dim embedding",
            latency_ms  = ms,
            details     = {"embed_dim": len(emb), "preferred_type": TYPES[user.preferred_type_idx]},
        )
        return emb, metrics

    def _stage2_ann_retrieval(
        self,
        user_emb: np.ndarray,
        top_k: int,
    ) -> Tuple[List[Tuple[str, float]], StageMetrics]:
        t0 = time.perf_counter()
        scores, pids = self._index.search(user_emb, top_k=top_k)
        ms = (time.perf_counter() - t0) * 1_000

        candidates = list(zip(pids, scores.tolist()))  # [(pid, score), …]
        metrics = StageMetrics(
            stage       = "2. ANN Index",
            description = f"Retrieved {len(candidates)} candidates (top-{top_k})",
            latency_ms  = ms,
            details     = {
                "catalogue_size": len(self._cat),
                "retrieved":      len(candidates),
                "top_ann_score":  round(scores[0], 4) if len(scores) else 0,
            },
        )
        return candidates, metrics

    def _stage3_ranking(
        self,
        user: UserProfile,
        candidates: List[Tuple[str, float]],
        top_n: int,
    ) -> Tuple[List[Dict], StageMetrics]:
        t0 = time.perf_counter()

        rows = []
        meta = []
        for pid, ann_score in candidates:
            product = self._cat.get(pid)
            if product is None:
                continue
            fv = build_ranking_row(user, product, ann_score=ann_score)
            rows.append(fv)
            meta.append({"pid": pid, "ann_score": ann_score})

        if not rows:
            return [], StageMetrics("3. Ranking Model", "No candidates", 0.0)

        X      = np.vstack(rows).astype(np.float32)
        probas = self._ranker.predict_proba(X)          # P(purchase)

        # Zip scores back with metadata, sort descending, take top_n
        scored = sorted(
            zip(meta, probas.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        ranked = []
        for rank, (m, prob) in enumerate(scored, 1):
            product = self._cat[m["pid"]]
            ranked.append({
                "rank":          rank,
                "product_id":    product.id,
                "name":          product.name,
                "provider":      product.provider,
                "type":          product.type,
                "price_per_day": product.price_per_day,
                "coverage_amount": product.coverage_amount,
                "max_trip_days": product.max_trip_days,
                "rating":        product.rating,
                "review_count":  product.review_count,
                "freshness_score": product.freshness_score,
                "is_sponsored":  product.is_sponsored,
                "features":      [FEATURES[i] for i in product.feature_indices],
                "destinations":  [DESTINATIONS[i] for i in product.destination_indices],
                "purchase_prob": round(prob, 4),
                "ann_score":     round(m["ann_score"], 4),
                "_product":      product,   # for business rules
            })

        ms = (time.perf_counter() - t0) * 1_000
        metrics = StageMetrics(
            stage       = "3. Ranking Model",
            description = f"Scored {len(rows)} candidates → top {len(ranked)}",
            latency_ms  = ms,
            details     = {
                "input_candidates": len(rows),
                "output_ranked":    len(ranked),
                "top_purchase_prob": round(ranked[0]["purchase_prob"], 4) if ranked else 0,
            },
        )
        return ranked, metrics

    def _stage4_business_rules(
        self,
        ranked: List[Dict],
    ) -> Tuple[List[Dict], StageMetrics]:
        t0 = time.perf_counter()
        applied = []
        pool    = [r.copy() for r in ranked]

        # Rule 1: Provider diversity
        provider_count: Dict[str, int] = {}
        diverse = []
        for item in pool:
            cnt = provider_count.get(item["provider"], 0)
            if cnt < MAX_PER_PROVIDER:
                provider_count[item["provider"]] = cnt + 1
                diverse.append(item)
        if len(diverse) < len(pool):
            applied.append(f"Provider diversity (max {MAX_PER_PROVIDER}/provider)")
        pool = diverse

        # Rule 2: Freshness demotion — stale items swapped one position down
        freshness_adjusted = False
        for i in range(len(pool) - 1):
            if (pool[i]["freshness_score"] < STALE_THRESHOLD
                    and pool[i + 1]["freshness_score"] >= STALE_THRESHOLD):
                pool[i], pool[i + 1] = pool[i + 1], pool[i]
                freshness_adjusted = True
        if freshness_adjusted:
            applied.append(f"Freshness demotion (threshold={STALE_THRESHOLD})")

        # Rule 3: Sponsored injection at fixed slots
        sponsored    = [p for p in pool if p["is_sponsored"]]
        non_sponsored = [p for p in pool if not p["is_sponsored"]]
        if sponsored:
            pool = non_sponsored
            injected = 0
            for pos in SPONSORED_POSITIONS:
                if injected >= len(sponsored):
                    break
                insert_at = min(pos, len(pool))
                entry = sponsored[injected].copy()
                entry["_sponsored_slot"] = True
                pool.insert(insert_at, entry)
                injected += 1
            applied.append(f"Sponsored injection at positions {SPONSORED_POSITIONS[:injected]}")

        # Rule 4: Hard cap
        results = pool[:FINAL_CAP]
        if len(pool) > FINAL_CAP:
            applied.append(f"Hard cap at {FINAL_CAP}")

        # Re-number ranks after rules
        for i, item in enumerate(results, 1):
            item["rank"] = i
            item.pop("_product", None)   # remove internal field before returning

        ms = (time.perf_counter() - t0) * 1_000
        metrics = StageMetrics(
            stage       = "4. Business Rules",
            description = f"{len(applied)} rule(s) applied → {len(results)} final results",
            latency_ms  = ms,
            details     = {"rules_applied": applied, "final_count": len(results)},
        )
        return results, metrics

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend(
        self,
        user: UserProfile,
        top_k_retrieval: int = 500,
        top_n_final:     int = 20,
    ) -> PipelineResult:
        """
        Run the full four-stage pipeline for one user.

        Parameters
        ----------
        user            : UserProfile describing the traveler.
        top_k_retrieval : number of ANN candidates to retrieve (Stage 2).
        top_n_final     : number of results after ranking (Stage 3), before
                          business rules further filter/inject (Stage 4).

        Returns
        -------
        PipelineResult with recommendations and per-stage metrics.
        """
        t_total = time.perf_counter()

        emb,        s1 = self._stage1_user_embedding(user)
        candidates, s2 = self._stage2_ann_retrieval(emb, top_k=top_k_retrieval)
        ranked,     s3 = self._stage3_ranking(user, candidates, top_n=top_n_final)
        results,    s4 = self._stage4_business_rules(ranked)

        total_ms = (time.perf_counter() - t_total) * 1_000
        return PipelineResult(
            recommendations  = results,
            stages           = [s1, s2, s3, s4],
            total_latency_ms = total_ms,
        )
