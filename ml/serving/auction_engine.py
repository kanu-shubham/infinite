"""Glue: pulls features, scores, runs the auction. Returns ranked slate."""
from __future__ import annotations

import numpy as np

from ml.models.auction import run_auction
from ml.models.features import Context, vectorize
from ml.serving.feature_store import FeatureStore
from ml.serving.predictor import Predictor
from ml.serving.schemas import CandidatePayload, ContextPayload, RankedAd, UserPayload


class AuctionEngine:
    def __init__(
        self,
        store: FeatureStore,
        predictor: Predictor,
        *,
        n_slots: int,
        reserve_cpc: float,
    ):
        self._store = store
        self._predictor = predictor
        self._n_slots = n_slots
        self._reserve_cpc = reserve_cpc

    async def rank(
        self,
        user: UserPayload,
        context: ContextPayload,
        candidates: list[CandidatePayload],
    ) -> list[RankedAd]:
        ad_ids = [c.ad_id for c in candidates]
        bids = np.fromiter((c.bid_cpc for c in candidates), dtype=np.float32, count=len(candidates))

        # Parallelise IO: user + ad features are independent.
        import asyncio

        user_task = asyncio.create_task(
            self._store.get_user(user.user_id, user.country, user.device)
        )
        ad_task = asyncio.create_task(self._store.get_ads(ad_ids))
        u_features, ad_features = await asyncio.gather(user_task, ad_task)

        ctx = Context(
            destination=context.destination,
            lead_time_days=context.lead_time_days,
            los=context.los,
            pax=context.pax,
            hour=context.hour,
            dow=context.dow,
        )
        X = vectorize(u_features, ad_features, bids, ctx)
        pctr = self._predictor.score(X).astype(np.float32)
        quality = np.fromiter((a.quality for a in ad_features), dtype=np.float32, count=len(ad_features))

        result = run_auction(
            pctr, bids, quality, n_slots=self._n_slots, reserve_cpc=self._reserve_cpc
        )

        out: list[RankedAd] = []
        for slot, idx in enumerate(result.order):
            out.append(
                RankedAd(
                    ad_id=ad_ids[int(idx)],
                    slot=slot + 1,
                    price_cpc=float(result.prices[slot]),
                    pctr=float(pctr[int(idx)]),
                    score=float(result.scores[slot]),
                )
            )
        return out
