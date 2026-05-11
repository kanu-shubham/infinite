"""Generalized Second Price auction with quality score multiplier.

For each impression we sort candidates by `rank_score = bid * pCTR * quality`
and charge each winning ad the minimum CPC needed to retain its slot — i.e. the
next-highest rank-score divided by the winner's `pCTR * quality`, plus an
epsilon increment. A `reserve_cpc` floor enforces a minimum revenue per click.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 0.01  # 1¢ increment over second-price truthful threshold.


@dataclass(slots=True)
class AuctionResult:
    order: np.ndarray       # candidate indices, descending rank score
    prices: np.ndarray      # CPC charged at the resulting slot
    scores: np.ndarray      # rank scores aligned with `order`


def run_auction(
    pctr: np.ndarray,
    bids: np.ndarray,
    quality: np.ndarray,
    *,
    n_slots: int,
    reserve_cpc: float = 0.05,
) -> AuctionResult:
    """All inputs must be 1D arrays of equal length aligned with the candidate list."""
    rank = bids * pctr * quality
    # argpartition + sort on the top-k slice is O(n) + O(k log k); for k≪n this is
    # ~3-5x faster than a full sort when n>=50.
    k = min(n_slots, rank.shape[0])
    if k <= 0:
        return AuctionResult(np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))

    top_unsorted = np.argpartition(-rank, k - 1)[:k] if k < rank.shape[0] else np.arange(rank.shape[0])
    top = top_unsorted[np.argsort(-rank[top_unsorted])]
    top_scores = rank[top]

    # next_score[i] is the rank score of the slot below i. For the last winning
    # slot we use the best losing score (reserve-equivalent if none exists).
    if rank.shape[0] > k:
        next_best_loser = rank[np.argpartition(-rank, k - 1)[k]]
    else:
        next_best_loser = 0.0
    next_score = np.empty(k, dtype=np.float32)
    next_score[:-1] = top_scores[1:]
    next_score[-1] = next_best_loser

    eff = pctr[top] * quality[top]
    # Guard against zero pCTR/quality (degenerate cold ads): fall back to bid.
    safe_eff = np.where(eff > 1e-9, eff, 1.0)
    prices = next_score / safe_eff + EPS
    prices = np.minimum(prices, bids[top])     # never charge above the bid
    prices = np.maximum(prices, reserve_cpc)   # never below reserve
    return AuctionResult(order=top, prices=prices.astype(np.float32), scores=top_scores)
