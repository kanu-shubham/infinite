"""Lightweight evaluation metrics used by the evaluator.

Keeps heavyweight `evaluate`-library metrics (ROUGE/BLEU) optional — they are
only imported when actually requested.
"""
from __future__ import annotations

import math
from typing import Sequence


def perplexity_from_loss(loss: float) -> float:
    return float(math.exp(loss)) if loss < 50 else float("inf")


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    hits = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return hits / len(predictions)


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    import evaluate  # lazy

    rouge = evaluate.load("rouge")
    return {k: float(v) for k, v in rouge.compute(predictions=predictions, references=references).items()}


def bleu_score(predictions: Sequence[str], references: Sequence[str]) -> float:
    import evaluate  # lazy

    bleu = evaluate.load("bleu")
    result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    return float(result["bleu"])
