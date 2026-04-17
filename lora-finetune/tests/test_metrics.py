from __future__ import annotations

import math

from lora_finetune.evaluation.metrics import exact_match, perplexity_from_loss


def test_perplexity_matches_exp() -> None:
    assert perplexity_from_loss(0.0) == 1.0
    assert math.isclose(perplexity_from_loss(1.0), math.e, rel_tol=1e-9)


def test_perplexity_overflow_guard() -> None:
    assert perplexity_from_loss(1e9) == float("inf")


def test_exact_match_ignores_whitespace() -> None:
    assert exact_match(["yes ", " no"], ["yes", "no"]) == 1.0
    assert exact_match(["yes", "maybe"], ["yes", "no"]) == 0.5
    assert exact_match([], []) == 0.0
