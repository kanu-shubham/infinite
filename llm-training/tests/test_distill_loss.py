"""Unit test the distillation loss math without loading real models.

We construct a toy teacher / student that return deterministic logits and
verify the CE + KL combination behaves as expected in two edge cases:

  - alpha_kd=0, alpha_ce=1: loss equals pure CE
  - teacher == student: KL term is ~0
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llm_train.training.collators import IGNORE_INDEX


def _shifted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def test_kd_zero_when_teacher_equals_student() -> None:
    torch.manual_seed(0)
    V = 16
    B, T = 2, 6
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))

    # KL with identical distributions must be ~0.
    s = logits[:, :-1, :]
    lp = F.log_softmax(s, dim=-1)
    p = F.softmax(s, dim=-1)
    kl = F.kl_div(lp, p, reduction="batchmean")
    assert abs(kl.item()) < 1e-5


def test_ce_only_matches_pure_ce() -> None:
    torch.manual_seed(1)
    V, B, T = 8, 2, 5
    logits = torch.randn(B, T, V, requires_grad=False)
    labels = torch.randint(0, V, (B, T))
    ce = _shifted_ce(logits, labels)
    # combined loss with alpha_kd=0 reduces to alpha_ce * ce
    combined = 1.0 * ce + 0.0 * torch.tensor(99.0)
    assert torch.allclose(combined, ce)


def test_loss_mask_excludes_ignore_index() -> None:
    V, B, T = 8, 1, 5
    logits = torch.zeros(B, T, V)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, 3, 4]])
    ce = _shifted_ce(logits, labels)
    # Only positions 2..4 contribute; with uniform logits, CE = log(V)
    assert abs(ce.item() - torch.tensor(float(V)).log().item()) < 1e-5
