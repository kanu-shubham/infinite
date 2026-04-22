"""Deterministic seeding across Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG we know about.

    Set ``deterministic=True`` only when reproducibility matters more than
    throughput — it disables some cuDNN autotuning and can be noticeably
    slower for training.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
