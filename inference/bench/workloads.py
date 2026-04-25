"""Workload presets for benchmarking.

Two production-relevant shapes:

  * ``shared_system``  -- every request reuses the same long system prompt
    (RAG/system-instructions pattern). Designed to expose Automatic
    Prefix Caching wins.

  * ``mixed``          -- mix of short and long prompts arriving together,
    designed to expose chunked-prefill wins (long prompts no longer
    starve concurrent decodes).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


SYSTEM_PROMPT_LONG = (
    "You are a meticulous senior engineer assistant. Always respond with a "
    "concise direct answer first, followed by a short justification. Avoid "
    "filler. When code is involved, prefer minimal diffs and explicit names. "
    "When uncertain, say so. The current product is an internal developer "
    "tool used by infrastructure engineers; assume the reader is technical "
    "and skip basic explanations. " * 8  # ~1k tokens of shared prefix
)

USER_TASKS = [
    "Summarize the tradeoffs of paged attention vs. classic KV caching.",
    "Explain continuous batching to a backend engineer in 4 bullets.",
    "What does chunked prefill change about TTFT under load?",
    "When does prefix caching hurt instead of help?",
    "Sketch an admission-control policy for an LLM gateway.",
    "How would you size KV cache for a 70B model on 8xH100?",
    "Compare speculative decoding to medusa heads in two sentences.",
    "What metrics matter most for an LLM inference SLO?",
]


@dataclass
class Prompt:
    messages: list[dict]
    max_tokens: int


def build(workload: str, n: int, seed: int = 0) -> list[Prompt]:
    rng = random.Random(seed)
    if workload == "shared_system":
        return [
            Prompt(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_LONG},
                    {"role": "user", "content": rng.choice(USER_TASKS)},
                ],
                max_tokens=128,
            )
            for _ in range(n)
        ]
    if workload == "mixed":
        out: list[Prompt] = []
        for _ in range(n):
            if rng.random() < 0.3:
                # Long prompt, short answer (prefill-heavy).
                out.append(Prompt(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_LONG},
                        {"role": "user", "content": rng.choice(USER_TASKS)},
                    ],
                    max_tokens=64,
                ))
            else:
                # Short prompt, longer answer (decode-heavy).
                out.append(Prompt(
                    messages=[
                        {"role": "user", "content": rng.choice(USER_TASKS)},
                    ],
                    max_tokens=192,
                ))
        return out
    raise ValueError(f"unknown workload: {workload}")
