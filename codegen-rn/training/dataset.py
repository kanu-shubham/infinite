"""
PyTorch Dataset for RN-CodeGen.

Each example is a (prompt, code) pair stored in a JSONL file with the
keys "input" and "target" (produced by data/preprocess.py).

The dataset tokenizes both sides and returns the tensors expected by
HuggingFace's Seq2SeqTrainer:

  input_ids       – tokenized prompt
  attention_mask  – 1 for real tokens, 0 for padding
  labels          – tokenized target code (-100 on padding positions so
                    the cross-entropy loss ignores them)
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RNCodeDataset(Dataset):
    """React Native code-generation dataset in seq2seq format."""

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 128,
        max_target_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.examples = self._load(file_path)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, path: str) -> list[dict]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {path}\n"
                "Run `python data/preprocess.py` first."
            )
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return self._encode(ex["input"], ex["target"])

    # ── Tokenization ──────────────────────────────────────────────────────────

    def _encode(self, source: str, target: str) -> dict:
        # Encode the prompt (encoder input)
        model_inputs = self.tokenizer(
            source,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Encode the target code (decoder input / labels)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Replace padding token ids in labels with -100 so they are ignored
        # by PyTorch's CrossEntropyLoss.
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         label_ids,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def decode_example(self, idx: int) -> dict:
        """Return the raw text for a given index (useful for debugging)."""
        return self.examples[idx]
