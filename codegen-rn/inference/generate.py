"""
Code generation with the fine-tuned RN-CodeGen model.

Usage (CLI)
-----------
python inference/generate.py \
    --prompt "Create a React Native FlatList with pull-to-refresh" \
    --model_dir ./checkpoints

Usage (Python)
--------------
from inference.generate import RNCodeGenerator

gen = RNCodeGenerator("./checkpoints")
code = gen.generate("Create a React Native search bar with debounce")
print(code)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from training.config import cfg


class RNCodeGenerator:
    """
    Wraps a fine-tuned CodeT5 model and exposes a simple .generate() method.

    Parameters
    ----------
    model_dir : str
        Path to the directory produced by train.py (contains config.json,
        pytorch_model.bin / model.safetensors, and tokenizer files).
    device : str | None
        "cuda", "cpu", or None (auto-detect).
    """

    TASK_PREFIX = "Generate React Native code: "

    def __init__(self, model_dir: str = cfg.output_dir, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_dir}  (device: {self.device})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        num_beams: int = cfg.num_beams,
        max_new_tokens: int = cfg.max_target_length,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
    ) -> str | list[str]:
        """
        Generate React Native code from a natural-language prompt.

        Parameters
        ----------
        prompt : str
            Natural-language description, e.g.
            "Create a bottom tab navigator with Home and Profile screens"
        num_beams : int
            Beam-search width.  Higher = better quality but slower.
            Use num_beams=1 with temperature > 0 for sampling.
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature (only applied when num_beams == 1).
        num_return_sequences : int
            Number of independent completions to return.

        Returns
        -------
        str or list[str]
            Generated code.  Returns a single string when
            num_return_sequences == 1, otherwise a list.
        """
        # Always prefix with the task tag used during training
        full_prompt = self.TASK_PREFIX + prompt

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=cfg.max_input_length,
            truncation=True,
        ).to(self.device)

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=cfg.early_stopping,
            num_return_sequences=num_return_sequences,
        )

        # Add temperature / sampling only when not using beam search
        if num_beams == 1:
            generate_kwargs.update(do_sample=True, temperature=temperature)

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)

        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0] if num_return_sequences == 1 else decoded


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="RN-CodeGen inference")
    p.add_argument("--prompt", required=True, help="Natural-language description")
    p.add_argument("--model_dir", default=cfg.output_dir, help="Path to fine-tuned model")
    p.add_argument("--num_beams", type=int, default=cfg.num_beams)
    p.add_argument("--max_new_tokens", type=int, default=cfg.max_target_length)
    p.add_argument("--n", type=int, default=1, help="Number of completions to return")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    gen = RNCodeGenerator(args.model_dir)
    result = gen.generate(
        args.prompt,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.n,
    )

    print("\n" + "=" * 60)
    print("GENERATED REACT NATIVE CODE")
    print("=" * 60)
    if isinstance(result, list):
        for i, code in enumerate(result, 1):
            print(f"\n--- Completion {i} ---\n{code}")
    else:
        print(result)
