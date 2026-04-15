"""
Evaluate the fine-tuned RN-CodeGen model on the held-out test set.

Metrics computed
----------------
  BLEU-4   — standard n-gram overlap metric for code generation
             (sacrebleu, corpus-level)
  CodeBLEU — BLEU variant that also rewards syntactic/structural similarity
             (optional; requires tree-sitter)

Usage
-----
python evaluation/evaluate.py --model_dir ./checkpoints

Outputs a results table and saves evaluation/results.json.
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sacrebleu
from tqdm import tqdm

from inference.generate import RNCodeGenerator
from training.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_test_set(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_bleu(hypotheses: list[str], references: list[str]) -> float:
    """
    Compute corpus-level BLEU-4 with sacrebleu.

    sacrebleu expects references as a list of lists (one list per reference
    translation), but for code generation we have a single reference per example.
    """
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score   # 0–100 float


def exact_match_accuracy(hypotheses: list[str], references: list[str]) -> float:
    """Fraction of examples where the generated code exactly matches the reference."""
    matches = sum(h.strip() == r.strip() for h, r in zip(hypotheses, references))
    return matches / len(references) * 100


def token_f1(hyp: str, ref: str) -> float:
    """Token-level F1 between a single prediction and reference."""
    hyp_tokens = set(hyp.split())
    ref_tokens = set(ref.split())
    if not hyp_tokens or not ref_tokens:
        return 0.0
    precision = len(hyp_tokens & ref_tokens) / len(hyp_tokens)
    recall    = len(hyp_tokens & ref_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall) * 100


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(model_dir: str, test_file: str, num_beams: int = cfg.num_beams):
    gen = RNCodeGenerator(model_dir)
    test_examples = load_test_set(test_file)

    logger.info(f"Evaluating on {len(test_examples)} test examples …")

    hypotheses: list[str] = []
    references: list[str] = []
    per_example_results: list[dict] = []

    for ex in tqdm(test_examples, desc="Generating"):
        predicted = gen.generate(ex["input"], num_beams=num_beams)
        reference = ex["target"]

        hypotheses.append(predicted)
        references.append(reference)
        per_example_results.append({
            "prompt":    ex["input"],
            "reference": reference,
            "predicted": predicted,
            "token_f1":  round(token_f1(predicted, reference), 2),
        })

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    bleu  = compute_bleu(hypotheses, references)
    em    = exact_match_accuracy(hypotheses, references)
    avg_f1 = sum(r["token_f1"] for r in per_example_results) / len(per_example_results)

    metrics = {
        "num_examples":      len(test_examples),
        "bleu_4":            round(bleu, 2),
        "exact_match_pct":   round(em, 2),
        "avg_token_f1_pct":  round(avg_f1, 2),
        "model_dir":         model_dir,
        "num_beams":         num_beams,
    }

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  RN-CODEGEN EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Examples evaluated : {metrics['num_examples']}")
    print(f"  BLEU-4             : {metrics['bleu_4']:.2f}")
    print(f"  Exact Match        : {metrics['exact_match_pct']:.2f}%")
    print(f"  Avg Token F1       : {metrics['avg_token_f1_pct']:.2f}%")
    print("=" * 50)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "per_example": per_example_results}, f, indent=2)
    logger.info(f"Full results saved to: {out_path}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=cfg.output_dir)
    p.add_argument("--test_file", default=cfg.test_file)
    p.add_argument("--num_beams", type=int, default=cfg.num_beams)
    args = p.parse_args()

    evaluate(args.model_dir, args.test_file, args.num_beams)
