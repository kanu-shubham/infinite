"""
Dataset preprocessing for RN-CodeGen fine-tuning.

Reads the raw JSONL dataset, tokenizes prompt-code pairs into
the CodeT5 seq2seq format, splits into train/val/test, and
saves the processed splits as JSONL files.
"""

import json
import random
from pathlib import Path

random.seed(42)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_for_codet5(example: dict) -> dict:
    """
    CodeT5 expects:
      - input:  the natural-language prompt
      - target: the code to generate
    We prefix the prompt with a task tag so the model learns the task type.
    """
    return {
        "input": f"Generate React Native code: {example['prompt']}",
        "target": example["code"],
    }


def split_dataset(examples: list[dict], val_ratio=0.1, test_ratio=0.1) -> tuple:
    random.shuffle(examples)
    n = len(examples)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    test = examples[:n_test]
    val = examples[n_test : n_test + n_val]
    train = examples[n_test + n_val :]
    return train, val, test


def save_jsonl(examples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples -> {path}")


def main():
    raw_path = Path(__file__).parent / "rn_dataset.jsonl"
    out_dir = Path(__file__).parent / "processed"

    raw = load_jsonl(str(raw_path))
    formatted = [format_for_codet5(ex) for ex in raw]
    train, val, test = split_dataset(formatted)

    save_jsonl(train, str(out_dir / "train.jsonl"))
    save_jsonl(val, str(out_dir / "val.jsonl"))
    save_jsonl(test, str(out_dir / "test.jsonl"))

    print(f"\nDataset summary:")
    print(f"  Total:      {len(formatted)}")
    print(f"  Train:      {len(train)}")
    print(f"  Validation: {len(val)}")
    print(f"  Test:       {len(test)}")


if __name__ == "__main__":
    main()
