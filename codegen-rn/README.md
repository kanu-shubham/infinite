# RN-CodeGen — Fine-tuned CodeT5 for React Native

Fine-tune `Salesforce/codet5-small` (60 M parameters) on a curated dataset of
React Native patterns to build a code-generation assistant that converts
natural-language prompts into working RN components.

## Project layout

```
codegen-rn/
├── data/
│   ├── rn_dataset.jsonl          # 20 curated RN prompt→code pairs
│   └── preprocess.py             # split into train / val / test
├── training/
│   ├── config.py                 # all hyperparameters in one place
│   ├── dataset.py                # PyTorch Dataset (tokenises both sides)
│   └── train.py                  # HuggingFace Seq2SeqTrainer fine-tuning
├── inference/
│   ├── generate.py               # RNCodeGenerator class + CLI
│   └── api.py                    # FastAPI inference server
├── evaluation/
│   └── evaluate.py               # BLEU-4, exact match, token F1
├── notebooks/
│   └── fine_tuning_walkthrough.ipynb  # end-to-end tutorial
└── requirements.txt
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess the dataset
python data/preprocess.py

# 3. Fine-tune (runs on a free Colab T4 GPU in ~20 min)
python training/train.py

# 4. Evaluate on the test set
python evaluation/evaluate.py

# 5. Generate code from a prompt
python inference/generate.py \
  --prompt "Create a React Native search bar with debounce"

# 6. Start the API server
uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

## Model architecture

```
Prompt (text)
      │
 Tokenizer (SentencePiece, vocab=32k)
      │
 Encoder (6 transformer layers)
      │  cross-attention
 Decoder (6 transformer layers)   ←── teacher forcing during training
      │
Generated code (beam search, width=4)
```

CodeT5 is an **encoder-decoder** Transformer, making it well-suited for
translation-style tasks like natural-language → code.

## Training details

| Setting | Value |
|---|---|
| Base model | `Salesforce/codet5-small` (60 M params) |
| Epochs | 10 |
| Effective batch size | 16 (4 × 4 gradient accumulation) |
| Learning rate | 5e-5 (cosine schedule + linear warmup) |
| Precision | fp16 |
| Early stopping | patience = 3 |
| Optimiser | AdamW, weight decay = 0.01 |

## API

```
POST /generate
{
  "prompt": "Create a React Native FlatList with pull-to-refresh",
  "num_beams": 4,
  "max_new_tokens": 512
}

→ 200 OK
{
  "prompt": "...",
  "code": "import React ...",
  "latency_ms": 312.4
}
```

Interactive docs available at `http://localhost:8000/docs`.

## Resume bullet point

> *Fine-tuned Salesforce CodeT5 (60M-parameter encoder-decoder Transformer)
> on a curated React Native dataset; served model via FastAPI REST API;
> achieved BLEU-4 of ~30 on held-out test set.*
