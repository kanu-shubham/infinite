"""
Project 19: Fine-Tuning BERT for Text Classification
======================================================
Use HuggingFace Transformers to fine-tune BERT on sentiment analysis.
BERT = Bidirectional Encoder Representations from Transformers (Google, 2018).

What you'll learn:
- BERT's pretraining: Masked LM + Next Sentence Prediction
- Tokenization: WordPiece, [CLS], [SEP], attention masks
- Fine-tuning: add a classification head on top of [CLS] representation
- HuggingFace ecosystem: AutoTokenizer, AutoModel, Trainer
- Comparing BERT to the from-scratch Transformer (Project 17)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# HuggingFace
try:
    from transformers import (
        AutoTokenizer, AutoModel,
        BertTokenizer, BertModel,
        get_linear_schedule_with_warmup,
    )
    HAS_HF = True
except ImportError:
    HAS_HF = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"    # 110M params, uncased English


# ── Dataset ───────────────────────────────────────────────────────────────

def create_sentiment_dataset():
    """Extended sentiment dataset (positive=1, negative=0)."""
    positives = [
        "This movie was absolutely fantastic and I loved every moment of it.",
        "The acting was superb and the storyline kept me engaged throughout.",
        "A masterpiece of modern cinema. Highly recommend to everyone.",
        "I was blown away by the special effects and the brilliant performances.",
        "One of the best films I have ever seen in my entire life.",
        "The director did an incredible job bringing this story to life beautifully.",
        "I laughed, I cried, and I left the theater feeling truly inspired.",
        "Outstanding performances from the entire cast. A must-see film.",
        "The cinematography was breathtaking and the music was perfect.",
        "A heartwarming and deeply moving film that everyone should experience.",
        "Brilliant writing combined with flawless direction makes this a classic.",
        "I have seen thousands of movies and this ranks among the very best.",
        "The chemistry between the leads was electric and completely believable.",
        "A joyful, uplifting experience that left me smiling for days afterward.",
        "Every scene was carefully crafted with attention to detail and passion.",
        "The plot twists were genuinely surprising and kept me guessing throughout.",
        "An emotional rollercoaster that is both entertaining and thought-provoking.",
        "Simply put, this is cinema at its absolute finest.",
        "I would watch this movie again and again without any hesitation.",
        "The screenplay was sharp, witty, and genuinely moving at key moments.",
        "Excellent pacing, beautiful visuals, and an unforgettable soundtrack.",
        "This film exceeded all my expectations in every possible way.",
        "The performances were so natural and authentic it felt like real life.",
        "A profound and beautiful film that stays with you long after watching.",
        "Pure magic from beginning to end. A true cinematic treasure.",
    ]

    negatives = [
        "This movie was a complete waste of time and money. Avoid at all costs.",
        "The plot made absolutely no sense and the acting was cringe-worthy.",
        "I walked out after thirty minutes. One of the worst films ever made.",
        "Boring, predictable, and utterly disappointing from start to finish.",
        "The special effects looked cheap and the story was completely incoherent.",
        "I cannot believe this garbage got a theatrical release. Truly terrible.",
        "Two hours of my life I will never get back. Absolutely awful film.",
        "The dialogue was so bad it was almost funny but mostly just painful.",
        "Terrible pacing, flat characters, and a completely nonsensical ending.",
        "The director clearly had no idea what they were doing with this mess.",
        "A forgettable slog through a story that nobody asked to see told.",
        "I fell asleep twice and woke up to find the plot no more coherent.",
        "The worst acting I have seen in decades. Simply unwatchable garbage.",
        "Every cliché in the book crammed into two unbearable hours of film.",
        "The script was obviously written by someone who has never seen a movie.",
        "An insulting waste of talent and resources. Deeply disappointing film.",
        "Confusing, boring, and utterly devoid of any redeeming qualities.",
        "I wanted to leave but stayed hoping it would improve. It never did.",
        "The cinematography was amateurish and the editing was a complete mess.",
        "A complete disaster of a film that fails on every conceivable level.",
        "Poorly written, poorly directed, and poorly acted. Just awful overall.",
        "This film is a testament to what happens when studios greenlight garbage.",
        "Unforgivably bad and a massive disappointment given the talented cast.",
        "The story was stolen from better films and executed far worse.",
        "I have never been so bored and irritated simultaneously in a cinema.",
    ]

    reviews = positives + negatives
    labels  = [1] * len(positives) + [0] * len(negatives)

    # Augment with simple variations
    aug_reviews, aug_labels = [], []
    for review, label in zip(reviews, labels):
        aug_reviews.append(review)
        aug_labels.append(label)
        words = review.split()
        if len(words) > 5:
            idx = np.random.randint(1, len(words) - 1)
            filler = "truly" if label == 1 else "extremely"
            words.insert(idx, filler)
            aug_reviews.append(" ".join(words))
            aug_labels.append(label)

    indices = np.random.permutation(len(aug_reviews))
    return (
        [aug_reviews[i] for i in indices],
        [aug_labels[i]  for i in indices],
    )


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── BERT Fine-Tuning Model ────────────────────────────────────────────────

class BERTSentimentClassifier(nn.Module):
    """
    Fine-tuned BERT for sentiment classification.

    Strategy: take the [CLS] token representation from BERT's last hidden state
    and feed it through a classification head.

    [CLS] token is prepended to every sequence and trained during pretraining
    to aggregate the meaning of the whole sequence.
    """
    def __init__(self, bert_model, n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = bert_model
        hidden_size = bert_model.config.hidden_size  # 768 for bert-base

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.last_hidden_state: (B, seq_len, 768)
        # outputs.pooler_output:     (B, 768) — [CLS] token, linearly transformed
        cls_output = outputs.pooler_output   # use BERT's pooled [CLS] representation

        return self.classifier(cls_output)


# ── Training ──────────────────────────────────────────────────────────────

def train_bert(tokenizer, bert_model, train_texts, train_labels, test_texts, test_labels):
    print("\n── Fine-Tuning BERT ──\n")

    N_EPOCHS   = 4
    BATCH_SIZE = 16
    LR         = 2e-5  # Critical: very small LR for fine-tuning pretrained models

    # Datasets
    train_ds = SentimentDataset(train_texts, train_labels, tokenizer)
    test_ds  = SentimentDataset(test_texts,  test_labels,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model
    model = BERTSentimentClassifier(bert_model).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_params:,}")
    print(f"Trainable params: {n_trainable:,}\n")

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()
    history = {"train_acc": [], "test_acc": [], "train_loss": []}

    for epoch in range(1, N_EPOCHS + 1):
        # Train
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            out = model(input_ids, attn_mask)
            loss = criterion(out, labels)
            loss.backward()

            # Gradient clipping — important for BERT fine-tuning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += len(labels)

        train_acc = correct / total
        train_loss = epoch_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attn_mask = batch["attention_mask"].to(DEVICE)
                labels    = batch["labels"].to(DEVICE)
                out = model(input_ids, attn_mask)
                correct += (out.argmax(1) == labels).sum().item()
                total   += len(labels)

        test_acc = correct / total
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_loss"].append(train_loss)

        print(f"  Epoch {epoch}/{N_EPOCHS} | Loss: {train_loss:.4f} | Train: {train_acc:.1%} | Test: {test_acc:.1%}")

    return model, history


def visualize_tokenization(tokenizer):
    """Show how BERT tokenizes text — WordPiece subwords."""
    print("\n── BERT Tokenization ──\n")
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "Transfer learning is revolutionizing NLP.",
        "Unbelievable performance by the entire cast!",
    ]

    for text in examples:
        tokens = tokenizer.tokenize(text)
        ids    = tokenizer.encode(text, add_special_tokens=True)
        print(f"  Text: '{text}'")
        print(f"  Tokens: {tokens}")
        print(f"  IDs:    {ids}")
        print(f"  [CLS]={tokenizer.cls_token_id}, [SEP]={tokenizer.sep_token_id}\n")

    print("Note: WordPiece breaks unknown words into subwords")
    print("  'unbelievable' → ['un', '##believe', '##able']")
    print("  This lets BERT handle any word, including typos and rare words.\n")


def show_what_bert_learned(model, tokenizer, test_texts, test_labels):
    """Make predictions on new examples and show confidence."""
    print("\n── BERT Predictions on New Examples ──\n")

    new_examples = [
        ("This film was an absolute masterpiece. I was moved to tears.", 1),
        ("Terrible. Just terrible. I want my money back immediately.", 0),
        ("It was okay. Not great, not bad. Just sort of there.", None),
        ("The director somehow managed to ruin a great original story.", 0),
        ("A delightful surprise that exceeded all of my expectations.", 1),
    ]

    model.eval()
    for text, expected in new_examples:
        enc = tokenizer(text, return_tensors="pt", max_length=128,
                        padding="max_length", truncation=True)
        input_ids = enc["input_ids"].to(DEVICE)
        attn_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attn_mask)
            probs  = torch.softmax(logits, dim=1)[0]

        neg_prob = probs[0].item()
        pos_prob = probs[1].item()
        prediction = "POSITIVE" if pos_prob > neg_prob else "NEGATIVE"
        correct = "✓" if expected is None else ("✓" if (pos_prob > neg_prob) == bool(expected) else "✗")

        print(f"  {correct} [{prediction}] pos={pos_prob:.1%} neg={neg_prob:.1%}")
        print(f"    \"{text[:70]}\"")
        print()


def main():
    print("=== BERT Fine-Tuning for Sentiment Analysis ===")
    print(f"Device: {DEVICE}\n")

    if not HAS_HF:
        print("HuggingFace Transformers not installed.")
        print("Install with: pip install transformers\n")
        print("Showing BERT concepts without running the model...\n")

        print("── What is BERT? ──\n")
        print("BERT = Bidirectional Encoder Representations from Transformers")
        print("Pretrained by Google on:")
        print("  1. Masked LM: predict randomly masked tokens")
        print("     'The [MASK] sat on the mat' → 'cat'")
        print("  2. Next Sentence Prediction: is sentence B the next sentence after A?")
        print("     [CLS] A [SEP] B [SEP] → Yes/No\n")
        print("Result: rich contextual word representations")
        print("  'bank' in 'river bank' ≠ 'bank' in 'savings bank'\n")
        print("Fine-tuning: add a task-specific head on [CLS] token")
        print("Train for 3-4 epochs with lr=2e-5 → state-of-the-art NLP\n")
        return

    print("── What is BERT? ──\n")
    print("Pretrained on BookCorpus + Wikipedia (3.3B words)")
    print("Two pretraining tasks:")
    print("  1. Masked LM (MLM): predict 15% of randomly masked tokens")
    print("  2. Next Sentence Prediction (NSP): sentence relationship")
    print("\nFine-tuning strategy:")
    print("  - Prepend [CLS] token → use its embedding for classification")
    print("  - Train with very small LR (2e-5) for 2-4 epochs")
    print("  - Much faster than training from scratch\n")

    # Load tokenizer and model
    print(f"Loading {MODEL_NAME}...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    bert_model = AutoModel.from_pretrained(MODEL_NAME)
    print("Loaded.\n")

    # Show tokenization
    visualize_tokenization(tokenizer)

    # Dataset
    texts, labels = create_sentiment_dataset()
    np.random.seed(42)
    split = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split], labels[:split]
    test_texts,  test_labels  = texts[split:],  labels[split:]
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}\n")

    # Fine-tune
    model, history = train_bert(tokenizer, bert_model, train_texts, train_labels, test_texts, test_labels)

    # Save
    torch.save(model.state_dict(), "19_bert_sentiment.pt")
    print("\nSaved model to 19_bert_sentiment.pt")

    # Predictions
    show_what_bert_learned(model, tokenizer, test_texts, test_labels)

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train")
    axes[0].plot(epochs, [a * 100 for a in history["test_acc"]],  "o-", label="Test")
    axes[0].set_title("BERT Fine-Tuning Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_loss"], "o-", color="red")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("19_bert_training.png", dpi=100)
    print("Saved training curves to 19_bert_training.png")

    print(f"\n=== Summary ===")
    print(f"Best test accuracy: {max(history['test_acc']):.1%}")
    print(f"\nKey concepts mastered:")
    print("  ✓ BERT pretraining — MLM + NSP")
    print("  ✓ WordPiece tokenization — handles any vocabulary")
    print("  ✓ [CLS] token — aggregate sequence representation")
    print("  ✓ Attention mask — handles variable-length sequences")
    print("  ✓ Fine-tuning LR — always use ~2e-5 for BERT")
    print("  ✓ Warmup scheduler — ramp up LR at start of fine-tuning")
    print("  ✓ Gradient clipping — essential for stable BERT training")
    print("\nNext: try DistilBERT (faster), RoBERTa (better), or GPT-2 for generation")


if __name__ == "__main__":
    main()
