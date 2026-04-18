"""
Project 5: Spam Email Classifier
==================================
Classify emails/messages as spam or not-spam (ham) using text features.

What you'll learn:
- Text preprocessing (tokenization, TF-IDF)
- Naive Bayes classifier (great for text)
- Working with text data pipelines
- Precision vs. Recall tradeoff
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_curve, roc_curve, auc,
)


def create_spam_dataset():
    """Create a synthetic spam dataset for demonstration."""
    np.random.seed(42)

    ham_templates = [
        "Hey, are you coming to the meeting tomorrow?",
        "Can you send me the project report by Friday?",
        "Let's grab lunch together this week.",
        "The quarterly results look promising this year.",
        "I've updated the shared document with new data.",
        "Don't forget about mom's birthday next week.",
        "Thanks for helping me with the presentation.",
        "The team dinner is scheduled for Saturday evening.",
        "Could you review my pull request when you get a chance?",
        "Great job on the client demo yesterday!",
        "I'll be working from home tomorrow.",
        "Please find the attached invoice for the project.",
        "The kids have a school play next Thursday.",
        "Can we reschedule our one-on-one to Wednesday?",
        "Just finished reading that book you recommended.",
        "The new software update fixed the bug we reported.",
        "Happy anniversary! Hope you have a wonderful day.",
        "Reminder: dentist appointment at 3pm today.",
        "The flight is confirmed for next Monday morning.",
        "I love the new design mockups you shared.",
    ]

    spam_templates = [
        "CONGRATULATIONS! You've won a $1000 gift card! Click here NOW!",
        "FREE VIAGRA! Best prices online! Order today!",
        "You are selected for a CASH PRIZE of $5000! Claim now!",
        "Make money fast! Work from home and earn $500/day!",
        "URGENT: Your account has been compromised. Click to verify.",
        "Hot singles in your area want to meet you tonight!",
        "LIMITED TIME OFFER: 90% OFF designer watches!",
        "You've been selected for a FREE iPhone! Reply WIN to claim!",
        "Lose 30 pounds in 30 days! Miracle weight loss pill!",
        "WINNER!! You have won the international lottery! Send details.",
        "Cheap medications delivered to your door! No prescription needed!",
        "Your PayPal account needs immediate verification! Click here!",
        "Earn $1000/week from home! No experience needed!",
        "FREE credit score check! Limited spots available! Act fast!",
        "Exclusive deal just for you! Buy one get ten FREE!",
        "ALERT: Suspicious activity detected on your bank account!",
        "Double your investment in 24 hours! Guaranteed returns!",
        "You qualify for a government grant of $25000! Apply now!",
        "Secret method to make money online revealed! Click here!",
        "Amazing deal! Luxury items at 95% discount! Ships FREE!",
    ]

    # Generate variations
    messages, labels = [], []
    for _ in range(250):
        template = np.random.choice(ham_templates)
        words = template.split()
        if len(words) > 3 and np.random.random() > 0.5:
            idx = np.random.randint(1, len(words) - 1)
            filler = np.random.choice(["really", "actually", "definitely", "perhaps", "probably"])
            words.insert(idx, filler)
        messages.append(" ".join(words))
        labels.append(0)

    for _ in range(150):
        template = np.random.choice(spam_templates)
        words = template.split()
        if np.random.random() > 0.5:
            words = [w.upper() if np.random.random() > 0.7 else w for w in words]
        messages.append(" ".join(words))
        labels.append(1)

    # Shuffle
    indices = np.random.permutation(len(messages))
    messages = [messages[i] for i in indices]
    labels = [labels[i] for i in indices]

    return pd.DataFrame({"message": messages, "label": labels})


def main():
    # ── 1. Load the dataset ──────────────────────────────────────────────
    df = create_spam_dataset()

    print("=== Spam Email Classifier ===")
    print(f"Total messages: {len(df)}")
    print(f"Ham (not spam): {(df['label'] == 0).sum()}")
    print(f"Spam:           {(df['label'] == 1).sum()}")
    print(f"Spam ratio:     {df['label'].mean():.1%}\n")

    print("Sample ham messages:")
    for msg in df[df["label"] == 0]["message"].head(3):
        print(f"  - {msg}")
    print("\nSample spam messages:")
    for msg in df[df["label"] == 1]["message"].head(3):
        print(f"  - {msg}")
    print()

    # ── 2. Text analysis ────────────────────────────────────────────────
    df["msg_length"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()
    df["caps_ratio"] = df["message"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df["exclamation_count"] = df["message"].str.count("!")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, feat in enumerate(["msg_length", "word_count", "caps_ratio", "exclamation_count"]):
        ax = axes[idx // 2, idx % 2]
        df[df["label"] == 0][feat].hist(ax=ax, bins=20, alpha=0.7, label="Ham", color="green")
        df[df["label"] == 1][feat].hist(ax=ax, bins=20, alpha=0.7, label="Spam", color="red")
        ax.set_title(feat.replace("_", " ").title())
        ax.legend()

    plt.suptitle("Text Feature Distributions: Ham vs Spam", fontsize=14)
    plt.tight_layout()
    plt.savefig("05_spam_text_analysis.png", dpi=100)
    print("Saved text analysis to 05_spam_text_analysis.png")

    # ── 3. Build pipelines ───────────────────────────────────────────────
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    # Pipeline: TF-IDF + Classifier
    pipelines = {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    # ── 4. Train and evaluate ────────────────────────────────────────────
    print("=== Model Comparison ===\n")
    best_name, best_acc = None, 0

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

        print(f"{name}:")
        print(f"  Test Accuracy:    {acc:.2%}")
        print(f"  CV Mean Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
        print()

        if acc > best_acc:
            best_name, best_acc = name, acc

    # ── 5. Detailed results for best model ───────────────────────────────
    best_pipeline = pipelines[best_name]
    y_pred = best_pipeline.predict(X_test)

    print(f"=== {best_name} Detailed Report ===")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"],
                yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{best_name} Confusion Matrix ({best_acc:.2%})")
    plt.tight_layout()
    plt.savefig("05_spam_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 05_spam_confusion_matrix.png")

    # ── 6. Show top predictive words ─────────────────────────────────────
    tfidf = best_pipeline.named_steps["tfidf"]
    feature_names = tfidf.get_feature_names_out()

    if best_name == "Naive Bayes":
        clf = best_pipeline.named_steps["clf"]
        log_probs = clf.feature_log_prob_
        spam_words_idx = np.argsort(log_probs[1] - log_probs[0])[-15:]
        ham_words_idx = np.argsort(log_probs[0] - log_probs[1])[-15:]
    else:
        clf = best_pipeline.named_steps["clf"]
        coefs = clf.coef_[0]
        spam_words_idx = np.argsort(coefs)[-15:]
        ham_words_idx = np.argsort(coefs)[:15]

    print("\nTop 15 words indicating SPAM:")
    for idx in reversed(spam_words_idx):
        print(f"  - {feature_names[idx]}")

    print("\nTop 15 words indicating HAM:")
    for idx in reversed(ham_words_idx):
        print(f"  - {feature_names[idx]}")


if __name__ == "__main__":
    main()
