"""
Project 6: Movie Review Sentiment Analysis
============================================
Classify movie reviews as positive or negative based on their text.

What you'll learn:
- Natural Language Processing (NLP) basics
- Bag-of-Words and TF-IDF representations
- Multiple classifiers for text
- Analyzing model predictions and errors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_movie_review_dataset():
    """Create a synthetic movie review dataset."""
    np.random.seed(42)

    positive_reviews = [
        "This movie was absolutely fantastic! Great acting and storytelling.",
        "A masterpiece of cinema. Every scene was beautifully crafted.",
        "I loved every minute of this film. Highly recommend it!",
        "Brilliant performances by the entire cast. A must-watch movie.",
        "One of the best movies I've seen this year. Outstanding direction.",
        "The plot twists were unexpected and kept me engaged throughout.",
        "A heartwarming story with excellent character development.",
        "Visually stunning with a powerful emotional impact.",
        "The soundtrack perfectly complemented the incredible visuals.",
        "An inspiring film that will stay with you long after watching.",
        "Superb writing and flawless execution. A true gem.",
        "This film exceeded all my expectations. Pure cinematic gold.",
        "The chemistry between the leads was electric and believable.",
        "A perfect blend of humor, drama, and action sequences.",
        "Gripping from start to finish. I couldn't look away.",
        "The director has outdone themselves with this beautiful film.",
        "Wonderful performances that bring the story to life perfectly.",
        "A delightful experience that appeals to all audiences.",
        "The special effects were amazing and served the story well.",
        "A triumphant achievement in modern filmmaking. Five stars.",
        "Incredibly moving and well-paced throughout the entire runtime.",
        "The dialogue was sharp, witty, and emotionally resonant.",
        "A joyful celebration of storytelling at its finest.",
        "This movie deserves every award it receives. Truly excellent.",
        "Fresh, original, and thoroughly entertaining from beginning to end.",
    ]

    negative_reviews = [
        "This movie was terrible. Complete waste of time and money.",
        "Awful acting and a nonsensical plot. Do not watch this.",
        "I walked out of the theater halfway through. So boring.",
        "The worst movie I've seen in years. Zero redeeming qualities.",
        "Predictable plot with flat, one-dimensional characters throughout.",
        "A painful experience. The dialogue was cringe-worthy and forced.",
        "Horrible pacing and confusing storyline. Totally disappointing.",
        "The special effects couldn't save this disaster of a script.",
        "An absolute mess from beginning to end. Save your money.",
        "Dull, uninspired, and a complete waste of talented actors.",
        "This sequel ruined everything good about the original film.",
        "The movie dragged on forever without any meaningful development.",
        "Poorly written characters that I couldn't care about at all.",
        "A forgettable film that adds nothing new to the genre.",
        "The trailer was better than the actual movie unfortunately.",
        "Overrated and overhyped. I expected so much more from this.",
        "The plot holes were so large you could drive a truck through.",
        "Lazy writing and poor direction make this unwatchable garbage.",
        "I've never been so bored watching a movie in my life.",
        "A disappointing failure that wastes its promising premise entirely.",
        "Terrible pacing makes two hours feel like five painful hours.",
        "The script feels like a rough first draft that needed work.",
        "Not even the star-studded cast could save this sinking ship.",
        "A tedious and frustrating viewing experience I wouldn't repeat.",
        "Uninspired direction and lackluster performances all around sadly.",
    ]

    reviews, labels = [], []
    for _ in range(200):
        template = np.random.choice(positive_reviews)
        adverbs = ["really", "truly", "absolutely", "definitely", "genuinely"]
        if np.random.random() > 0.5:
            words = template.split()
            idx = np.random.randint(0, max(1, len(words) - 1))
            words.insert(idx, np.random.choice(adverbs))
            template = " ".join(words)
        reviews.append(template)
        labels.append(1)

    for _ in range(200):
        template = np.random.choice(negative_reviews)
        adverbs = ["very", "extremely", "incredibly", "utterly", "completely"]
        if np.random.random() > 0.5:
            words = template.split()
            idx = np.random.randint(0, max(1, len(words) - 1))
            words.insert(idx, np.random.choice(adverbs))
            template = " ".join(words)
        reviews.append(template)
        labels.append(0)

    indices = np.random.permutation(len(reviews))
    reviews = [reviews[i] for i in indices]
    labels = [labels[i] for i in indices]

    return pd.DataFrame({"review": reviews, "sentiment": labels})


def main():
    # ── 1. Load the dataset ──────────────────────────────────────────────
    df = create_movie_review_dataset()

    print("=== Movie Sentiment Analysis ===")
    print(f"Total reviews:    {len(df)}")
    print(f"Positive reviews: {(df['sentiment'] == 1).sum()}")
    print(f"Negative reviews: {(df['sentiment'] == 0).sum()}\n")

    print("Sample positive review:")
    print(f"  \"{df[df['sentiment'] == 1]['review'].iloc[0]}\"\n")
    print("Sample negative review:")
    print(f"  \"{df[df['sentiment'] == 0]['review'].iloc[0]}\"\n")

    # ── 2. Text statistics ───────────────────────────────────────────────
    df["word_count"] = df["review"].str.split().str.len()
    df["char_count"] = df["review"].str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for sentiment, label, color in [(1, "Positive", "green"), (0, "Negative", "red")]:
        subset = df[df["sentiment"] == sentiment]
        axes[0].hist(subset["word_count"], bins=15, alpha=0.7, label=label, color=color)
        axes[1].hist(subset["char_count"], bins=15, alpha=0.7, label=label, color=color)

    axes[0].set_title("Word Count Distribution")
    axes[0].set_xlabel("Words")
    axes[0].legend()
    axes[1].set_title("Character Count Distribution")
    axes[1].set_xlabel("Characters")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("06_sentiment_distributions.png", dpi=100)
    print("Saved distributions to 06_sentiment_distributions.png")

    # ── 3. Feature extraction comparison ─────────────────────────────────
    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}\n")

    vectorizers = {
        "Bag-of-Words": CountVectorizer(max_features=5000, stop_words="english"),
        "TF-IDF": TfidfVectorizer(max_features=5000, stop_words="english"),
    }

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(max_iter=5000, random_state=42),
    }

    # ── 4. Train all combinations ────────────────────────────────────────
    print("=== Model Comparison ===\n")
    print(f"{'Vectorizer':<16} {'Classifier':<22} {'Accuracy':>10} {'CV Mean':>10}")
    print("-" * 62)

    best_combo, best_acc = None, 0
    all_results = {}

    for vec_name, vectorizer in vectorizers.items():
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        for clf_name, clf_template in classifiers.items():
            # Clone the classifier for fresh training
            from sklearn.base import clone
            clf = clone(clf_template)

            clf.fit(X_train_vec, y_train)
            y_pred = clf.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)

            cv_scores = cross_val_score(clf_template, X_train_vec, y_train, cv=5)

            combo_name = f"{vec_name} + {clf_name}"
            all_results[combo_name] = acc
            print(f"{vec_name:<16} {clf_name:<22} {acc:>9.2%} {cv_scores.mean():>9.2%}")

            if acc > best_acc:
                best_combo = combo_name
                best_acc = acc
                best_vectorizer = vectorizer
                best_clf = clf

    print(f"\nBest combination: {best_combo} ({best_acc:.2%})\n")

    # ── 5. Detailed report for best model ────────────────────────────────
    X_test_vec = best_vectorizer.transform(X_test)
    y_pred = best_clf.predict(X_test_vec)

    print(f"=== {best_combo} Detailed Report ===")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{best_combo}\nConfusion Matrix ({best_acc:.2%})")
    plt.tight_layout()
    plt.savefig("06_sentiment_confusion_matrix.png", dpi=100)
    print("Saved confusion matrix to 06_sentiment_confusion_matrix.png")

    # ── 6. Model comparison bar chart ────────────────────────────────────
    plt.figure(figsize=(10, 6))
    names = list(all_results.keys())
    accs = list(all_results.values())
    colors = ["#4CAF50" if a == max(accs) else "#2196F3" for a in accs]

    bars = plt.barh(names, accs, color=colors)
    for bar, acc_val in zip(bars, accs):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{acc_val:.1%}", va="center")
    plt.xlabel("Accuracy")
    plt.title("Model Comparison: Vectorizer + Classifier Combinations")
    plt.xlim(0, 1.15)
    plt.tight_layout()
    plt.savefig("06_sentiment_model_comparison.png", dpi=100)
    print("Saved model comparison to 06_sentiment_model_comparison.png")

    # ── 7. Show misclassified examples ───────────────────────────────────
    misclassified = X_test[y_test != y_pred]
    if len(misclassified) > 0:
        print(f"\n=== Misclassified Examples ({len(misclassified)} total) ===")
        for i, (idx, review) in enumerate(misclassified.head(5).items()):
            actual = "Positive" if y_test.iloc[y_test.index.get_loc(idx)] == 1 else "Negative"
            print(f"\n  [{actual} classified as {'Negative' if actual == 'Positive' else 'Positive'}]")
            print(f"  \"{review[:80]}...\"")


if __name__ == "__main__":
    main()
