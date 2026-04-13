# Machine Learning Projects

12 hands-on projects covering beginner to intermediate ML. Each is a self-contained Python script — loads data, trains models, evaluates results, and saves visualizations.

## Setup

```bash
cd ml-projects
pip install -r requirements.txt
```

## Beginner Projects (01–07)

| # | Project | ML Type | Key Concepts |
|---|---------|---------|--------------|
| 1 | [Iris Classification](01_iris_classification.py) | Classification | KNN, train/test split, confusion matrix |
| 2 | [House Price Prediction](02_house_price_prediction.py) | Regression | Linear Regression, Random Forest, MAE/RMSE/R² |
| 3 | [Titanic Survival](03_titanic_survival.py) | Classification | Missing data, feature engineering, Logistic Regression |
| 4 | [MNIST Digits](04_mnist_digit_recognition.py) | Deep Learning | Neural networks, TensorFlow/Keras, image data |
| 5 | [Spam Classifier](05_spam_classifier.py) | NLP | TF-IDF, Naive Bayes, text pipelines |
| 6 | [Sentiment Analysis](06_sentiment_analysis.py) | NLP | Bag-of-Words, SVM, model comparison |
| 7 | [Customer Segmentation](07_customer_segmentation.py) | Clustering | K-Means, Elbow Method, PCA |

## Intermediate Projects (08–12)

| # | Project | ML Type | Key Concepts |
|---|---------|---------|--------------|
| 8 | [Time Series Forecasting](08_time_series_forecasting.py) | Regression | Lag features, rolling stats, temporal train/test split |
| 9 | [Fraud Detection](09_fraud_detection.py) | Classification | Imbalanced data, SMOTE, ROC/PR curves, threshold tuning |
| 10 | [Ensemble Methods](10_ensemble_methods.py) | Regression | XGBoost, LightGBM, hyperparameter tuning, learning curves |
| 11 | [Feature Engineering](11_feature_engineering.py) | Regression | Encoding, interactions, selection, step-by-step impact |
| 12 | [MLOps Basics](12_mlops_basics.py) | MLOps | Model save/load, experiment tracking, reproducibility |

## Running a Project

```bash
python 08_time_series_forecasting.py
```

Each script prints results to the terminal and saves `.png` visualizations in the current directory.

---

## What Each Intermediate Project Teaches

### Project 8: Time Series Forecasting
- Why you must **never** randomly shuffle time series data
- Lag features (yesterday's temp predicts today's)
- Rolling statistics (7-day mean, 30-day std)
- Cyclical encoding (sin/cos for day-of-year)
- Error analysis by month

### Project 9: Fraud Detection (Imbalanced Data)
- The trap: 98% accuracy that catches zero fraud
- `class_weight="balanced"` — easiest fix
- ROC curve vs Precision-Recall curve (PR is better for rare events)
- Threshold tuning: business impact of catching more fraud vs false alarms
- Measuring savings in dollars, not just accuracy

### Project 10: Ensemble Methods
- Bagging vs Boosting — how they differ conceptually
- XGBoost and LightGBM with early stopping
- `GridSearchCV` for hyperparameter tuning
- Learning curves to detect overfitting/underfitting
- Speed vs accuracy tradeoffs

### Project 11: Feature Engineering
- Raw numeric → +categorical encoding → +transformations → +interactions → +selection
- Target encoding, ordinal encoding, frequency encoding
- Log transform for skewed distributions
- Domain-specific interactions (bath/bed ratio, house age × sqft)
- Measuring MAE improvement at each step

### Project 12: MLOps Basics
- `joblib.dump` / `joblib.load` for model persistence
- Building `sklearn.Pipeline` to bundle scaler + model
- Lightweight experiment tracker (logs params, metrics, artifacts to JSON)
- Data fingerprinting for reproducibility verification
- Finding and loading the best run from experiment history

---

## Recommended Learning Order

**Beginner path:**
1 → 2 → 3 → 4 → 5 → 6 → 7

**Intermediate path (after completing beginner):**
8 → 9 → 10 → 11 → 12
