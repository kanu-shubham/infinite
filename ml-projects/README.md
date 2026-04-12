# Machine Learning Beginner Projects

7 hands-on projects to learn ML from scratch. Each project is a self-contained Python script that loads data, trains models, evaluates results, and saves visualizations.

## Setup

```bash
cd ml-projects
pip install -r requirements.txt
```

## Projects

| # | Project | ML Type | Key Concepts |
|---|---------|---------|--------------|
| 1 | [Iris Classification](01_iris_classification.py) | Supervised (Classification) | KNN, train/test split, confusion matrix |
| 2 | [House Price Prediction](02_house_price_prediction.py) | Supervised (Regression) | Linear Regression, Random Forest, MAE/RMSE/R² |
| 3 | [Titanic Survival](03_titanic_survival.py) | Supervised (Classification) | Missing data, feature engineering, Logistic Regression |
| 4 | [MNIST Digits](04_mnist_digit_recognition.py) | Deep Learning | Neural networks, TensorFlow/Keras, image data |
| 5 | [Spam Classifier](05_spam_classifier.py) | NLP (Classification) | TF-IDF, Naive Bayes, text pipelines |
| 6 | [Sentiment Analysis](06_sentiment_analysis.py) | NLP (Classification) | Bag-of-Words, SVM, model comparison |
| 7 | [Customer Segmentation](07_customer_segmentation.py) | Unsupervised (Clustering) | K-Means, Elbow Method, PCA visualization |

## Running a Project

```bash
python 01_iris_classification.py
```

Each script prints results to the terminal and saves `.png` visualizations in the current directory.

## Recommended Learning Order

1. **Start here** — Project 1 (Iris): simplest dataset, core ML workflow
2. **Regression** — Project 2 (House Prices): predicting numbers instead of categories
3. **Real-world data** — Project 3 (Titanic): missing values, feature engineering
4. **Deep learning** — Project 4 (MNIST): neural networks with TensorFlow
5. **Text data** — Projects 5 & 6 (Spam, Sentiment): NLP fundamentals
6. **Unsupervised** — Project 7 (Segmentation): clustering without labels

## What Each Project Teaches

### Project 1: Iris Classification
- Load and explore a dataset with pandas
- Split data into training and testing sets
- Normalize features with StandardScaler
- Train a K-Nearest Neighbors classifier
- Read a confusion matrix and classification report

### Project 2: House Price Prediction
- Correlation analysis between features
- Linear Regression vs Random Forest comparison
- Regression metrics: MAE, RMSE, R²
- Feature importance analysis

### Project 3: Titanic Survival
- Handle missing data (fill with median/mode)
- Engineer new features (FamilySize, IsAlone, AgeGroup)
- Encode categorical variables (one-hot encoding)
- Cross-validation for robust evaluation

### Project 4: MNIST Digit Recognition
- Work with image data (pixel arrays)
- Build a neural network (Dense layers, Dropout)
- Monitor training with loss curves
- Falls back to scikit-learn if TensorFlow isn't installed

### Project 5: Spam Classifier
- Text preprocessing with TF-IDF vectorization
- Naive Bayes — the classic text classification algorithm
- sklearn Pipeline for clean ML workflows
- Identify the most predictive words for spam/ham

### Project 6: Sentiment Analysis
- Compare Bag-of-Words vs TF-IDF representations
- Benchmark multiple classifiers on the same data
- Analyze misclassified examples to understand errors

### Project 7: Customer Segmentation
- Unsupervised learning — no labels needed
- Elbow Method and Silhouette Score to pick K
- PCA for dimensionality reduction and visualization
- Interpret clusters with business meaning
