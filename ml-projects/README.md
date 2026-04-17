# Machine Learning Projects

19 hands-on projects from beginner to deep learning. Each is a self-contained Python script — loads data, trains models, evaluates results, and saves visualizations.

## Setup

```bash
cd ml-projects
pip install -r requirements.txt
```

---

## Beginner Projects (01–07, 21–23)

| # | Project | ML Type | Key Concepts |
|---|---------|---------|--------------|
| 1 | [Iris Classification](01_iris_classification.py) | Classification | KNN, train/test split, confusion matrix |
| 2 | [House Price Prediction](02_house_price_prediction.py) | Regression | Linear Regression, Random Forest, MAE/RMSE/R² |
| 3 | [Titanic Survival](03_titanic_survival.py) | Classification | Missing data, feature engineering, Logistic Regression |
| 4 | [MNIST Digits](04_mnist_digit_recognition.py) | Deep Learning | Neural networks, TensorFlow/Keras, image data |
| 5 | [Spam Classifier](05_spam_classifier.py) | NLP | TF-IDF, Naive Bayes, text pipelines |
| 6 | [Sentiment Analysis](06_sentiment_analysis.py) | NLP | Bag-of-Words, SVM, model comparison |
| 7 | [Customer Segmentation](07_customer_segmentation.py) | Clustering | K-Means, Elbow Method, PCA |
| 21 | [SVM](21_svm.py) | Classification | Margin, support vectors, kernel trick (RBF/Poly/Linear), C parameter |
| 22 | [PCA & Dimensionality Reduction](22_pca_dimensionality_reduction.py) | Unsupervised | Explained variance, scree plot, compression, t-SNE |
| 23 | [Neural Network from Scratch](23_neural_network_from_scratch.py) | Deep Learning | Backprop, chain rule, gradient checking — NumPy only |

## Intermediate Projects (08–12)

| # | Project | ML Type | Key Concepts |
|---|---------|---------|--------------|
| 8 | [Time Series Forecasting](08_time_series_forecasting.py) | Regression | Lag features, rolling stats, temporal train/test split |
| 9 | [Fraud Detection](09_fraud_detection.py) | Classification | Imbalanced data, ROC/PR curves, threshold tuning |
| 10 | [Ensemble Methods](10_ensemble_methods.py) | Regression | XGBoost, LightGBM, hyperparameter tuning, learning curves |
| 11 | [Feature Engineering](11_feature_engineering.py) | Regression | Encoding, interactions, selection, step-by-step impact |
| 12 | [MLOps Basics](12_mlops_basics.py) | MLOps | Model save/load, experiment tracking, reproducibility |

## Deep Learning Projects (13–20)

| # | Project | Framework | Key Concepts |
|---|---------|-----------|--------------|
| 13 | [PyTorch Fundamentals](13_pytorch_fundamentals.py) | PyTorch | Tensors, autograd, nn.Module, training loop |
| 14 | [CNN Image Classification](14_cnn_image_classification.py) | PyTorch | Conv2d, BatchNorm, MaxPool, data augmentation, CIFAR-10 |
| 15 | [Object Detection](15_object_detection.py) | PyTorch + torchvision | IoU, NMS, Faster R-CNN, bounding boxes |
| 16 | [Image Segmentation](16_image_segmentation.py) | PyTorch | U-Net, skip connections, per-pixel prediction, DeepLabV3 |
| 17 | [Transformers & Attention](17_transformers_attention.py) | PyTorch | Self-attention, multi-head attention, positional encoding |
| 18 | [Transfer Learning ResNet](18_transfer_learning_resnet.py) | PyTorch | ResNet, feature extraction vs fine-tuning, residual blocks |
| 19 | [BERT Fine-Tuning](19_bert_fine_tuning.py) | HuggingFace | BERT, WordPiece tokenization, [CLS] token, warmup LR |
| 20 | [Cross-Attention](20_cross_attention.py) | PyTorch | Cross-attention, encoder-decoder, seq2seq, causal mask, teacher forcing |

---

## Running a Project

```bash
python 13_pytorch_fundamentals.py
```

Each script prints results to the terminal and saves `.png` visualizations in the current directory.

---

## Recommended Learning Order

**Beginner (start here):**
1 → 2 → 3 → 4 → 5 → 6 → 7

**Intermediate (after beginner):**
8 → 9 → 10 → 11 → 12

**Deep Learning (after intermediate):**
13 → 14 → 15 → 16 → 17 → 18 → 19 → 20

---

## What Each Deep Learning Project Teaches

### Project 13: PyTorch Fundamentals
- Tensors — shape, dtype, device, operations
- Autograd — `loss.backward()`, `.grad`, `zero_grad()`
- `nn.Module` — building networks as Python classes
- `Dataset` / `DataLoader` — batching, shuffling, iteration
- The 5-step training loop: `zero_grad → forward → loss → backward → step`
- Save/load with `state_dict` (correct way, not `torch.save(model)`)

### Project 14: CNN Image Classification (CIFAR-10)
- Why CNNs beat FC nets on images (spatial structure)
- `Conv2d` — learns spatial filters (edges, textures, shapes)
- `BatchNorm2d` — stabilizes training, allows higher LR
- `MaxPool2d` — downsamples, adds translation invariance
- `AdaptiveAvgPool2d` — replaces giant FC layers
- Data augmentation — random flip, crop, color jitter
- Cosine annealing LR scheduler
- Visualizing learned convolutional filters

### Project 15: Object Detection
- Classification vs Detection vs Segmentation
- Bounding boxes — `[x_min, y_min, x_max, y_max]` format
- IoU (Intersection over Union) — the detection metric
- NMS (Non-Maximum Suppression) — removing duplicate boxes
- Faster R-CNN — backbone + RPN + RoI head (two-stage)
- Using COCO-pretrained models for zero-shot detection

### Project 16: Image Segmentation
- Semantic segmentation — label every single pixel
- U-Net architecture — encoder-decoder with skip connections
- Skip connections — copy encoder features to decoder
- Why skip connections preserve fine spatial detail
- Per-pixel cross-entropy loss
- mIoU — standard segmentation evaluation metric
- DeepLabV3 pretrained for real-world segmentation

### Project 17: Transformers & Attention from Scratch
- Self-attention formula: `Attention(Q,K,V) = softmax(QK.T/√dk) V`
- Q/K/V projections — query, key, value roles explained
- Multi-head attention — parallel heads learn different relationships
- Positional encoding — sin/cos waves inject sequence order
- Transformer block — attention + FFN + residuals + LayerNorm
- Gradient clipping — critical for stable Transformer training
- Full sequence classifier using Transformer blocks

### Project 18: Transfer Learning with ResNet
- ResNet residual connections — solve vanishing gradients
- ImageNet pretraining — 1.28M images, 1000 classes
- Feature extraction — freeze backbone, train only head (fast)
- Fine-tuning — unfreeze all with small LR (more accurate)
- From scratch vs transfer learning comparison
- Visualizing shallow vs deep feature maps
- When to use each strategy

### Project 19: BERT Fine-Tuning
- BERT pretraining — Masked LM + Next Sentence Prediction
- WordPiece tokenization — handles any word, typos, rare words
- `[CLS]` token — aggregate sentence representation
- Attention mask — handles variable-length sequences
- Fine-tuning LR — always ~2e-5 for BERT (very important)
- Linear warmup scheduler — ramp up LR at start
- HuggingFace `AutoTokenizer` + `AutoModel` ecosystem
