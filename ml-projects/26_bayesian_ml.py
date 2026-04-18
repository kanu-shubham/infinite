"""
Project 26: Bayesian ML — Uncertainty-Aware Predictions
=========================================================
Standard ML gives a single prediction. Bayesian ML gives a prediction
WITH confidence: "I'm 90% sure" vs "I have no idea".

This matters for medical diagnosis, autonomous driving, and finance —
anywhere the cost of a wrong confident prediction is catastrophic.

What you'll learn:
- Bayesian inference: update beliefs with data (Bayes' theorem)
- Gaussian Processes: the non-parametric Bayesian regression method
- Bayesian Neural Networks: uncertainty via dropout at inference
- Bayesian hyperparameter optimization with Optuna
- Calibration: is your model's confidence actually accurate?
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Part 1: Bayes' Theorem ────────────────────────────────────────────────

def explain_bayes():
    print("── Bayes' Theorem ──\n")
    print("""
  P(θ | data) = P(data | θ) × P(θ) / P(data)
                ─────────────────────────────
  Posterior    = Likelihood  × Prior / Evidence

  In plain English:
    Prior:      what we believe BEFORE seeing data
    Likelihood: how probable is the data given these parameters?
    Posterior:  updated belief AFTER seeing data

  Example — Medical test:
    Disease prevalence (prior):      P(disease) = 0.01 (1% of population)
    Test sensitivity (likelihood):   P(positive | disease) = 0.95
    Test specificity:                P(negative | healthy) = 0.90

    If you test positive:
    P(disease | positive) = 0.95×0.01 / (0.95×0.01 + 0.10×0.99) = ~8.7%

    Even with a 95% accurate test, a positive result only means 8.7% chance!
    This is why Bayesian thinking is critical in medicine.
    """)


# ── Part 2: Gaussian Process Regression ───────────────────────────────────

def demo_gaussian_process():
    """
    Gaussian Process (GP): instead of fitting a single function,
    it maintains a distribution over all possible functions.

    At each test point, GP gives:
    - Mean prediction (like regular regression)
    - Uncertainty estimate (confidence interval)

    The uncertainty is:
    - Small where we have data (model is confident)
    - Large in regions with no data (model is uncertain)
    """
    print("\n── Gaussian Process Regression ──\n")

    np.random.seed(42)

    # Training data (sparse — gaps intentional to show uncertainty)
    X_train = np.array([-4, -3, -1, 0, 1, 3, 4]).reshape(-1, 1)
    y_train = np.sin(X_train.ravel()) + np.random.randn(len(X_train)) * 0.1

    # Test points (dense grid)
    X_test = np.linspace(-6, 6, 300).reshape(-1, 1)
    y_true = np.sin(X_test.ravel())

    # Different kernels
    kernels = {
        "RBF (smooth)":    ConstantKernel(1.0) * RBF(length_scale=1.0),
        "Matern (rough)":  ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (kernel_name, kernel) in zip(axes, kernels.items()):
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=5)
        gp.fit(X_train, y_train)

        y_mean, y_std = gp.predict(X_test, return_std=True)

        ax.plot(X_test, y_true, "k--", linewidth=1, label="True function", alpha=0.5)
        ax.plot(X_test, y_mean, "b-", linewidth=2, label="GP mean")
        ax.fill_between(X_test.ravel(),
                        y_mean - 2*y_std, y_mean + 2*y_std,
                        alpha=0.3, color="blue", label="95% confidence")
        ax.scatter(X_train, y_train, c="red", zorder=5, s=60, label="Training data")

        ax.set_title(f"Gaussian Process — {kernel_name}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2.5, 2.5)

        # Annotate uncertainty regions
        ax.annotate("Uncertain here\n(no data)", xy=(-5, 0),
                    fontsize=8, color="red", ha="center")

    plt.suptitle("Gaussian Process: Uncertainty Grows Away from Data", fontsize=13)
    plt.tight_layout()
    plt.savefig("26_gaussian_process.png", dpi=100)
    print("Saved GP plot to 26_gaussian_process.png")
    print(f"Optimized kernel: {gp.kernel_}")


# ── Part 3: Monte Carlo Dropout (Bayesian NN approximation) ──────────────

class BayesianNN(nn.Module):
    """
    Bayesian Neural Network via Monte Carlo Dropout (Gal & Ghahramani, 2016).

    Key insight: keeping dropout ACTIVE at inference time and running
    multiple forward passes gives samples from an approximate posterior.

    Run 50 forward passes → 50 different predictions → take mean + std.
    This gives uncertainty estimates without changing the training procedure!
    """
    def __init__(self, in_features, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),      # kept active at inference
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

    def mc_predict(self, x, n_samples=100):
        """
        Monte Carlo inference: run n_samples forward passes with dropout active.
        Returns mean and std of predictions = uncertainty estimate.
        """
        self.train()   # keep dropout active!
        preds = torch.stack([self(x) for _ in range(n_samples)], dim=0)
        return preds.mean(0), preds.std(0)


def demo_bayesian_nn():
    print("\n── Monte Carlo Dropout (Bayesian NN) ──\n")

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(DEVICE)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32).to(DEVICE)

    model = BayesianNN(X_train_s.shape[1], dropout_p=0.3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr_t)
        loss = criterion(out, y_tr_t)
        loss.backward()
        optimizer.step()

    # MC inference
    mean_preds, std_preds = model.mc_predict(X_te_t, n_samples=100)
    mean_preds = mean_preds.cpu().detach().numpy().ravel()
    std_preds  = std_preds.cpu().detach().numpy().ravel()

    # Sort by uncertainty
    sorted_idx = std_preds.argsort()
    acc = ((mean_preds > 0.5).astype(int) == y_test).mean()
    print(f"Test accuracy: {acc:.1%}")
    print(f"Mean uncertainty (std): {std_preds.mean():.4f}")
    print(f"Most certain predictions (std={std_preds[sorted_idx[:5]].mean():.4f}): "
          f"acc={((mean_preds[sorted_idx[:10]] > 0.5).astype(int) == y_test[sorted_idx[:10]]).mean():.1%}")
    print(f"Least certain predictions (std={std_preds[sorted_idx[-5:]].mean():.4f}): "
          f"acc={((mean_preds[sorted_idx[-10:]] > 0.5).astype(int) == y_test[sorted_idx[-10:]]).mean():.1%}")

    # Plot uncertainty
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(mean_preds, std_preds, c=y_test, cmap="RdBu", alpha=0.6, s=40)
    axes[0].axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Mean Prediction (probability)")
    axes[0].set_ylabel("Uncertainty (std across MC samples)")
    axes[0].set_title("Prediction vs Uncertainty\n(high uncertainty near decision boundary)")
    axes[0].grid(True, alpha=0.3)

    # Calibration
    axes[1].hist(std_preds[y_test == 0], bins=20, alpha=0.6, label="Malignant", density=True)
    axes[1].hist(std_preds[y_test == 1], bins=20, alpha=0.6, label="Benign",    density=True)
    axes[1].set_xlabel("Prediction Uncertainty (std)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Uncertainty by True Class")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("26_bayesian_nn_uncertainty.png", dpi=100)
    print("Saved uncertainty plot to 26_bayesian_nn_uncertainty.png")

    return model


# ── Part 4: Bayesian Hyperparameter Optimization with Optuna ─────────────

def demo_optuna():
    print("\n── Bayesian Hyperparameter Optimization (Optuna) ──\n")

    if not HAS_OPTUNA:
        print("Optuna not installed. Install with: pip install optuna")
        print("Optuna uses Tree-Structured Parzen Estimator (TPE) —")
        print("a Bayesian method that learns which hyperparameters work")
        print("best from previous trials, rather than random search.\n")
        print("Example usage:")
        print("""
  import optuna

  def objective(trial):
      lr  = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
      n_layers = trial.suggest_int("n_layers", 1, 5)
      dropout  = trial.suggest_float("dropout", 0.1, 0.5)
      # ... build and train model with these params
      return validation_loss

  study = optuna.create_study(direction="minimize")
  study.optimize(objective, n_trials=100)
  print(study.best_params)
        """)
        return

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    def objective(trial):
        C       = trial.suggest_float("C", 1e-3, 100, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver  = "liblinear" if penalty == "l1" else "lbfgs"
        model   = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
        model.fit(X_train_s, y_train)
        return -model.score(X_test_s, y_test)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    print(f"Best params: {study.best_params}")
    print(f"Best accuracy: {-study.best_value:.2%}")

    # Optimization history
    trials_df = study.trials_dataframe()
    plt.figure(figsize=(10, 4))
    plt.plot(-trials_df["value"], "o-", markersize=4, alpha=0.6)
    plt.xlabel("Trial #")
    plt.ylabel("Accuracy")
    plt.title("Optuna Bayesian Hyperparameter Search\n"
              "(each trial learns from previous ones — smarter than grid search)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("26_optuna_search.png", dpi=100)
    print("Saved Optuna history to 26_optuna_search.png")


def main():
    print("=== Bayesian ML: Uncertainty-Aware Predictions ===\n")

    explain_bayes()
    demo_gaussian_process()
    demo_bayesian_nn()
    demo_optuna()

    print("\n=== Summary ===")
    print("Key concepts mastered:")
    print("  ✓ Bayes' theorem — prior → posterior with data")
    print("  ✓ Gaussian Process — distribution over functions, uncertainty bands")
    print("  ✓ GP kernels — RBF (smooth) vs Matern (rough)")
    print("  ✓ MC Dropout — uncertainty via multiple forward passes")
    print("  ✓ Uncertainty = width of confidence interval")
    print("  ✓ Bayesian HP tuning (Optuna) — smarter than grid search")
    print("\nRule: use Bayesian uncertainty when the cost of a wrong")
    print("confident prediction is HIGH (medicine, finance, safety).")


if __name__ == "__main__":
    main()
