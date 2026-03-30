import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ParameterGrid, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Feature columns (must match feature_engineering.py)
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "word_count", "char_count", "sentence_count",
    "avg_word_length", "avg_sentence_length", "sentence_length_variance",
    "type_token_ratio", "hapax_legomena_ratio", "lexical_density",
    "punctuation_ratio", "comma_ratio", "exclamation_rate",
    "question_rate", "starts_with_capital_ratio", "digit_ratio",
    "paragraph_count", "avg_paragraph_length", "stopword_ratio",
    "repetition_rate", "perplexity", "burstiness"
]


def load_data(data_dir="data/processed"):
    """Load feature-engineered train/val/test sets."""
    train = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val_features.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test_features.csv"))

    available_features = [f for f in FEATURE_COLS if f in train.columns]
    missing = [f for f in FEATURE_COLS if f not in train.columns]

    if missing:
        print(f"Warning: Missing features (skipped): {missing}")

    print(f"Using {len(available_features)} features: {available_features}")

    X_train = train[available_features].fillna(0)
    y_train = train["label"]
    X_val = val[available_features].fillna(0)
    y_val = val["label"]
    X_test = test[available_features].fillna(0)
    y_test = test["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test, available_features


def grid_search_with_progress(estimator, param_grid, X, y, model_name, cv=5, scoring="f1"):
    """Run grid search with a tqdm progress bar showing ETA."""
    params_list = list(ParameterGrid(param_grid))
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    best_score = -1
    best_params = None

    pbar = tqdm(params_list, desc=f"{model_name}", unit="combo",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for params in pbar:
        model = estimator.set_params(**params)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            pbar.set_postfix({"best_f1": f"{best_score:.4f}"})

    best_model = estimator.set_params(**best_params)
    best_model.fit(X, y)

    return best_model, best_params, best_score


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier with grid search.
    No scaling needed — tree-based models are scale-invariant.
    """
    print("\n" + "="*50)
    print("Training Random Forest")
    print("="*50)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    best_model, best_params, best_score = grid_search_with_progress(
        RandomForestClassifier(random_state=42),
        param_grid, X_train, y_train, "Random Forest"
    )

    print(f"\nBest params: {best_params}")
    print(f"Best CV F1: {best_score:.4f}")

    val_preds = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    metrics = evaluate_model(y_val, val_preds, val_proba, "Random Forest")

    return best_model, metrics


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────

def evaluate_model(y_true, y_pred, y_proba, model_name):
    """Compute and print all evaluation metrics."""
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_proba)
    }

    print(f"\n--- {model_name} Validation Results ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=['Human', 'AI'])}")

    return metrics


def evaluate_on_test(model, X_test, y_test):
    """Final evaluation on the held-out test set."""
    print(f"\n{'='*50}")
    print("FINAL TEST EVALUATION: Random Forest")
    print(f"{'='*50}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba, "Random Forest (Test)")
    return metrics, y_pred, y_proba


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_dir="models"):
    """Plot confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Human", "AI"],
        yticklabels=["Human", "AI"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Random Forest")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "random_forest_confusion_matrix.png"), dpi=150)
    print(f"Saved confusion matrix to {save_dir}/random_forest_confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, y_proba, save_dir="models"):
    """Plot ROC curve."""
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"Random Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "random_forest_roc_curve.png"), dpi=150)
    print(f"Saved ROC curve to {save_dir}/random_forest_roc_curve.png")
    plt.close()


def plot_feature_importance(model, feature_names, save_dir="models", top_n=15):
    """Plot top N most important features."""
    os.makedirs(save_dir, exist_ok=True)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_importances[::-1], color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "random_forest_feature_importance.png"), dpi=150)
    print(f"Saved feature importance chart to {save_dir}/random_forest_feature_importance.png")
    plt.close()


# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────

def save_model(model, metrics, feature_names, save_dir="models"):
    """Save the Random Forest model and metadata."""
    os.makedirs(save_dir, exist_ok=True)

    model_name = metrics["model"].lower().replace(" ", "_")

    joblib.dump(model, os.path.join(save_dir, f"{model_name}_model.joblib"))
    print(f"Saved model to {save_dir}/{model_name}_model.joblib")

    metadata = {
        "model_name": metrics["model"],
        "features": feature_names,
        "metrics": metrics
    }
    with open(os.path.join(save_dir, f"{model_name}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {save_dir}/{model_name}_metadata.json")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

if __name__ == "__main__":
    pipeline_start = time.time()

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data()

    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Features: {len(feature_names)}")

    # Train Random Forest
    rf_model, val_metrics = train_random_forest(X_train, y_train, X_val, y_val)

    # Final evaluation on test set
    test_metrics, test_preds, test_proba = evaluate_on_test(rf_model, X_test, y_test)

    # Generate plots
    plot_confusion_matrix(y_test, test_preds)
    plot_roc_curve(y_test, test_proba)
    plot_feature_importance(rf_model, feature_names)

    # Save model
    save_model(rf_model, test_metrics, feature_names)

    total_time = time.time() - pipeline_start
    mins, secs = divmod(int(total_time), 60)
    hrs, mins = divmod(mins, 60)

    print("\n" + "="*50)
    print("RANDOM FOREST TRAINING COMPLETE")
    print("="*50)
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Total time: {hrs}h {mins}m {secs}s")
    print(f"Model and plots saved to models/")
