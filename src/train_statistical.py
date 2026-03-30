import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, ParameterGrid, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve
)
from sklearn.preprocessing import StandardScaler
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

    # Check which features are available
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
    best_model = None

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

    # Refit best model on full training data
    best_model = estimator.set_params(**best_params)
    best_model.fit(X, y)

    return best_model, best_params, best_score


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train Logistic Regression as our baseline model.
    Uses StandardScaler because LR is sensitive to feature scales.
    """
    print("\n" + "="*50)
    print("Training Logistic Regression (Baseline)")
    print("="*50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    best_model, best_params, best_score = grid_search_with_progress(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid, X_train_scaled, y_train, "Logistic Regression"
    )

    print(f"\nBest params: {best_params}")
    print(f"Best CV F1: {best_score:.4f}")

    val_preds = best_model.predict(X_val_scaled)
    val_proba = best_model.predict_proba(X_val_scaled)[:, 1]

    metrics = evaluate_model(y_val, val_preds, val_proba, "Logistic Regression")

    return best_model, scaler, metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest classifier.
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

    return best_model, None, metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost classifier — our primary model.
    Typically the strongest performer on tabular data.
    """
    print("\n" + "="*50)
    print("Training XGBoost (Primary Model)")
    print("="*50)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    best_model, best_params, best_score = grid_search_with_progress(
        XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
        param_grid, X_train, y_train, "XGBoost"
    )

    print(f"\nBest params: {best_params}")
    print(f"Best CV F1: {best_score:.4f}")

    val_preds = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    metrics = evaluate_model(y_val, val_preds, val_proba, "XGBoost")

    return best_model, None, metrics


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


def evaluate_on_test(model, X_test, y_test, model_name, scaler=None):
    """Final evaluation on the held-out test set."""
    print(f"\n{'='*50}")
    print(f"FINAL TEST EVALUATION: {model_name}")
    print(f"{'='*50}")

    if scaler:
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_model(y_test, y_pred, y_proba, f"{model_name} (Test)")
    return metrics, y_pred, y_proba


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_model_comparison(all_metrics, save_dir="models"):
    """Bar chart comparing all models across metrics."""
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(all_metrics)
    metrics_to_plot = ["accuracy", "f1", "precision", "recall", "auroc"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=row["model"])

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Validation Set")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    print(f"Saved model comparison chart to {save_dir}/model_comparison.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="models"):
    """Plot confusion matrix for the best model."""
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
    ax.set_title(f"Confusion Matrix — {model_name}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    print(f"Saved confusion matrix to {save_dir}/confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, y_proba, model_name, save_dir="models"):
    """Plot ROC curve for the best model."""
    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=150)
    print(f"Saved ROC curve to {save_dir}/roc_curve.png")
    plt.close()


def plot_feature_importance(model, feature_names, model_name, save_dir="models", top_n=15):
    """Plot top N most important features."""
    os.makedirs(save_dir, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't support feature importance extraction.")
        return

    # Sort and take top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_features)), top_importances[::-1], color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=150)
    print(f"Saved feature importance chart to {save_dir}/feature_importance.png")
    plt.close()


# ──────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────

def save_model(model, scaler, metrics, feature_names, save_dir="models"):
    """Save the best model, scaler, and metadata."""
    os.makedirs(save_dir, exist_ok=True)

    model_name = metrics["model"].lower().replace(" ", "_")

    joblib.dump(model, os.path.join(save_dir, f"{model_name}_model.joblib"))
    print(f"Saved model to {save_dir}/{model_name}_model.joblib")

    if scaler:
        joblib.dump(scaler, os.path.join(save_dir, f"{model_name}_scaler.joblib"))
        print(f"Saved scaler to {save_dir}/{model_name}_scaler.joblib")

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

    # Train all three models
    lr_model, lr_scaler, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    rf_model, _, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_model, _, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)

    # Compare models
    all_metrics = [lr_metrics, rf_metrics, xgb_metrics]
    plot_model_comparison(all_metrics)

    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY (Validation Set)")
    print("="*60)
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df.set_index("model")
    print(comparison_df.round(4).to_string())

    # Pick the best model by F1 score
    best_idx = np.argmax([m["f1"] for m in all_metrics])
    best_metrics = all_metrics[best_idx]
    best_model_name = best_metrics["model"]
    print(f"\n>>> Best model: {best_model_name} (F1 = {best_metrics['f1']:.4f})")

    # Get the actual model object and scaler
    models = {
        "Logistic Regression": (lr_model, lr_scaler),
        "Random Forest": (rf_model, None),
        "XGBoost": (xgb_model, None)
    }
    best_model, best_scaler = models[best_model_name]

    # Final evaluation on test set
    test_metrics, test_preds, test_proba = evaluate_on_test(
        best_model, X_test, y_test, best_model_name, best_scaler
    )

    # Generate all plots for the best model
    plot_confusion_matrix(y_test, test_preds, best_model_name)
    plot_roc_curve(y_test, test_proba, best_model_name)
    plot_feature_importance(best_model, feature_names, best_model_name)

    # Save the best model
    save_model(best_model, best_scaler, test_metrics, feature_names)

    # Also save all three models for comparison
    
    save_model(lr_model, lr_scaler, lr_metrics, feature_names)
    save_model(rf_model, None, rf_metrics, feature_names)
    save_model(xgb_model, None, xgb_metrics, feature_names)

    total_time = time.time() - pipeline_start
    mins, secs = divmod(int(total_time), 60)
    hrs, mins = divmod(mins, 60)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best model: {best_model_name}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Total time: {hrs}h {mins}m {secs}s")
    print(f"All models and plots saved to models/")