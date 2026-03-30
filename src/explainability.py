"""
Phase 5: Explainability
─────────────────────────
• SHAP analysis for statistical models (XGBoost, Random Forest, Logistic Regression)
• Captum (Integrated Gradients) for the DistilBERT transformer
• Generates plots saved to models/explainability/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from captum.attr import LayerIntegratedGradients
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Config
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

SAVE_DIR = "models/explainability"
DATA_DIR = "data/processed"
MODEL_DIR = "models"


# ──────────────────────────────────────────────
# Part A: SHAP for Statistical Models
# ──────────────────────────────────────────────

def load_test_features():
    """Load the test feature set."""
    test = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    available = [f for f in FEATURE_COLS if f in test.columns]
    X_test = test[available].fillna(0)
    y_test = test["label"]
    return X_test, y_test, available


def shap_xgboost(X_test, feature_names):
    """SHAP TreeExplainer for XGBoost (fast, exact)."""
    print("\n[SHAP] XGBoost — TreeExplainer")
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=21)
    plt.title("XGBoost — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_xgboost_bar.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved shap_xgboost_bar.png")
    plt.close("all")

    # Beeswarm plot (shows direction of effect)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                      show=False, max_display=21)
    plt.title("XGBoost — SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_xgboost_beeswarm.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved shap_xgboost_beeswarm.png")
    plt.close("all")

    return shap_values


def shap_random_forest(X_test, feature_names):
    """SHAP TreeExplainer for Random Forest."""
    print("\n[SHAP] Random Forest — TreeExplainer")
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.joblib"))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, RF may return (n, features, 2) or [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_test, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=21)
    plt.title("Random Forest — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_rf_bar.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved shap_rf_bar.png")
    plt.close("all")

    return sv


def shap_logistic_regression(X_test, feature_names):
    """SHAP LinearExplainer for Logistic Regression."""
    print("\n[SHAP] Logistic Regression — LinearExplainer")
    model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_scaler.joblib"))

    X_test_scaled = scaler.transform(X_test)

    explainer = shap.LinearExplainer(model, X_test_scaled)
    shap_values = explainer.shap_values(X_test_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=21)
    plt.title("Logistic Regression — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "shap_lr_bar.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved shap_lr_bar.png")
    plt.close("all")

    return shap_values


def shap_dependence_plots(shap_values, X_test, feature_names, top_n=4):
    """Generate SHAP dependence plots for top features (XGBoost)."""
    print("\n[SHAP] Generating dependence plots for top features...")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, ax in zip(top_indices, axes.flatten()):
        shap.dependence_plot(idx, shap_values, X_test,
                             feature_names=feature_names, ax=ax, show=False)
    plt.suptitle("XGBoost — SHAP Dependence Plots (Top 4 Features)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(SAVE_DIR, "shap_xgb_dependence.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved shap_xgb_dependence.png")
    plt.close("all")


# ──────────────────────────────────────────────
# Part B: Captum for DistilBERT
# ──────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def captum_distilbert(num_samples=10):
    """
    Use Captum's LayerIntegratedGradients on DistilBERT embeddings
    to find which tokens drive the AI vs Human prediction.
    """
    print("\n[Captum] DistilBERT — LayerIntegratedGradients")

    # Captum's Integrated Gradients requires float64 internally,
    # which MPS doesn't support — use CPU (only 10 samples, fast enough)
    device = torch.device("cpu")
    print(f"  Device: {device} (Captum IG requires CPU for float64 support)")

    # Load model & tokenizer
    model_path = os.path.join(MODEL_DIR, "distilbert")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load a few test samples
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text"])
    # Pick balanced samples: half human, half AI
    human_samples = test_df[test_df["label"] == 0].sample(num_samples // 2, random_state=42)
    ai_samples = test_df[test_df["label"] == 1].sample(num_samples // 2, random_state=42)
    samples = pd.concat([human_samples, ai_samples]).reset_index(drop=True)

    # Define forward function that takes input_ids and attention_mask
    def forward_func(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    # Set up LayerIntegratedGradients on the word embedding layer
    lig = LayerIntegratedGradients(forward_func, model.distilbert.embeddings.word_embeddings)

    all_results = []

    for i in tqdm(range(len(samples)), desc="  Attributing tokens"):
        text = samples.iloc[i]["text"]
        true_label = samples.iloc[i]["label"]

        # Tokenize
        encoding = tokenizer(
            text, truncation=True, max_length=512,
            padding="max_length", return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Get model prediction
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_class = torch.argmax(logits, dim=1).item()
            pred_prob = torch.softmax(logits, dim=1)[0, pred_class].item()

        # Baseline: PAD token ids
        ref_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

        try:
            attributions, delta = lig.attribute(
                inputs=input_ids,
                baselines=ref_input_ids,
                additional_forward_args=(attention_mask,),
                target=pred_class,
                n_steps=25,
                return_convergence_delta=True
            )

            # Sum across embedding dimensions to get per-token attribution
            attr_scores = attributions.sum(dim=-1).squeeze(0)  # (seq_len,)
            attr_scores = attr_scores.cpu().detach().numpy()

            # Get tokens (only non-padding)
            tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
            seq_len = attention_mask.sum().item()

            all_results.append({
                "text_preview": text[:100],
                "true_label": "AI" if true_label == 1 else "Human",
                "pred_label": "AI" if pred_class == 1 else "Human",
                "pred_prob": pred_prob,
                "tokens": tokens[:seq_len],
                "attributions": attr_scores[:seq_len],
                "convergence_delta": delta.item() if delta.dim() == 0 else delta.mean().item()
            })
        except Exception as e:
            print(f"  Warning: Failed on sample {i}: {e}")
            continue

    # Plot token attributions for a few samples
    _plot_token_attributions(all_results[:6])

    # Generate summary
    _summarize_captum_results(all_results)

    return all_results


def _plot_token_attributions(results, max_tokens=60):
    """Visualize token-level attributions for select samples."""
    if not results:
        print("  No attribution results to plot.")
        return
    n = min(len(results), 6)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.5 * n))
    if n == 1:
        axes = [axes]

    for idx, (result, ax) in enumerate(zip(results[:n], axes)):
        tokens = result["tokens"][:max_tokens]
        attrs = result["attributions"][:max_tokens]

        # Normalize attributions for coloring
        abs_max = max(np.abs(attrs).max(), 1e-8)
        norm_attrs = attrs / abs_max

        colors = []
        for a in norm_attrs:
            if a > 0:
                colors.append((1.0, 0.4, 0.4, min(abs(a), 1.0)))  # red = pushes AI
            else:
                colors.append((0.4, 0.4, 1.0, min(abs(a), 1.0)))  # blue = pushes Human

        ax.bar(range(len(tokens)), attrs, color=colors, edgecolor="none")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=6)
        ax.set_ylabel("Attribution")
        ax.set_title(
            f"Sample {idx+1} | True: {result['true_label']} | "
            f"Pred: {result['pred_label']} ({result['pred_prob']:.2%})",
            fontsize=10
        )
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "captum_token_attributions.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved captum_token_attributions.png")
    plt.close("all")


def _summarize_captum_results(results):
    """Print a summary of Captum attribution analysis."""
    print(f"\n{'='*50}")
    print("Captum Attribution Summary")
    print(f"{'='*50}")
    if not results:
        print("  No successful attributions to summarize.")
        return
    print(f"  Samples analyzed: {len(results)}")

    correct = sum(1 for r in results if r["true_label"] == r["pred_label"])
    print(f"  Correct predictions: {correct}/{len(results)}")

    avg_delta = np.mean([abs(r["convergence_delta"]) for r in results])
    print(f"  Avg convergence delta: {avg_delta:.6f} (lower = more reliable)")

    # Find most commonly important tokens across AI predictions
    ai_results = [r for r in results if r["pred_label"] == "AI"]
    if ai_results:
        token_importance = {}
        for r in ai_results:
            for tok, attr in zip(r["tokens"], r["attributions"]):
                if tok in ("[CLS]", "[SEP]", "[PAD]"):
                    continue
                if tok not in token_importance:
                    token_importance[tok] = []
                token_importance[tok].append(attr)

        # Average attribution per token
        avg_importance = {t: np.mean(v) for t, v in token_importance.items() if len(v) >= 2}
        if avg_importance:
            top_ai = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Top tokens pushing toward AI prediction:")
            for tok, score in top_ai:
                print(f"    {tok:>15s}  {score:+.4f}")


# ──────────────────────────────────────────────
# Part C: Cross-Model Comparison
# ──────────────────────────────────────────────

def compare_feature_rankings(shap_xgb, shap_rf, shap_lr, feature_names):
    """Compare feature importance rankings across all 3 statistical models."""
    print("\n[Compare] Feature importance rankings across models")

    def rank_features(shap_vals):
        shap_arr = np.array(shap_vals)
        # Handle 3D arrays (n_samples, n_features, n_classes) — take class 1
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1]
        mean_abs = np.abs(shap_arr).mean(axis=0).flatten()
        indices = np.argsort(mean_abs)[::-1]
        return {feature_names[int(indices[rank])]: rank + 1 for rank in range(len(indices))}

    xgb_ranks = rank_features(shap_xgb)
    rf_ranks = rank_features(shap_rf)
    lr_ranks = rank_features(shap_lr)

    comparison = pd.DataFrame({
        "Feature": feature_names,
        "XGBoost Rank": [xgb_ranks[f] for f in feature_names],
        "Random Forest Rank": [rf_ranks[f] for f in feature_names],
        "Logistic Reg Rank": [lr_ranks[f] for f in feature_names]
    })
    comparison["Avg Rank"] = comparison[["XGBoost Rank", "Random Forest Rank", "Logistic Reg Rank"]].mean(axis=1)
    comparison = comparison.sort_values("Avg Rank")

    print("\n" + comparison.to_string(index=False))

    # Save to CSV
    comparison.to_csv(os.path.join(SAVE_DIR, "feature_ranking_comparison.csv"), index=False)
    print(f"\n  Saved feature_ranking_comparison.csv")

    # Heatmap
    rank_data = comparison.set_index("Feature")[["XGBoost Rank", "Random Forest Rank", "Logistic Reg Rank"]]
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(rank_data, annot=True, fmt=".0f", cmap="YlOrRd_r", ax=ax, linewidths=0.5)
    ax.set_title("Feature Importance Ranking Across Models\n(1 = most important)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "feature_ranking_heatmap.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved feature_ranking_heatmap.png")
    plt.close("all")

    return comparison


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 5: Explainability Analysis")
    print("=" * 60)

    # Load test features
    X_test, y_test, feature_names = load_test_features()
    print(f"\nTest set: {len(X_test)} samples, {len(feature_names)} features")

    # Use a subsample for SHAP speed (still representative)
    shap_sample_size = min(1000, len(X_test))
    X_shap = X_test.sample(shap_sample_size, random_state=42)
    print(f"SHAP subsample: {shap_sample_size} samples")

    # ── A: SHAP for all 3 statistical models ──
    shap_xgb = shap_xgboost(X_shap, feature_names)
    shap_rf = shap_random_forest(X_shap, feature_names)
    shap_lr = shap_logistic_regression(X_shap, feature_names)

    # Dependence plots for XGBoost
    shap_dependence_plots(shap_xgb, X_shap, feature_names)

    # Cross-model comparison
    compare_feature_rankings(shap_xgb, shap_rf, shap_lr, feature_names)

    # ── B: Captum for DistilBERT ──
    captum_results = captum_distilbert(num_samples=10)

    # ── Done ──
    print(f"\n{'='*60}")
    print("EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"All outputs saved to {SAVE_DIR}/")
    print("  - shap_xgboost_bar.png         (XGBoost global feature importance)")
    print("  - shap_xgboost_beeswarm.png    (XGBoost feature effect directions)")
    print("  - shap_xgb_dependence.png      (Top 4 feature dependence plots)")
    print("  - shap_rf_bar.png              (Random Forest importance)")
    print("  - shap_lr_bar.png              (Logistic Regression importance)")
    print("  - feature_ranking_heatmap.png  (Cross-model ranking comparison)")
    print("  - feature_ranking_comparison.csv")
    print("  - captum_token_attributions.png (DistilBERT token-level attributions)")
