"""
Phase 6: Robustness Evaluation
───────────────────────────────
Tests the best model (XGBoost) against:
  1. Synonym substitution attacks
  2. Character-level perturbations (typos, case, punctuation)
  3. Mixed-origin text (human + AI spliced)
  4. Confidence calibration analysis
  5. Text truncation (partial input)

Outputs saved to models/robustness/
"""

import os
import json
import random
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

SAVE_DIR = "models/robustness"
DATA_DIR = "data/processed"
MODEL_DIR = "models"

FEATURE_COLS = [
    "word_count", "char_count", "sentence_count",
    "avg_word_length", "avg_sentence_length", "sentence_length_variance",
    "type_token_ratio", "hapax_legomena_ratio", "lexical_density",
    "punctuation_ratio", "comma_ratio", "exclamation_rate",
    "question_rate", "starts_with_capital_ratio", "digit_ratio",
    "paragraph_count", "avg_paragraph_length", "stopword_ratio",
    "repetition_rate", "perplexity", "burstiness"
]

random.seed(42)
np.random.seed(42)


# ──────────────────────────────────────────────
# Feature extraction (mirrors feature_engineering.py)
# ──────────────────────────────────────────────

def extract_features_single(text):
    """Extract statistical features from a single text string."""
    import re
    import math

    words = text.split()
    word_count = len(words)
    char_count = len(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)

    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sentence_length = np.mean(sent_lengths) if sent_lengths else 0
    sentence_length_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0

    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / max(word_count, 1)

    word_freq = {}
    for w in words:
        wl = w.lower()
        word_freq[wl] = word_freq.get(wl, 0) + 1
    hapax = sum(1 for c in word_freq.values() if c == 1)
    hapax_legomena_ratio = hapax / max(word_count, 1)

    # Lexical density (content words / total words) — approximate via stopwords
    STOPWORDS = {"the","a","an","is","are","was","were","be","been","being","have","has","had",
                 "do","does","did","will","would","shall","should","may","might","must","can",
                 "could","i","you","he","she","it","we","they","me","him","her","us","them",
                 "my","your","his","its","our","their","this","that","these","those","and",
                 "but","or","nor","not","so","yet","both","either","neither","each","every",
                 "all","any","few","more","most","other","some","such","no","only","own","same",
                 "than","too","very","just","because","as","until","while","of","at","by","for",
                 "with","about","against","between","through","during","before","after","above",
                 "below","to","from","up","down","in","out","on","off","over","under","again",
                 "further","then","once","here","there","when","where","why","how","what","which",
                 "who","whom","if","into"}
    content_words = [w for w in words if w.lower() not in STOPWORDS]
    lexical_density = len(content_words) / max(word_count, 1)

    punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}"\'/\\')
    punctuation_ratio = punct_count / max(char_count, 1)
    comma_ratio = text.count(',') / max(char_count, 1)
    exclamation_rate = text.count('!') / max(sentence_count, 1)
    question_rate = text.count('?') / max(sentence_count, 1)

    capital_sentences = sum(1 for s in sentences if s and s[0].isupper())
    starts_with_capital_ratio = capital_sentences / max(sentence_count, 1)

    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / max(char_count, 1)

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraph_count = max(len(paragraphs), 1)
    avg_paragraph_length = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0

    stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
    stopword_ratio = stopword_count / max(word_count, 1)

    # Repetition rate: fraction of bigrams that repeat
    if len(words) > 1:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        unique_bigrams = set(bigrams)
        repetition_rate = 1 - len(unique_bigrams) / len(bigrams)
    else:
        repetition_rate = 0

    # Perplexity and burstiness: set to 0 (need GPT-2 model to compute)
    # These will be filled from the original features if available
    perplexity = 0
    burstiness = 0

    return {
        "word_count": word_count, "char_count": char_count,
        "sentence_count": sentence_count, "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_variance": sentence_length_variance,
        "type_token_ratio": type_token_ratio,
        "hapax_legomena_ratio": hapax_legomena_ratio,
        "lexical_density": lexical_density,
        "punctuation_ratio": punctuation_ratio, "comma_ratio": comma_ratio,
        "exclamation_rate": exclamation_rate, "question_rate": question_rate,
        "starts_with_capital_ratio": starts_with_capital_ratio,
        "digit_ratio": digit_ratio, "paragraph_count": paragraph_count,
        "avg_paragraph_length": avg_paragraph_length,
        "stopword_ratio": stopword_ratio, "repetition_rate": repetition_rate,
        "perplexity": perplexity, "burstiness": burstiness
    }


# ──────────────────────────────────────────────
# Load model and data
# ──────────────────────────────────────────────

def load_model():
    """Load the best XGBoost model."""
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.joblib"))
    return model


def load_test_data():
    """Load test set with both raw text and features."""
    test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv")).dropna(subset=["text"])
    test_feat = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))

    available = [f for f in FEATURE_COLS if f in test_feat.columns]
    X_test = test_feat[available].fillna(0)
    y_test = test_feat["label"]

    return test_raw, X_test, y_test, available


def predict_from_text(model, texts, original_perplexity=None, original_burstiness=None):
    """Extract features from raw text and predict."""
    features_list = []
    for i, text in enumerate(texts):
        feats = extract_features_single(text)
        # Use original perplexity/burstiness if available (can't recompute without GPT-2)
        if original_perplexity is not None:
            feats["perplexity"] = original_perplexity[i]
        if original_burstiness is not None:
            feats["burstiness"] = original_burstiness[i]
        features_list.append(feats)

    df = pd.DataFrame(features_list)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    return preds, probas


# ──────────────────────────────────────────────
# Attack 1: Synonym Substitution
# ──────────────────────────────────────────────

SYNONYM_MAP = {
    "good": ["great", "fine", "decent", "nice"],
    "bad": ["poor", "terrible", "awful", "lousy"],
    "important": ["significant", "crucial", "vital", "key"],
    "help": ["assist", "aid", "support", "facilitate"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "make": ["create", "produce", "generate", "build"],
    "show": ["demonstrate", "indicate", "reveal", "display"],
    "think": ["believe", "consider", "assume", "suppose"],
    "provide": ["offer", "supply", "deliver", "furnish"],
    "include": ["contain", "encompass", "comprise", "involve"],
    "large": ["big", "huge", "massive", "enormous"],
    "small": ["tiny", "little", "minor", "slight"],
    "different": ["various", "diverse", "distinct", "varied"],
    "however": ["nevertheless", "nonetheless", "yet", "still"],
    "therefore": ["thus", "hence", "consequently", "accordingly"],
    "also": ["additionally", "moreover", "furthermore", "besides"],
    "example": ["instance", "case", "illustration", "sample"],
    "result": ["outcome", "consequence", "effect", "product"],
    "people": ["individuals", "persons", "humans", "folks"],
    "because": ["since", "as", "given that", "due to the fact that"],
}


def synonym_attack(text, swap_rate=0.1):
    """Replace a fraction of words with synonyms."""
    words = text.split()
    new_words = []
    for w in words:
        w_lower = w.lower().strip(".,;:!?\"'()")
        if w_lower in SYNONYM_MAP and random.random() < swap_rate:
            replacement = random.choice(SYNONYM_MAP[w_lower])
            # Preserve capitalization
            if w[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            trailing = ""
            for c in reversed(w):
                if c in ".,;:!?\"'()":
                    trailing = c + trailing
                else:
                    break
            new_words.append(replacement + trailing)
        else:
            new_words.append(w)
    return " ".join(new_words)


# ──────────────────────────────────────────────
# Attack 2: Character-level Perturbations
# ──────────────────────────────────────────────

def typo_attack(text, rate=0.02):
    """Insert random typos: swap adjacent chars, double letters, drop letters."""
    chars = list(text)
    n_changes = max(1, int(len(chars) * rate))
    indices = random.sample(range(1, len(chars) - 1), min(n_changes, len(chars) - 2))

    for idx in indices:
        action = random.choice(["swap", "double", "drop"])
        if action == "swap" and idx < len(chars) - 1:
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        elif action == "double":
            chars.insert(idx, chars[idx])
        elif action == "drop" and chars[idx].isalpha():
            chars[idx] = ""

    return "".join(chars)


def case_attack(text, rate=0.05):
    """Randomly flip case of characters."""
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < rate:
            chars[i] = chars[i].swapcase()
    return "".join(chars)


def punctuation_attack(text, rate=0.3):
    """Remove a fraction of punctuation marks."""
    result = []
    for c in text:
        if c in ".,;:!?" and random.random() < rate:
            continue
        result.append(c)
    return "".join(result)


# ──────────────────────────────────────────────
# Attack 3: Mixed-Origin Text
# ──────────────────────────────────────────────

def create_mixed_texts(human_texts, ai_texts, n_samples=200):
    """Splice human and AI text together at various ratios."""
    mixed = []
    labels = []  # 1 if majority AI, 0 if majority human
    ratios = []

    for ai_ratio in [0.25, 0.50, 0.75]:
        for _ in range(n_samples // 3):
            h_text = random.choice(human_texts)
            a_text = random.choice(ai_texts)

            h_sentences = re.split(r'(?<=[.!?])\s+', h_text)
            a_sentences = re.split(r'(?<=[.!?])\s+', a_text)

            if not h_sentences or not a_sentences:
                continue

            total_sents = max(len(h_sentences), len(a_sentences))
            n_ai = max(1, int(total_sents * ai_ratio))
            n_human = max(1, total_sents - n_ai)

            selected_h = h_sentences[:n_human]
            selected_a = a_sentences[:n_ai]

            # Interleave
            combined = []
            hi, ai = 0, 0
            for j in range(n_human + n_ai):
                if j % 2 == 0 and hi < len(selected_h):
                    combined.append(selected_h[hi])
                    hi += 1
                elif ai < len(selected_a):
                    combined.append(selected_a[ai])
                    ai += 1

            if combined:
                mixed.append(" ".join(combined))
                labels.append(1 if ai_ratio >= 0.5 else 0)
                ratios.append(ai_ratio)

    return mixed, labels, ratios


# ──────────────────────────────────────────────
# Attack 4: Text Truncation
# ──────────────────────────────────────────────

def truncate_text(text, keep_ratio=0.5):
    """Keep only a fraction of the text (from the beginning)."""
    words = text.split()
    n_keep = max(5, int(len(words) * keep_ratio))
    return " ".join(words[:n_keep])


# ──────────────────────────────────────────────
# Attack 5: Confidence Calibration
# ──────────────────────────────────────────────

def calibration_analysis(y_true, y_proba):
    """Analyze probability calibration."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Reliability diagram
    ax1.plot(prob_pred, prob_true, "s-", color="steelblue", label="XGBoost")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Confidence histogram
    ax2.hist(y_proba, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Predicted Probability (AI class)")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Confidence Distribution")
    ax2.axvline(0.5, color="red", linestyle="--", label="Decision boundary")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "calibration.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved calibration.png")
    plt.close("all")

    # Expected Calibration Error
    bin_counts = np.histogram(y_proba, bins=10, range=(0, 1))[0]
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_counts[:len(prob_true)] / len(y_proba)))
    return ece


# ──────────────────────────────────────────────
# Run all attacks and generate report
# ──────────────────────────────────────────────

def run_robustness_evaluation():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("Phase 6: Robustness Evaluation (XGBoost)")
    print("=" * 60)

    model = load_model()
    test_raw, X_test, y_test, feature_names = load_test_data()
    test_feat = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))
    texts = test_feat["text"].tolist()
    labels = test_feat["label"].tolist()

    # Baseline performance on clean test set
    print("\n[Baseline] Clean test set")
    clean_preds = model.predict(X_test)
    clean_probas = model.predict_proba(X_test)[:, 1]
    baseline_acc = accuracy_score(y_test, clean_preds)
    baseline_f1 = f1_score(y_test, clean_preds)
    baseline_auroc = roc_auc_score(y_test, clean_probas)
    print(f"  Accuracy: {baseline_acc:.4f} | F1: {baseline_f1:.4f} | AUROC: {baseline_auroc:.4f}")

    # Subsample for text-based attacks (feature extraction is slow without GPT-2)
    n_attack = min(500, len(texts))
    sample_idx = random.sample(range(len(texts)), n_attack)
    sample_texts = [texts[i] for i in sample_idx]
    sample_labels = np.array([labels[i] for i in sample_idx])

    # Get original perplexity/burstiness for the sampled indices
    orig_perplexity = test_feat["perplexity"].values[sample_idx] if "perplexity" in test_feat.columns else None
    orig_burstiness = test_feat["burstiness"].values[sample_idx] if "burstiness" in test_feat.columns else None

    results = {
        "attack": ["Clean (baseline)"],
        "accuracy": [baseline_acc],
        "f1": [baseline_f1],
        "accuracy_drop": [0.0],
        "f1_drop": [0.0]
    }

    # Sub-baseline on the sample
    sample_preds, sample_probas = predict_from_text(model, sample_texts, orig_perplexity, orig_burstiness)
    sample_acc = accuracy_score(sample_labels, sample_preds)
    sample_f1 = f1_score(sample_labels, sample_preds)
    print(f"\n[Sub-baseline] {n_attack} sample subset (re-extracted features)")
    print(f"  Accuracy: {sample_acc:.4f} | F1: {sample_f1:.4f}")

    # ── Attack 1: Synonym Substitution ──
    for swap_rate in [0.1, 0.3, 0.5]:
        attack_name = f"Synonym swap ({int(swap_rate*100)}%)"
        print(f"\n[Attack] {attack_name}")
        attacked = [synonym_attack(t, swap_rate) for t in tqdm(sample_texts, desc="  Attacking")]
        preds, probas = predict_from_text(model, attacked, orig_perplexity, orig_burstiness)
        acc = accuracy_score(sample_labels, preds)
        f1 = f1_score(sample_labels, preds)
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Drop: {sample_acc - acc:+.4f}")
        results["attack"].append(attack_name)
        results["accuracy"].append(acc)
        results["f1"].append(f1)
        results["accuracy_drop"].append(sample_acc - acc)
        results["f1_drop"].append(sample_f1 - f1)

    # ── Attack 2: Character Perturbations ──
    for name, attack_fn, rate in [
        ("Typos (2%)", typo_attack, 0.02),
        ("Typos (5%)", typo_attack, 0.05),
        ("Case flip (5%)", case_attack, 0.05),
        ("Remove punct (30%)", punctuation_attack, 0.3),
        ("Remove punct (60%)", punctuation_attack, 0.6),
    ]:
        print(f"\n[Attack] {name}")
        attacked = [attack_fn(t, rate) for t in tqdm(sample_texts, desc="  Attacking")]
        preds, probas = predict_from_text(model, attacked, orig_perplexity, orig_burstiness)
        acc = accuracy_score(sample_labels, preds)
        f1 = f1_score(sample_labels, preds)
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Drop: {sample_acc - acc:+.4f}")
        results["attack"].append(name)
        results["accuracy"].append(acc)
        results["f1"].append(f1)
        results["accuracy_drop"].append(sample_acc - acc)
        results["f1_drop"].append(sample_f1 - f1)

    # ── Attack 3: Mixed-Origin Text ──
    human_texts = [texts[i] for i in range(len(texts)) if labels[i] == 0]
    ai_texts = [texts[i] for i in range(len(texts)) if labels[i] == 1]

    for ai_ratio in [0.25, 0.50, 0.75]:
        attack_name = f"Mixed text ({int(ai_ratio*100)}% AI)"
        print(f"\n[Attack] {attack_name}")
        mixed, mixed_labels, _ = create_mixed_texts(human_texts, ai_texts, n_samples=300)
        mixed_labels = np.array(mixed_labels)
        preds, probas = predict_from_text(model, mixed)
        acc = accuracy_score(mixed_labels, preds)
        f1 = f1_score(mixed_labels, preds)
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f}")
        results["attack"].append(attack_name)
        results["accuracy"].append(acc)
        results["f1"].append(f1)
        results["accuracy_drop"].append(sample_acc - acc)
        results["f1_drop"].append(sample_f1 - f1)

    # ── Attack 4: Truncation ──
    for keep in [0.75, 0.50, 0.25]:
        attack_name = f"Truncation (keep {int(keep*100)}%)"
        print(f"\n[Attack] {attack_name}")
        truncated = [truncate_text(t, keep) for t in sample_texts]
        preds, probas = predict_from_text(model, truncated, orig_perplexity, orig_burstiness)
        acc = accuracy_score(sample_labels, preds)
        f1 = f1_score(sample_labels, preds)
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Drop: {sample_acc - acc:+.4f}")
        results["attack"].append(attack_name)
        results["accuracy"].append(acc)
        results["f1"].append(f1)
        results["accuracy_drop"].append(sample_acc - acc)
        results["f1_drop"].append(sample_f1 - f1)

    # ── Attack 5: Calibration ──
    print(f"\n[Calibration] Analyzing prediction confidence...")
    ece = calibration_analysis(y_test, clean_probas)
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")

    # ── Generate Summary Plot ──
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(SAVE_DIR, "robustness_results.csv"), index=False)
    print(f"\n  Saved robustness_results.csv")

    # Bar chart of accuracy under each attack
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    attacks = results_df["attack"]
    x = range(len(attacks))

    colors = ["#2ecc71" if d <= 0.01 else "#f39c12" if d <= 0.05 else "#e74c3c"
              for d in results_df["accuracy_drop"]]

    ax1.barh(x, results_df["accuracy"], color=colors, edgecolor="white")
    ax1.set_yticks(x)
    ax1.set_yticklabels(attacks, fontsize=9)
    ax1.set_xlabel("Accuracy")
    ax1.set_title("XGBoost Accuracy Under Adversarial Attacks")
    ax1.axvline(baseline_acc, color="gray", linestyle="--", alpha=0.7, label=f"Baseline ({baseline_acc:.3f})")
    ax1.set_xlim(0, 1.05)
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    ax2.barh(x, results_df["f1"], color=colors, edgecolor="white")
    ax2.set_yticks(x)
    ax2.set_yticklabels(attacks, fontsize=9)
    ax2.set_xlabel("F1 Score")
    ax2.set_title("XGBoost F1 Score Under Adversarial Attacks")
    ax2.axvline(baseline_f1, color="gray", linestyle="--", alpha=0.7, label=f"Baseline ({baseline_f1:.3f})")
    ax2.set_xlim(0, 1.05)
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "robustness_summary.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved robustness_summary.png")
    plt.close("all")

    # ── Degradation curve for truncation ──
    trunc_ratios = [1.0, 0.75, 0.50, 0.25]
    trunc_accs = [sample_acc]
    trunc_f1s = [sample_f1]
    for keep in [0.75, 0.50, 0.25]:
        row = results_df[results_df["attack"] == f"Truncation (keep {int(keep*100)}%)"]
        trunc_accs.append(row["accuracy"].values[0])
        trunc_f1s.append(row["f1"].values[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trunc_ratios, trunc_accs, "o-", color="steelblue", label="Accuracy", linewidth=2)
    ax.plot(trunc_ratios, trunc_f1s, "s-", color="coral", label="F1 Score", linewidth=2)
    ax.set_xlabel("Fraction of Text Retained")
    ax.set_ylabel("Score")
    ax.set_title("Performance Degradation Under Text Truncation")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0.2, 1.05)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "truncation_curve.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved truncation_curve.png")
    plt.close("all")

    # ── Final Summary ──
    print(f"\n{'='*60}")
    print("ROBUSTNESS EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\nExpected Calibration Error: {ece:.4f}")
    print(f"\nAll outputs saved to {SAVE_DIR}/")

    # Save summary JSON
    summary = {
        "model": "XGBoost",
        "baseline_accuracy": baseline_acc,
        "baseline_f1": baseline_f1,
        "baseline_auroc": baseline_auroc,
        "ece": ece,
        "attacks": results_df.to_dict(orient="records")
    }
    with open(os.path.join(SAVE_DIR, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved robustness_summary.json")


if __name__ == "__main__":
    run_robustness_evaluation()
