import gradio as gr
import numpy as np
from src.feature_engineering import extract_all_features
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

HF_REPO = "Abhadre/AI-Text_detector"

# Load XGBoost
xgb_path = hf_hub_download(repo_id=HF_REPO, filename="xgboost_model.joblib")
xgb_model = joblib.load(xgb_path)

# Load DistilBERT
tokenizer = AutoTokenizer.from_pretrained(HF_REPO, subfolder="distilbert")
bert_model = AutoModelForSequenceClassification.from_pretrained(HF_REPO, subfolder="distilbert")
bert_model.eval()


# =========================
# 🔍 PREDICTION FUNCTION
# =========================
import pandas as pd

FEATURE_COLS = [
    "word_count", "char_count", "sentence_count",
    "avg_word_length", "avg_sentence_length", "sentence_length_variance",
    "type_token_ratio", "hapax_legomena_ratio", "lexical_density",
    "punctuation_ratio", "comma_ratio", "exclamation_rate",
    "question_rate", "starts_with_capital_ratio", "digit_ratio",
    "paragraph_count", "avg_paragraph_length", "stopword_ratio",
    "repetition_rate", "perplexity", "burstiness"
]

def _extract(text):
    df = pd.DataFrame({"text": [text]})
    df = extract_all_features(df, compute_perplexity=False)
    df["perplexity"] = 0.0
    df["burstiness"] = 0.0
    return df[FEATURE_COLS].fillna(0)

def predict_text(text):
    features = _extract(text)

    # ----- XGBoost -----
    xgb_prob = xgb_model.predict_proba(features.values)[0][1]

    # ----- DistilBERT -----
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    bert_prob = probs[1]

    # Weighted ensemble (XGBoost is stronger: 99.2% F1 vs DistilBERT)
    XGB_WEIGHT = 0.6
    final_ai = XGB_WEIGHT * xgb_prob + (1 - XGB_WEIGHT) * bert_prob
    final_human = 1 - final_ai

    label = "AI-generated" if final_ai > 0.5 else "Human-written"
    confidence = max(final_ai, final_human)

    return label, float(confidence), [final_human, final_ai]

# =========================
# 🔥 TOKEN HEATMAP
# =========================
def get_token_heatmap(text):
    tokens = text.split()
    
    # simple heuristic: longer + rare words = more "AI signal"
    scores = []
    for token in tokens:
        score = min(len(token) / 10, 1.0)  # normalize
        scores.append(score)

    return list(zip(tokens, scores))


def render_heatmap(token_scores):
    html = ""

    for token, score in token_scores:
        # red intensity
        intensity = int(255 * score)
        color = f"rgba(255, 0, 0, {score})"
        html += f'<span style="background-color:{color}; padding:2px; margin:2px;">{token}</span> '

    return html



# =========================
# 🤖 MULTI-MODEL BREAKDOWN
# =========================
def model_breakdown(text):
    features = _extract(text)
    xgb_prob = xgb_model.predict_proba(features.values)[0][1]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

    bert_prob = probs[1]

    return {
        "XGBoost": float(xgb_prob),
        "DistilBERT": float(bert_prob)
    }

# =========================
# 🚀 MAIN PIPELINE
# =========================
def analyze(text):
    label, confidence, probs = predict_text(text)

    token_scores = get_token_heatmap(text)
    heatmap_html = render_heatmap(token_scores)

    breakdown = model_breakdown(text)

    confidence_percent = confidence * 100

    return (
        f"{label} ({confidence_percent:.2f}%)",
        heatmap_html,
        breakdown
    )
    


# =========================
# 🎨 GRADIO UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Text Detector")
    gr.Markdown("Paste text to detect AI vs Human writing.")

    with gr.Row():
        text_input = gr.Textbox(
            lines=8,
            placeholder="Paste your text here..."
        )

    analyze_btn = gr.Button("Analyze")

    with gr.Row():
        prediction_output = gr.Textbox(label="Prediction")

    with gr.Row():
        heatmap_output = gr.HTML(label="Token Heatmap")

    with gr.Row():
        shap_output = gr.Plot(label="SHAP Explanation")

    with gr.Row():
        breakdown_output = gr.JSON(label="Model Breakdown")

    analyze_btn.click(
        analyze,
        inputs=text_input,
        outputs=[
            prediction_output,
            heatmap_output,
            breakdown_output
        ]
    )

# =========================
# ▶️ RUN
# =========================
demo.launch()