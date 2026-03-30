import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from src.feature_engineering import extract_all_features

app = Flask(__name__)

# Lazy-loaded models (avoid segfault from loading all at import time)
xgb_model = None
tokenizer = None
bert_model = None


def load_models():
    global xgb_model, tokenizer, bert_model
    if xgb_model is None:
        print("Loading DistilBERT...")
        tokenizer = AutoTokenizer.from_pretrained("models/distilbert")
        bert_model = AutoModelForSequenceClassification.from_pretrained("models/distilbert")
        bert_model.eval()
        print("Loading XGBoost...")
        xgb_model = joblib.load("models/xgboost_model.joblib")
        print("Models loaded.")

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
    df = extract_all_features(df, compute_perplexity=True)
    return df[FEATURE_COLS].fillna(0)


def predict_text(text):
    load_models()
    features = _extract(text)

    # XGBoost
    xgb_prob = xgb_model.predict_proba(features.values)[0][1]

    # DistilBERT
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    bert_prob = float(probs[1])

    # Weighted ensemble
    XGB_WEIGHT = 0.6
    final_ai = XGB_WEIGHT * xgb_prob + (1 - XGB_WEIGHT) * bert_prob
    final_human = 1 - final_ai

    label = "AI-Generated" if final_ai > 0.5 else "Human-Written"
    confidence = max(final_ai, final_human)

    return {
        "label": label,
        "confidence": round(float(confidence) * 100, 2),
        "ai_probability": round(float(final_ai) * 100, 2),
        "human_probability": round(float(final_human) * 100, 2),
    }


def get_token_heatmap(text):
    tokens = text.split()
    result = []
    for token in tokens:
        score = min(len(token) / 10, 1.0)
        result.append({"token": token, "score": round(score, 3)})
    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400

    result = predict_text(text)
    result["heatmap"] = get_token_heatmap(text)
    return jsonify(result)


if __name__ == "__main__":
    with app.app_context():
        print("Preloading models before starting server...")
        load_models()
        print("Ready!")
    app.run(debug=False, port=5000)
