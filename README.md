---
title: AI Text Detector
emoji: 🧠
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# AI Text Detector

A machine learning system that classifies text as **human-written** or **AI-generated** using an ensemble approach combining statistical feature analysis (XGBoost) with a fine-tuned transformer model (DistilBERT).

## Model Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 99.18%  |
| F1 Score  | 99.17%  |
| Precision | 99.26%  |
| Recall    | 99.07%  |
| AUROC     | 99.96%  |

## Key Features

- **Ensemble Architecture** — Combines XGBoost (statistical features) and DistilBERT (transformer) with weighted scoring for robust predictions
- **21 Engineered Features** — Includes vocabulary richness, punctuation patterns, sentence structure, GPT-2 perplexity, and burstiness
- **Explainability** — SHAP-based feature importance analysis and Captum token-level attribution for interpretable results
- **Robustness Tested** — Evaluated against adversarial attacks including synonym substitution, character perturbations, and mixed-origin text
- **Web Interface** — Clean Flask-based UI with confidence visualization and token heatmap analysis

## Data Sources

The model is trained on a diverse, multi-source dataset:

- **HC3 Dataset** — Human and ChatGPT answers across Wikipedia, Reddit ELI5, medicine, and finance domains
- **data2.csv** — Essay-based human vs AI text samples
- **AI_Human.csv** — Mixed-domain human vs AI text samples

All datasets are merged, deduplicated, class-balanced, and split into 80/10/10 train/val/test sets.

## Setup

```bash
git clone https://github.com/yourusername/AI-text-Detector.git
cd AI-text-Detector
pip install -r requirements.txt
```

For data collection, create `config/.env` with your API keys:

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Usage

### Train the Models

The trained models are not included in the repository due to file size. Run the following to generate them:

```bash
python src/data_collection.py        # Collect and merge datasets
python src/feature_engineering.py     # Extract 21 statistical + LM features
python src/train_statistical.py       # Train XGBoost, Random Forest, Logistic Regression
python src/train_transformer.py       # Fine-tune DistilBERT
python src/explainability.py          # Generate SHAP + Captum analysis (optional)
python src/robustness.py              # Run adversarial robustness tests (optional)
```

> Training auto-detects Apple Silicon (MPS) / CUDA / CPU.

### Run the App

```bash
python flask_app.py
```

Open http://localhost:5000 in your browser.

## Tech Stack

- **ML/DL:** PyTorch, HuggingFace Transformers, scikit-learn, XGBoost
- **Explainability:** SHAP, Captum
- **Data:** pandas, NumPy, HuggingFace Datasets
- **Visualization:** matplotlib, seaborn
- **Web App:** Flask, HTML/CSS/JavaScript
- **APIs:** OpenAI, Anthropic

## License

MIT License — see [LICENSE](LICENSE) for details.
