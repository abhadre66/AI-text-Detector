# AI Text Detector

A machine learning system that classifies text as **human-written** or **AI-generated** using a dual-model architecture — statistical features (XGBoost) + fine-tuned transformer (DistilBERT).

## Results

### Statistical Model (XGBoost) — Test Set

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 99.18%  |
| F1 Score  | 99.17%  |
| Precision | 99.26%  |
| Recall    | 99.07%  |
| AUROC     | 99.96%  |

## Project Structure

```
AI-text-Detector/
├── app.py                          # Gradio web interface (upcoming)
├── requirements.txt                # Python dependencies
├── config/
│   └── .env                        # API keys (OpenAI, Anthropic)
├── data/
│   ├── raw/                        # Raw collected data
│   │   ├── hc3_raw.csv
│   │   └── gpt4_generated.csv
│   └── processed/                  # Train/val/test splits + features
│       ├── train.csv / val.csv / test.csv
│       └── train_features.csv / val_features.csv / test_features.csv
├── models/                         # Saved models + plots
│   ├── xgboost_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── random_forest_model.joblib
│   ├── distilbert/                 # Fine-tuned transformer
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── src/
│   ├── data_collection.py          # HC3 dataset + GPT-4 API generation
│   ├── feature_engineering.py      # 21 statistical + 2 LM features
│   ├── train_statistical.py        # Logistic Regression, Random Forest, XGBoost
│   ├── train_transformer.py        # DistilBERT fine-tuning
│   ├── explainability.py           # SHAP + Captum (upcoming)
│   └── utils.py                    # Utility functions (upcoming)
└── notebooks/                      # Exploratory notebooks
```

## Pipeline

The project follows a 7-phase pipeline:

### Phase 1: Data Collection
Collects human vs AI text from two sources:
- **HC3 Dataset** (HuggingFace) — Human + ChatGPT answers across domains (Wikipedia, Reddit ELI5, medicine, finance)
- **GPT-4 API** — Generated responses using 200 HC3 prompts

Data is cleaned, deduplicated, class-balanced, and split 80/10/10 into train/val/test (~42K training samples).

### Phase 2: Feature Engineering
Extracts 21 features per text sample:
- **Basic stats** — word count, sentence count, avg word/sentence length, sentence length variance
- **Vocabulary richness** — type-token ratio, hapax legomena ratio, lexical density
- **Punctuation & style** — punctuation ratio, comma ratio, exclamation/question rate
- **Structure** — paragraph count, avg paragraph length, stopword ratio, repetition rate
- **Language model features** — GPT-2 perplexity and burstiness (most discriminative)

### Phase 3: Statistical Model Training
Trains three classifiers on the extracted features with grid search + 5-fold cross-validation:
- **Logistic Regression** — baseline with L1/L2 regularization
- **Random Forest** — 108 hyperparameter combinations
- **XGBoost** — 192 hyperparameter combinations (selected as best model)

All training includes `tqdm` progress bars with ETA and running best F1 score.

### Phase 4: Transformer Fine-Tuning
Fine-tunes **DistilBERT** (`distilbert-base-uncased`) on raw text:
- Reads text directly (no handcrafted features needed)
- 3 epochs, batch size 16, learning rate 2e-5 with linear warmup
- Auto-detects Apple Silicon (MPS) / CUDA / CPU
- Saves best checkpoint by validation F1

### Phase 5–7 (Upcoming)
- **Explainability** — SHAP values for feature importance, Captum for token-level attribution
- **Robustness Evaluation** — Adversarial test suite with paraphrased AI text
- **Demo UI** — Gradio app with both models and visual explanations

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-text-Detector.git
cd AI-text-Detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys (optional, for data collection only)

Create `config/.env`:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## Usage

Run each phase sequentially:

```bash
# Phase 1: Collect data
python src/data_collection.py

# Phase 2: Extract features
python src/feature_engineering.py

# Phase 3: Train statistical models
python src/train_statistical.py

# Phase 4: Train transformer
python src/train_transformer.py
```

## Tech Stack

- **ML/DL:** PyTorch, HuggingFace Transformers, scikit-learn, XGBoost
- **Explainability:** SHAP, Captum
- **Data:** pandas, NumPy, HuggingFace Datasets
- **Visualization:** matplotlib, seaborn
- **UI:** Gradio
- **APIs:** OpenAI, Anthropic

## License

MIT License — see [LICENSE](LICENSE) for details.
