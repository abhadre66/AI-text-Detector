import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# Step 1: Dataset class — tokenizes raw text
# ──────────────────────────────────────────────

class TextDataset(Dataset):
    """Tokenizes text on-the-fly and returns input_ids, attention_mask, label."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ──────────────────────────────────────────────
# Step 2: Load and prepare data
# ──────────────────────────────────────────────

def load_data(data_dir="data/processed"):
    """Load raw text train/val/test CSVs."""
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    # Drop rows with missing text
    train = train.dropna(subset=["text"])
    val = val.dropna(subset=["text"])
    test = test.dropna(subset=["text"])

    return train, val, test


def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=16, max_length=512):
    """Create PyTorch DataLoaders from dataframes."""
    train_dataset = TextDataset(
        train_df["text"].tolist(), train_df["label"].tolist(),
        tokenizer, max_length
    )
    val_dataset = TextDataset(
        val_df["text"].tolist(), val_df["label"].tolist(),
        tokenizer, max_length
    )
    test_dataset = TextDataset(
        test_df["text"].tolist(), test_df["label"].tolist(),
        tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────
# Step 3: Training loop with progress bar
# ──────────────────────────────────────────────

def train_one_epoch(model, train_loader, optimizer, scheduler, device):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc="  Training", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


# ──────────────────────────────────────────────
# Step 4: Evaluation
# ──────────────────────────────────────────────

def evaluate(model, data_loader, device):
    """Evaluate model. Returns labels, predictions, and probabilities."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    pbar = tqdm(data_loader, desc="  Evaluating", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def print_metrics(y_true, y_pred, y_proba, label=""):
    """Compute and print all metrics. Returns dict."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_proba)
    }

    print(f"\n--- {label} Results ---")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Human', 'AI'])}")

    return metrics


# ──────────────────────────────────────────────
# Step 5: Visualization (same style as statistical)
# ──────────────────────────────────────────────

def plot_training_history(train_losses, val_f1s, save_dir="models"):
    """Plot training loss and validation F1 over epochs."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_f1s, "g-o", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "transformer_training_history.png"), dpi=150)
    print(f"Saved training history to {save_dir}/transformer_training_history.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Human", "AI"], yticklabels=["Human", "AI"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — DistilBERT")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "transformer_confusion_matrix.png"), dpi=150)
    print(f"Saved confusion matrix to {save_dir}/transformer_confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_true, y_proba, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"DistilBERT (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — DistilBERT")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "transformer_roc_curve.png"), dpi=150)
    print(f"Saved ROC curve to {save_dir}/transformer_roc_curve.png")
    plt.close()


# ──────────────────────────────────────────────
# Step 6: Save model
# ──────────────────────────────────────────────

def save_model(model, tokenizer, metrics, save_dir="models/distilbert"):
    """Save the fine-tuned model, tokenizer, and metadata."""
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved model and tokenizer to {save_dir}/")

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump({"model": "DistilBERT", "metrics": metrics}, f, indent=2)
    print(f"Saved metadata to {save_dir}/metadata.json")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

if __name__ == "__main__":
    pipeline_start = time.time()

    # ── Config ──
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    SAVE_DIR = "models"

    # ── Device ──
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (this will be slow)")

    # ── Step 1: Load data ──
    print("\n[1/6] Loading data...")
    train_df, val_df, test_df = load_data()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Step 2: Tokenizer & DataLoaders ──
    print("\n[2/6] Tokenizing text...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer, BATCH_SIZE, MAX_LENGTH
    )
    print(f"  Batches per epoch: {len(train_loader)}")

    # ── Step 3: Load pretrained model ──
    print("\n[3/6] Loading pretrained DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)

    # ── Step 4: Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # ── Step 5: Training loop ──
    print(f"\n[4/6] Training for {EPOCHS} epochs...")
    best_val_f1 = 0
    train_losses = []
    val_f1s = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'─'*50}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'─'*50}")

        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(avg_loss)

        # Validate
        val_labels, val_preds, val_probs = evaluate(model, val_loader, device)
        val_metrics = print_metrics(val_labels, val_preds, val_probs, f"Epoch {epoch} Validation")
        val_f1s.append(val_metrics["f1"])

        epoch_time = time.time() - epoch_start
        mins, secs = divmod(int(epoch_time), 60)
        print(f"\n  Epoch time: {mins}m {secs}s | Avg loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_model(model, tokenizer, val_metrics, os.path.join(SAVE_DIR, "distilbert"))
            print(f"  >>> New best model saved (F1: {best_val_f1:.4f})")

    # ── Step 6: Final test evaluation ──
    print(f"\n{'='*50}")
    print("[5/6] Final Test Evaluation")
    print(f"{'='*50}")

    # Reload best model
    model = DistilBertForSequenceClassification.from_pretrained(
        os.path.join(SAVE_DIR, "distilbert")
    )
    model.to(device)

    test_labels, test_preds, test_probs = evaluate(model, test_loader, device)
    test_metrics = print_metrics(test_labels, test_preds, test_probs, "Test Set (Best Model)")

    # Save updated metadata with test metrics
    with open(os.path.join(SAVE_DIR, "distilbert", "metadata.json"), "w") as f:
        json.dump({
            "model": "DistilBERT",
            "val_metrics": {"f1": best_val_f1},
            "test_metrics": test_metrics
        }, f, indent=2)

    # ── Plots ──
    print(f"\n[6/6] Generating plots...")
    plot_training_history(train_losses, val_f1s, SAVE_DIR)
    plot_confusion_matrix(test_labels, test_preds, SAVE_DIR)
    plot_roc_curve(test_labels, test_probs, SAVE_DIR)

    # ── Summary ──
    total_time = time.time() - pipeline_start
    mins, secs = divmod(int(total_time), 60)
    hrs, mins = divmod(mins, 60)

    print(f"\n{'='*50}")
    print("TRANSFORMER TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Model:      DistilBERT (distilbert-base-uncased)")
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Test F1:     {test_metrics['f1']:.4f}")
    print(f"Test AUROC:  {test_metrics['auroc']:.4f}")
    print(f"Total time:  {hrs}h {mins}m {secs}s")
    print(f"Saved to:    {SAVE_DIR}/distilbert/")
