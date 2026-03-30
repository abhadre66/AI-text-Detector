import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

# ──────────────────────────────────────────────
# Individual feature functions
# ──────────────────────────────────────────────

def avg_word_length(text):
    words = text.split()
    if not words:
        return 0
    return np.mean([len(w) for w in words])


def avg_sentence_length(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    return np.mean([len(s.split()) for s in sentences])


def sentence_length_variance(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0
    lengths = [len(s.split()) for s in sentences]
    return np.var(lengths)


def type_token_ratio(text):
    """Vocabulary richness: unique words / total words."""
    words = text.lower().split()
    if not words:
        return 0
    return len(set(words)) / len(words)


def hapax_legomena_ratio(text):
    """Ratio of words that appear exactly once."""
    words = text.lower().split()
    if not words:
        return 0
    freq = Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    return hapax / len(words)


def punctuation_ratio(text):
    """Ratio of punctuation characters to total characters."""
    if not text:
        return 0
    punct_count = sum(1 for c in text if c in '.,;:!?-—()[]{}"\'/\\')
    return punct_count / len(text)


def comma_ratio(text):
    if not text:
        return 0
    return text.count(',') / len(text)


def paragraph_count(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return len(paragraphs)


def avg_paragraph_length(text):
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return 0
    return np.mean([len(p.split()) for p in paragraphs])


def stopword_ratio(text):
    """Ratio of common English stopwords."""
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
        'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this', 'it',
        'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'us', 'them'
    }
    words = text.lower().split()
    if not words:
        return 0
    return sum(1 for w in words if w in stopwords) / len(words)


def repetition_rate(text, n=3):
    """Ratio of repeated n-grams (measures repetitiveness)."""
    words = text.lower().split()
    if len(words) < n:
        return 0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    freq = Counter(ngrams)
    repeated = sum(c - 1 for c in freq.values() if c > 1)
    return repeated / len(ngrams) if ngrams else 0


def lexical_density(text):
    """Ratio of content words (not stopwords) to total words."""
    return 1 - stopword_ratio(text)


def exclamation_rate(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0
    excl_count = text.count('!')
    return excl_count / len(sentences)


def question_rate(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0
    q_count = text.count('?')
    return q_count / len(sentences)


def starts_with_capital_ratio(text):
    """Ratio of sentences that start with a capital letter."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    caps = sum(1 for s in sentences if s[0].isupper())
    return caps / len(sentences)


def digit_ratio(text):
    if not text:
        return 0
    return sum(1 for c in text if c.isdigit()) / len(text)


def word_count(text):
    return len(text.split())


def char_count(text):
    return len(text)


def sentence_count(text):
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


# ──────────────────────────────────────────────
# Burstiness (requires a language model)
# ──────────────────────────────────────────────

def compute_perplexity_features(texts, batch_size=8):
    """
    Compute perplexity and burstiness using GPT-2 as a reference model.
    Perplexity = how 'surprised' a language model is by the text.
    Burstiness = variance in per-sentence perplexity.

    AI text tends to have LOW perplexity (predictable) and LOW burstiness (uniform).
    Human text tends to have HIGHER perplexity and HIGHER burstiness (varied).
    """
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("Loading GPT-2 for perplexity computation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model.eval()

    perplexities = []
    burstiness_scores = []

    for text in tqdm(texts, desc="Computing perplexity"):
        # Overall perplexity
        try:
            encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = encodings.input_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
            ppl = math.exp(loss)
        except Exception:
            ppl = 0.0

        perplexities.append(ppl)

        # Per-sentence perplexity for burstiness
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 3]

        sent_ppls = []
        for sent in sentences[:20]:  # cap at 20 sentences for speed
            try:
                enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
                ids = enc.input_ids.to(device)
                with torch.no_grad():
                    out = model(ids, labels=ids)
                    sent_ppls.append(math.exp(out.loss.item()))
            except Exception:
                continue

        burstiness = np.var(sent_ppls) if len(sent_ppls) > 1 else 0.0
        burstiness_scores.append(burstiness)

    return perplexities, burstiness_scores


# ──────────────────────────────────────────────
# Main feature extraction pipeline
# ──────────────────────────────────────────────

def extract_all_features(df, compute_perplexity=True):
    """
    Extract all statistical features from a DataFrame with a 'text' column.
    Returns the DataFrame with new feature columns added.
    """
    print(f"Extracting features from {len(df)} samples...")

    # Fast features (no model needed)
    tqdm.pandas(desc="Basic features")

    df["word_count"] = df["text"].apply(word_count)
    df["char_count"] = df["text"].apply(char_count)
    df["sentence_count"] = df["text"].apply(sentence_count)
    df["avg_word_length"] = df["text"].apply(avg_word_length)
    df["avg_sentence_length"] = df["text"].apply(avg_sentence_length)
    df["sentence_length_variance"] = df["text"].apply(sentence_length_variance)
    df["type_token_ratio"] = df["text"].apply(type_token_ratio)
    df["hapax_legomena_ratio"] = df["text"].apply(hapax_legomena_ratio)
    df["punctuation_ratio"] = df["text"].apply(punctuation_ratio)
    df["comma_ratio"] = df["text"].apply(comma_ratio)
    df["stopword_ratio"] = df["text"].apply(stopword_ratio)
    df["lexical_density"] = df["text"].apply(lexical_density)
    df["repetition_rate"] = df["text"].apply(repetition_rate)
    df["paragraph_count"] = df["text"].apply(paragraph_count)
    df["avg_paragraph_length"] = df["text"].apply(avg_paragraph_length)
    df["exclamation_rate"] = df["text"].apply(exclamation_rate)
    df["question_rate"] = df["text"].apply(question_rate)
    df["starts_with_capital_ratio"] = df["text"].apply(starts_with_capital_ratio)
    df["digit_ratio"] = df["text"].apply(digit_ratio)

    print(f"Extracted 19 basic features.")

    # Perplexity + burstiness (slow, needs GPU ideally)
    if compute_perplexity:
        perplexities, burstiness = compute_perplexity_features(df["text"].tolist())
        df["perplexity"] = perplexities
        df["burstiness"] = burstiness
        print(f"Extracted perplexity + burstiness features.")

    print(f"Total features: {len([c for c in df.columns if c not in ['text', 'label', 'label_name', 'source_model', 'domain', 'prompt']])}")

    return df


if __name__ == "__main__":
    # Process each split
    #for split in ["train", "val", "test"]: 
    for split in ["test"]:
        print(f"\n{'='*50}")
        
        print(f"Processing {split} set...")
        print(f"{'='*50}")

        input_path = f"data/processed/{split}.csv"
        output_path = f"data/processed/{split}_features.csv"

        df = pd.read_csv(input_path)
        df = extract_all_features(df, compute_perplexity=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    print("\nDone! Feature extraction complete for all splits.")