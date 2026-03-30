import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def collect_hc3_data(save_dir="data/raw"):
    """
    Pull the HC3 dataset from HuggingFace.
    It contains human answers and ChatGPT answers to the same questions.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Loading HC3 dataset...")
    dataset = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)

    records = []
    for split in dataset:
        for row in tqdm(dataset[split], desc=f"Processing {split}"):
            question = row["question"]
            source = row.get("source", "unknown")

            # Human answers
            for answer in row["human_answers"]:
                if len(answer.strip()) > 50:
                    records.append({
                        "text": answer.strip(),
                        "label": 0,
                        "source_dataset": "hc3",
                        "domain": source,
                    })

            # ChatGPT answers
            for answer in row["chatgpt_answers"]:
                if len(answer.strip()) > 50:
                    records.append({
                        "text": answer.strip(),
                        "label": 1,
                        "source_dataset": "hc3",
                        "domain": source,
                    })

    df = pd.DataFrame(records)
    output_path = os.path.join(save_dir, "hc3_raw.csv")
    df.to_csv(output_path, index=False)
    print(f"HC3: {len(df)} samples (Human: {(df['label']==0).sum()}, AI: {(df['label']==1).sum()})")
    return df


def load_external_datasets(raw_dir="data/raw"):
    """
    Load data2.csv and AI_Human.csv, standardize columns, and return combined DataFrame.
    """
    frames = []

    # --- data2.csv ---
    path1 = os.path.join(raw_dir, "data2.csv")
    if os.path.exists(path1):
        d2 = pd.read_csv(path1)
        d2 = d2[["text", "label"]].copy()
        d2["label"] = d2["label"].astype(int)
        d2["source_dataset"] = "data2"
        d2["domain"] = "essays"
        frames.append(d2)
        print(f"data2.csv: {len(d2)} samples (Human: {(d2['label']==0).sum()}, AI: {(d2['label']==1).sum()})")
    else:
        print(f"WARNING: {path1} not found, skipping.")

    # --- AI_Human.csv ---
    path2 = os.path.join(raw_dir, "AI_Human.csv")
    if os.path.exists(path2):
        ah = pd.read_csv(path2)
        ah = ah.rename(columns={"generated": "label"})[["text", "label"]].copy()
        ah["label"] = ah["label"].astype(int)
        ah["source_dataset"] = "ai_human"
        ah["domain"] = "mixed"
        frames.append(ah)
        print(f"AI_Human.csv: {len(ah)} samples (Human: {(ah['label']==0).sum()}, AI: {(ah['label']==1).sum()})")
    else:
        print(f"WARNING: {path2} not found, skipping.")

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def merge_all_datasets(raw_dir="data/raw", save_path="data/raw/combined_raw.csv"):
    """
    Merge HC3 + data2.csv + AI_Human.csv into one unified dataset.
    Deduplicates on text content.
    """
    frames = []

    # Load HC3 if available
    hc3_path = os.path.join(raw_dir, "hc3_raw.csv")
    if os.path.exists(hc3_path):
        hc3 = pd.read_csv(hc3_path)
        # Standardize columns
        hc3 = hc3[["text", "label"]].copy()
        hc3["source_dataset"] = "hc3"
        hc3["domain"] = "qa"
        frames.append(hc3)
        print(f"HC3: {len(hc3)} samples")
    else:
        print(f"WARNING: {hc3_path} not found. Run with --mode full to collect HC3 first.")

    # Load external datasets
    ext_df = load_external_datasets(raw_dir)
    if len(ext_df) > 0:
        frames.append(ext_df)

    if not frames:
        print("ERROR: No datasets found!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Clean
    combined = combined.dropna(subset=["text"])
    combined["label"] = combined["label"].astype(int)

    # Deduplicate
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"])
    print(f"\nDeduplication: {before} -> {len(combined)} samples")

    # Summary
    print(f"\n--- Combined Dataset ---")
    print(f"Total: {len(combined)}")
    print(f"  Human: {(combined['label']==0).sum()} ({(combined['label']==0).mean()*100:.1f}%)")
    print(f"  AI:    {(combined['label']==1).sum()} ({(combined['label']==1).mean()*100:.1f}%)")
    print(f"  Sources: {combined['source_dataset'].value_counts().to_dict()}")

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")

    return combined


def preprocess_and_split(raw_path="data/raw/combined_raw.csv", save_dir="data/processed"):
    """
    Clean the data and split into train/val/test sets.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(raw_path)

    # Basic cleaning
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 50]
    df = df.drop_duplicates(subset=["text"])

    # Truncate very long texts to ~1000 words
    df["text"] = df["text"].apply(lambda x: " ".join(str(x).split()[:1000]))

    # Balance classes
    min_class_size = df["label"].value_counts().min()
    df_balanced = df.groupby("label").apply(
        lambda x: x.sample(n=min_class_size, random_state=42)
    ).reset_index(drop=True)

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split: 80% train, 10% val, 10% test
    n = len(df_balanced)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_df = df_balanced[:train_end]
    val_df = df_balanced[train_end:val_end]
    test_df = df_balanced[val_end:]

    # Save splits
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train label distribution: {train_df['label'].value_counts().to_dict()}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "merge"], default="merge",
                        help="full: collect HC3 from HuggingFace + merge all + split. "
                             "merge: merge existing raw files + split (no download).")
    args = parser.parse_args()

    if args.mode == "full":
        # Step 1: Download HC3
        collect_hc3_data()

        # Step 2: Merge all datasets
        merge_all_datasets()

        # Step 3: Split
        preprocess_and_split()

    elif args.mode == "merge":
        # Merge existing raw files and split (no API calls, no downloads)
        merge_all_datasets()
        preprocess_and_split()
