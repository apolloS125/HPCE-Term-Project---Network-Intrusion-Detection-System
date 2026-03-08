#!/usr/bin/env python3
"""
Download and preprocess CICIDS2017 dataset for Network IDS project.

Uses kagglehub to download the cleaned & preprocessed CICIDS2017 dataset
from Kaggle (ericanacletoribeiro/cicids2017-cleaned-and-preprocessed).

Usage:
    python download_real_data.py

Output:
    data/train_data.csv, train_labels.csv, test_data.csv, test_labels.csv
"""

import numpy as np
import pandas as pd
import os
import shutil
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import kagglehub
from kagglehub import KaggleDatasetAdapter


def download_cicids():
    """Download and preprocess CICIDS2017 dataset using kagglehub."""
    print("=" * 60)
    print("  CICIDS2017 Dataset (cleaned & preprocessed via kagglehub)")
    print("=" * 60)

    os.makedirs("data/cicids", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("  Loading dataset from Kaggle via kagglehub...")
    print("  Dataset: ericanacletoribeiro/cicids2017-cleaned-and-preprocessed")
    print()

    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "ericanacletoribeiro/cicids2017-cleaned-and-preprocessed",
        "cicids2017_cleaned.csv",
    )

    print(f"  Total records: {len(df)}")

    # Clean data
    print("  Cleaning data...")

    # Get label column (could be 'Label', 'Attack Type', etc.)
    label_col = None
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in ('label', 'attack type', 'attack_type', 'class'):
            label_col = col
            break

    if label_col is None:
        print("  ERROR: Cannot find label column")
        return None, None, None, None

    # Map CICIDS labels to numeric categories
    # 0=Benign, 1-N=Attack types
    df[label_col] = df[label_col].str.strip()
    unique_labels = df[label_col].unique()
    print(f"  Unique labels: {len(unique_labels)}")
    for lbl in sorted(unique_labels):
        count = (df[label_col] == lbl).sum()
        print(f"    {lbl}: {count:,}")

    # Encode labels
    le = LabelEncoder()
    df['category'] = le.fit_transform(df[label_col])

    # Select numeric features only
    feature_cols = []
    for col in df.columns:
        if col not in [label_col, 'category']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    feature_cols.append(col)
            except:
                pass

    print(f"  Numeric features: {len(feature_cols)}")

    # Drop rows with NaN/Inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    print(f"  Records after cleaning: {len(df)}")

    X = df[feature_cols].values.astype(np.float64)
    y = df['category'].values.astype(int)

    # Normalize
    print("  Min-Max normalizing...")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Remove constant features
    var = X.var(axis=0)
    non_const = var > 1e-10
    X = X[:, non_const]
    print(f"  Features after removing constants: {X.shape[1]}")

    # Split 70/30
    print("  Splitting 70/30 train/test...")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split = int(len(X) * 0.7)

    train_X = X[indices[:split]]
    train_y = y[indices[:split]]
    test_X = X[indices[split:]]
    test_y = y[indices[split:]]

    # Save to data/cicids/
    print("  Saving processed data...")
    np.savetxt("data/cicids/train_data.csv", train_X, delimiter=",", fmt="%.8f")
    np.savetxt("data/cicids/train_labels.csv", train_y, delimiter=",", fmt="%d")
    np.savetxt("data/cicids/test_data.csv", test_X, delimiter=",", fmt="%.8f")
    np.savetxt("data/cicids/test_labels.csv", test_y, delimiter=",", fmt="%d")

    # Save label mapping
    with open("data/cicids/label_mapping.txt", "w") as f:
        for i, name in enumerate(le.classes_):
            f.write(f"{i}: {name}\n")

    # Copy to data/ root for default usage
    for fname in ['train_data.csv', 'train_labels.csv', 'test_data.csv', 'test_labels.csv']:
        shutil.copy(f"data/cicids/{fname}", f"data/{fname}")

    print(f"\n  Done! Train: {len(train_X)} | Test: {len(test_X)} | Features: {X.shape[1]} | Classes: {len(le.classes_)}")
    print(f"  Files saved to data/cicids/ and data/")

    return train_X, train_y, test_X, test_y


if __name__ == "__main__":
    download_cicids()
