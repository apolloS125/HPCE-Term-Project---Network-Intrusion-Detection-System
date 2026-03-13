#!/usr/bin/env python3
"""
Download CICIDS2017 and split into train/test.
Output: raw (unscaled) CSV files in data/raw/
"""

import numpy as np
import pandas as pd
import os, shutil, argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import kagglehub
from kagglehub import KaggleDatasetAdapter

HOLDOUT_CLASS_IDS = [0, 1, 6]
TEST_SIZE         = 0.3
RANDOM_SEED       = 42
MIN_SAMPLES       = 100

def download_cicids(holdout_ids=None):
    if holdout_ids is None:
        holdout_ids = HOLDOUT_CLASS_IDS
    if isinstance(holdout_ids, int):
        holdout_ids = [holdout_ids]
    holdout_ids = sorted(list(set(holdout_ids)))

    print("=" * 60)
    print("  CICIDS2017 Dataset - Download & Split")
    print(f"  Holdout class IDs: {holdout_ids}")
    print("=" * 60)

    os.makedirs("data/raw", exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────
    print("\n[1/5] Downloading dataset from Kaggle...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "ericanacletoribeiro/cicids2017-cleaned-and-preprocessed",
        "cicids2017_cleaned.csv",
    )
    print(f"      Total records: {len(df):,}")

    # ── Step 2: Find label column ────────────────────────────
    print("\n[2/5] Finding label column...")
    label_col = None
    for col in df.columns:
        if col.strip().lower() in ('label', 'attack type', 'attack_type', 'class'):
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column. Columns: {df.columns.tolist()}")
    df[label_col] = df[label_col].str.strip()
    print(f"      Found: '{label_col}'")

    # ── Step 3: Clean (inf/nan เท่านั้น — ยังไม่ scale) ─────
    print("\n[3/5] Cleaning features...")
    feature_cols = [
        col for col in df.columns
        if col != label_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    print(f"      Records after cleaning: {len(df):,}")
    print(f"      Features: {len(feature_cols)}")

    # ── Step 4: Encode labels ────────────────────────────────
    print("\n[4/5] Encoding labels...")
    label_counts = df[label_col].value_counts()
    valid_labels  = label_counts[label_counts >= MIN_SAMPLES].index.tolist()
    df = df[df[label_col].isin(valid_labels)].copy()

    le = LabelEncoder()
    df['category'] = le.fit_transform(df[label_col])
    n_classes = len(le.classes_)

    holdout_ids   = [h for h in holdout_ids if h < n_classes]
    holdout_names = [le.classes_[h] for h in holdout_ids]
    svm_class_ids = [i for i in range(n_classes) if i not in holdout_ids]
    svm_names     = [le.classes_[i] for i in svm_class_ids]

    print(f"\n      {'ID':<5} {'Name':<40} {'Count':>10}   Role")
    print(f"      {'-'*72}")
    for i, name in enumerate(le.classes_):
        count = int((df['category'] == i).sum())
        role  = "HOLDOUT" if i in holdout_ids else "SVM train"
        print(f"      {i:<5} {name:<40} {count:>10,}   {role}")

    # ── Step 5: Split & Save raw ─────────────────────────────
    print("\n[5/5] Splitting and saving raw data...")
    X = df[feature_cols].values.astype(np.float64)
    y = df['category'].values.astype(int)

    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # กรอง holdout ออกจาก train — แต่ย้ายไป test แทนที่จะทิ้ง
    mask_train   = np.isin(y_train_all, holdout_ids, invert=True)
    mask_ho_spill = ~mask_train                          # holdout ที่ตกฝั่ง train

    X_train = X_train_all[mask_train]
    y_train = y_train_all[mask_train]

    # เพิ่ม holdout ที่เคยตกฝั่ง train เข้า test (ไม่ถูก train อยู่แล้ว)
    X_test  = np.vstack([X_test,  X_train_all[mask_ho_spill]])
    y_test  = np.concatenate([y_test, y_train_all[mask_ho_spill]])

    ho_moved = int(mask_ho_spill.sum())
    print(f"      Train (no holdout): {len(X_train):,} samples")
    print(f"      Holdout moved train→test: {ho_moved:,}")
    print(f"      Test  (all classes): {len(X_test):,} samples")

    # Save
    np.savetxt("data/raw/train_data.csv",   X_train, delimiter=",", fmt="%.8f")
    np.savetxt("data/raw/train_labels.csv", y_train, fmt="%d")
    np.savetxt("data/raw/test_data.csv",    X_test,  delimiter=",", fmt="%.8f")
    np.savetxt("data/raw/test_labels.csv",  y_test,  fmt="%d")

    # Save feature names (preprocessing จะต้องใช้)
    with open("data/raw/feature_names.txt", "w") as f:
        f.write("\n".join(feature_cols))

    # holdout_config.txt
    with open("data/raw/holdout_config.txt", "w") as f:
        f.write(f"holdout_class_ids={','.join(map(str, holdout_ids))}\n")
        f.write(f"holdout_class_names={','.join(holdout_names)}\n")
        f.write(f"svm_class_ids={','.join(map(str, svm_class_ids))}\n")
        f.write(f"svm_class_names={','.join(svm_names)}\n")
        f.write(f"n_classes_total={n_classes}\n")
        f.write(f"n_features_raw={X_train.shape[1]}\n")
        f.write(f"n_train_raw={len(X_train)}\n")
        f.write(f"n_test={len(X_test)}\n")

    print(f"\n{'='*60}")
    print(f"  Raw output → data/raw/")
    print(f"  Next step  → python preprocessing.py")
    print(f"{'='*60}")

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=int, nargs="+", default=HOLDOUT_CLASS_IDS)
    args = parser.parse_args()
    download_cicids(holdout_ids=args.holdout)