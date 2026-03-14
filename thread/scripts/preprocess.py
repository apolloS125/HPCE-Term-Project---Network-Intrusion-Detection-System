
#!/usr/bin/env python3
"""
Preprocessing pipeline (Option 3):
  1. Remove constant features
  2. Undersample majority classes
  3. StandardScaler (fit on train only)
  4. Save processed data + scaler + class_weight config
"""

import numpy as np
import os, joblib
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# ── Config ───────────────────────────────────────────────────
UNDERSAMPLE_STRATEGY = {
    2: 100_000,   # DDoS
    3: 150_000,   # DoS
    4: 300_000,   # Normal Traffic  ← ลดจาก 1.66M
    # 5: Port Scanning — ไม่แตะ (minority สุด ~90K)
}
RANDOM_SEED = 42

def preprocess():
    print("=" * 60)
    print("  Preprocessing Pipeline — Option 3")
    print("  (Undersample + StandardScaler + class_weight config)")
    print("=" * 60)

    os.makedirs("data/processed", exist_ok=True)

    # ── Load raw ─────────────────────────────────────────────
    print("\n[1/4] Loading raw data...")
    X_train = np.loadtxt("data/raw/train_data.csv",   delimiter=",")
    y_train = np.loadtxt("data/raw/train_labels.csv", dtype=int)
    X_test  = np.loadtxt("data/raw/test_data.csv",    delimiter=",")
    y_test  = np.loadtxt("data/raw/test_labels.csv",  dtype=int)
    print(f"      Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Remove constant features ──────────────────────────────
    print("\n[2/4] Removing constant features...")
    var       = X_train.var(axis=0)
    non_const = var > 1e-10
    X_train   = X_train[:, non_const]
    X_test    = X_test[:, non_const]
    print(f"      Features remaining: {X_train.shape[1]}")

    # ── Undersample (train only) ──────────────────────────────
    print("\n[3/4] Undersampling majority classes...")
    print(f"      Before: {len(X_train):,} samples")

    # ปรับ strategy — ไม่แตะ class ที่มีน้อยกว่า target
    unique, counts = np.unique(y_train, return_counts=True)
    strategy = {}
    for cls, cnt in zip(unique, counts):
        if cls in UNDERSAMPLE_STRATEGY and cnt > UNDERSAMPLE_STRATEGY[cls]:
            strategy[cls] = UNDERSAMPLE_STRATEGY[cls]
        # ถ้าน้อยกว่า target หรือไม่ได้ระบุ → คงเดิม

    rus = RandomUnderSampler(sampling_strategy=strategy, random_state=RANDOM_SEED)
    X_train, y_train = rus.fit_resample(X_train, y_train)
    print(f"      After:  {len(X_train):,} samples")

    unique_after, counts_after = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique_after, counts_after):
        print(f"        Class {cls}: {cnt:,}")

    # ── StandardScaler ────────────────────────────────────────
    print("\n[4/4] Scaling (fit on train only)...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)     # ← transform เท่านั้น ไม่ fit

    # Save processed
    np.savetxt("data/processed/train_data.csv",   X_train, delimiter=",", fmt="%.8f")
    np.savetxt("data/processed/train_labels.csv", y_train, fmt="%d")
    np.savetxt("data/processed/test_data.csv",    X_test,  delimiter=",", fmt="%.8f")
    np.savetxt("data/processed/test_labels.csv",  y_test,  fmt="%d")

    # Save scaler (สำหรับ inference ทีหลัง)
    joblib.dump(scaler, "data/processed/scaler.pkl")
    print("      Scaler saved → data/processed/scaler.pkl")

    # Save class_weight config สำหรับ SVM
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_cls = len(unique_train)
    with open("data/processed/class_weight_config.txt", "w") as f:
        f.write("# class_weight='balanced' equivalent\n")
        f.write("# weight = n_samples / (n_classes * n_samples_per_class)\n\n")
        for cls, cnt in zip(unique_train, counts_train):
            w = total / (n_cls * cnt)
            f.write(f"class_{cls}_weight={w:.6f}\n")

    print(f"\n{'='*60}")
    print(f"  Processed output → data/processed/")
    print(f"  Next step        → python train_svm.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    preprocess()
