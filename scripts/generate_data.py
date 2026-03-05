#!/usr/bin/env python3
"""
Generate synthetic network traffic data (KDD Cup 99 style).
Creates labeled data for SVM training and unlabeled data for DBSCAN testing.

Usage:
    python generate_data.py [N_SAMPLES] [N_FEATURES]
    Default: 10000 samples, 34 features

Output files:
    data/train_data.csv   - Training features (70%)
    data/train_labels.csv - Training labels
    data/test_data.csv    - Test features (30%)
    data/test_labels.csv  - Test labels
"""
import numpy as np
import sys
import os
import time

def generate_kdd_like_data(n_samples=10000, n_features=34, seed=42):
    np.random.seed(seed)

    # 5 classes: 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R
    # Realistic class distribution (imbalanced like KDD)
    class_weights = [0.20, 0.60, 0.10, 0.08, 0.02]
    labels = np.random.choice(5, size=n_samples, p=class_weights)

    # Generate cluster centers for each class
    centers = np.random.randn(5, n_features) * 3
    # Make classes more separable
    centers[0] += 2.0   # Normal
    centers[1] -= 3.0   # DoS
    centers[2] += 5.0   # Probe
    centers[3] -= 1.0   # R2L (close to normal - hard to classify)
    centers[4] += 0.5   # U2R (close to normal - hard to classify)

    # Generate data around centers with varying spread
    spreads = [0.8, 1.0, 0.9, 1.5, 2.0]  # R2L and U2R have more spread
    data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        c = labels[i]
        data[i] = centers[c] + np.random.randn(n_features) * spreads[c]

    # Min-Max normalize to [0, 1]
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero
    data = (data - mins) / ranges

    return data, labels

def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    n_features = int(sys.argv[2]) if len(sys.argv) > 2 else 34

    print(f"Generating {n_samples} samples with {n_features} features...")
    start = time.time()

    data, labels = generate_kdd_like_data(n_samples, n_features)

    # Split 70/30
    split_idx = int(n_samples * 0.7)
    indices = np.random.permutation(n_samples)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_data = data[train_idx]
    train_labels = labels[train_idx]
    test_data = data[test_idx]
    test_labels = labels[test_idx]

    # Save
    os.makedirs("data", exist_ok=True)
    np.savetxt("data/train_data.csv", train_data, delimiter=",", fmt="%.8f")
    np.savetxt("data/train_labels.csv", train_labels, delimiter=",", fmt="%d")
    np.savetxt("data/test_data.csv", test_data, delimiter=",", fmt="%.8f")
    np.savetxt("data/test_labels.csv", test_labels, delimiter=",", fmt="%d")

    elapsed = time.time() - start

    class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    print(f"\nDataset generated in {elapsed:.2f}s")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test:  {len(test_data)} samples")
    print(f"  Features: {n_features}")
    print(f"\nClass distribution (train):")
    for i, name in enumerate(class_names):
        count = np.sum(train_labels == i)
        print(f"  {name}: {count} ({count/len(train_labels)*100:.1f}%)")
    print(f"\nFiles saved to data/")

if __name__ == "__main__":
    main()
