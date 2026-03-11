#!/usr/bin/env python3
"""
Overfitting Analysis Plot
=========================
Reads per-technique accuracy CSVs and epoch error CSVs,
then produces two plots:
  1. Grouped bar chart: Train vs Test accuracy per technique
  2. Epoch error convergence curves (avg across class pairs) per technique

Run from the directory where *_accuracy.csv and *_epoch_errors.csv live:
    python scripts/plot_overfitting.py
or specify an output directory:
    python scripts/plot_overfitting.py --outdir /path/to/results
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

TECHNIQUES = ["seq", "openmp", "thread", "mpi", "cuda", "pyspark"]
LABELS     = ["Sequential", "OpenMP", "C++ Thread", "MPI", "CUDA", "PySpark"]
COLORS     = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def load_accuracy(prefix, search_dir):
    """Return (train_acc, test_acc) or (None, None) if file missing."""
    path = os.path.join(search_dir, f"{prefix}_accuracy.csv")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        line = f.readline().strip()
    try:
        parts = line.split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None


def load_epoch_errors(prefix, search_dir):
    """
    Returns (train_dict, val_dict) where each is
    {(class_a, class_b): list_of_error_rates}.
    error_rate = errors / n_samples per epoch.
    """
    path = os.path.join(search_dir, f"{prefix}_epoch_errors.csv")
    if not os.path.exists(path):
        return None, None
    train_data = {}
    val_data = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            ca, cb = int(parts[0]), int(parts[1])
            train_errors, n_train = int(parts[3]), int(parts[4])
            val_errors, n_val = int(parts[5]), int(parts[6])
            key = (ca, cb)
            train_rate = train_errors / n_train if n_train > 0 else 0.0
            val_rate = val_errors / n_val if n_val > 0 else 0.0
            train_data.setdefault(key, []).append(train_rate)
            val_data.setdefault(key, []).append(val_rate)
    return (train_data if train_data else None,
            val_data if val_data else None)


def average_epoch_curve(epoch_dict):
    """Average error rate across all class pairs; pad shorter series."""
    if not epoch_dict:
        return None
    max_len = max(len(v) for v in epoch_dict.values())
    all_series = []
    for v in epoch_dict.values():
        padded = v + [v[-1]] * (max_len - len(v))  # hold last value
        all_series.append(padded)
    return np.mean(all_series, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Plot overfitting analysis")
    parser.add_argument("--outdir", default=".", help="Directory to search for CSV files and save plots")
    args = parser.parse_args()

    search_dir = args.outdir

    # ── collect data ─────────────────────────────────────────────────────────
    acc_data       = {}   # prefix -> (train, test)
    train_curves   = {}   # prefix -> avg_curve array   (training error)
    val_curves     = {}   # prefix -> avg_curve array   (validation error)

    for prefix, label in zip(TECHNIQUES, LABELS):
        train, test = load_accuracy(prefix, search_dir)
        if train is not None:
            acc_data[prefix] = (train, test)
            print(f"[{label}]  Train={train:.2f}%  Test={test:.2f}%")

        ep_train, ep_val = load_epoch_errors(prefix, search_dir)
        if ep_train is not None:
            curve = average_epoch_curve(ep_train)
            if curve is not None:
                train_curves[prefix] = curve
                print(f"[{label}]  Train curve: {len(curve)} epochs, "
                      f"final error rate={curve[-1]:.4f}")
        if ep_val is not None:
            curve = average_epoch_curve(ep_val)
            if curve is not None:
                val_curves[prefix] = curve
                print(f"[{label}]  Val   curve: {len(curve)} epochs, "
                      f"final error rate={curve[-1]:.4f}")

    if not acc_data and not train_curves:
        print("No CSV files found in", search_dir)
        print("Make sure *_accuracy.csv and *_epoch_errors.csv exist there.")
        sys.exit(1)

    # ── figure layout ─────────────────────────────────────────────────────────
    has_curves = train_curves or val_curves
    n_rows = 1 + (1 if has_curves else 0)
    fig = plt.figure(figsize=(14, 5 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.45)

    # ── Plot 1: Train vs Test Accuracy bar chart ──────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    keys_present = [p for p in TECHNIQUES if p in acc_data]
    labels_present = [LABELS[TECHNIQUES.index(p)] for p in keys_present]
    colors_present = [COLORS[TECHNIQUES.index(p)] for p in keys_present]

    x = np.arange(len(keys_present))
    width = 0.35
    train_vals = [acc_data[p][0] for p in keys_present]
    test_vals  = [acc_data[p][1] for p in keys_present]

    bars1 = ax1.bar(x - width/2, train_vals, width, label="Train Accuracy",
                    color=[c + "BB" for c in colors_present],  # slight transparency
                    edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + width/2, test_vals, width, label="Test Accuracy",
                    color=colors_present,
                    edgecolor="white", linewidth=0.8)

    # annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_present, fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Train vs Test Accuracy — Overfitting Check", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=10)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # highlight gap
    for i, p in enumerate(keys_present):
        tr, te = acc_data[p]
        gap = tr - te
        if gap > 5:
            ax1.annotate(f"gap={gap:.1f}%", xy=(x[i], max(tr, te) + 1),
                         ha="center", fontsize=8, color="crimson")

    # ── Plot 2: Epoch Error Convergence (Train + Val) ───────────────────────
    if has_curves:
        ax2 = fig.add_subplot(gs[1])
        for prefix in TECHNIQUES:
            label  = LABELS[TECHNIQUES.index(prefix)]
            color  = COLORS[TECHNIQUES.index(prefix)]

            if prefix in train_curves:
                curve = train_curves[prefix]
                epochs = np.arange(1, len(curve) + 1)
                ax2.plot(epochs, [v * 100 for v in curve],
                         label=f"{label} (train)", color=color, linewidth=2,
                         marker="o", markersize=3,
                         markevery=max(1, len(curve)//20))

            if prefix in val_curves:
                curve = val_curves[prefix]
                epochs = np.arange(1, len(curve) + 1)
                ax2.plot(epochs, [v * 100 for v in curve],
                         label=f"{label} (val)", color=color, linewidth=2,
                         linestyle="--", marker="s", markersize=3,
                         markevery=max(1, len(curve)//20))

        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Avg Error Rate (%)", fontsize=12)
        ax2.set_title("SVM Training Convergence — Train vs Validation Error Rate\n"
                      "(averaged across all 1-vs-1 class pairs)", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=9, ncol=2)
        ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax2.set_axisbelow(True)

    out_path = os.path.join(search_dir, "overfitting_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    main()
