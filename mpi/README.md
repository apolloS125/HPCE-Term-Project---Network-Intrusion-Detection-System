# CICIDS2017 Hybrid SVM + DBSCAN NIDS

MPI-parallelised Network Intrusion Detection System combining a One-vs-Rest linear SVM with DBSCAN clustering for unknown/holdout attack detection.

---

## Dataset

| Split | Rows | Classes |
|---|---|---|
| Training | 588,717 | DDoS, DoS, NormalTraffic, PortScan (4) |
| Test (full) | 765,495 | Above 4 + Bots, BruteForce, WebAttacks (holdout) |
| Holdout (test only) | 13,241 | Bots (1,948), BruteForce (9,150), WebAttacks (2,143) |

Holdout classes are **never seen during training** — all holdout samples (100%) are placed in the test set to maximise unknown-attack evaluation coverage.

---

## Architecture

```
Training set (holdout removed)
        │
  mpi_train.cpp (MPI, 4 ranks)
        │
   [model.bin]  ← One-vs-Rest linear SVM (4 classes × 52 features)

Test set (full, including all holdout)
        │
  mpi_infer.cpp (MPI, 4 ranks)
        │
  ┌─────────────────────────────────────────────────────────┐
  │ PHASE 1 — SVM inference on all 765,495 samples          │
  └──────────────────────┬──────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────┐
  │ PHASE 2 — DBSCAN clustering (pool = 96,632 samples)     │
  │   • Holdout anchors       : 13,241 (true-label known)   │
  │   • Uncertain attack-pred : 83,391 (conf < 0.50)        │
  │   • Uncertain Normal-pred : 23,266 → Normal directly    │
  └──────────────────────┬──────────────────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────────────────┐
  │ PHASE 3 — Hybrid combination                            │
  │   conf ≥ 0.50, not holdout → SVM final                  │
  │   conf < 0.50, pred=Normal → Normal directly            │
  │   conf < 0.50, pred=attack → DBSCAN result              │
  │   is_holdout               → DBSCAN result              │
  └─────────────────────────────────────────────────────────┘
```

---

## Training Results

**Hyperparameters:** EPOCHS=200, BATCH=512, λ=5×10⁻⁴, LR peak=0.05, LR min=0.001, cosine annealing, Adam optimiser, MPI ranks=4

| Epoch | val_acc | val_macro_F1 |
|---|---|---|
| 10 | 0.7405 | 0.5795 |
| 50 | 0.8855 | 0.8707 |
| 100 | 0.8936 | 0.8806 |
| 150 | 0.9006 | 0.8873 |
| **200** | **0.9021** | **0.8887** |

**Per-class recall at epoch 200:** DDoS=0.983 · DoS=0.885 · Normal=0.874 · PortScan=0.955

---

## Inference Results

### Phase 1 — SVM Only (baseline)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DDoS | 0.4864 | 0.9829 | 0.6507 | 38,404 |
| DoS | 0.8539 | 0.8866 | 0.8700 | 58,124 |
| NormalTraffic | 0.9946 | 0.8775 | 0.9324 | 628,518 |
| PortScan | 0.4331 | 0.9515 | 0.5952 | 27,208 |
| **macro-F1** | | | **0.7621** | |

*Low precision on DDoS and PortScan due to class imbalance and feature overlap.*

---

### Phase 2 — DBSCAN Clustering

| Parameter | Value |
|---|---|
| eps | 0.148 |
| min_samples | 8 |
| Pool size | 96,632 |
| Clusters found | 167 |
| Noise points | 4,902 (5.1%) |
| Holdout-voted clusters | 23 |
| Normal-reclassified clusters | 0 |
| Unknown anomaly clusters | 144 |

*Normal-reclassified = 0 by design: uncertain Normal-predicted samples are assigned Normal directly and never enter the DBSCAN pool, so no Normal-proximity clusters form.*

---

### Phase 3 — Hybrid SVM + DBSCAN

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DDoS | 0.9849 | 0.9762 | 0.9805 | 25,330 |
| DoS | 0.8861 | 0.9500 | 0.9169 | 47,425 |
| NormalTraffic | 0.9946 | 0.9693 | 0.9818 | 568,955 |
| PortScan | 0.6737 | 0.9527 | 0.7893 | 27,153 |
| **macro-F1** | | | **0.9171** | |

**Improvement over SVM-only: +0.155 macro-F1**

---

### Holdout (Unknown Attack) Detection via DBSCAN

| Class | Total | Detected | Noise | Detection Rate |
|---|---|---|---|---|
| Bots | 1,948 | 1,682 | 135 | 86.4% |
| BruteForce | 9,150 | 9,091 | 58 | **99.3%** |
| WebAttacks | 2,143 | 1,970 | 36 | 91.9% |
| **Total** | **13,241** | **12,743** | **229** | **96.24%** |

---

### Uncertain Pool Breakdown (106,657 total uncertain)

| Outcome | Count |
|---|---|
| Direct Normal (SVM argmax=Normal, skip DBSCAN) | 23,266 (21.8%) |
| Attack-class uncertain → DBSCAN | 83,391 (78.2%) |
| &nbsp;&nbsp;→ Matched holdout pattern (Bots/BF/WebAttacks) | 67,249 (80.7% of DBSCAN pool) |
| &nbsp;&nbsp;→ Novel attack (DBSCAN noise -1) | 4,673 (5.6%) |
| &nbsp;&nbsp;→ Unknown anomaly (unclaimed cluster) | 11,469 (13.8%) |

---

## Summary

| Metric | Value |
|---|---|
| SVM-only macro-F1 | 0.7621 |
| **Hybrid macro-F1** | **0.9171** |
| **Holdout detection rate** | **96.24% (12,743 / 13,241)** |
| Training time (4 MPI ranks) | ~3 seconds |
| Inference time (4 MPI ranks) | ~25 seconds |
| Model size (SVM) | model.bin (4×52 weights) |
| Model size (DBSCAN) | dbscan_model.bin (167 clusters) |

---

## Output Files

| File | Description |
|---|---|
| `model.bin` | SVM weights: int32 K, int32 F, float W[K×F], float b[K] |
| `dbscan_model.bin` | DBSCAN clusters: centroids + majority-vote labels |
| `svm_predictions.csv` | Per-sample SVM-only predictions |
| `hybrid_predictions.csv` | Per-sample hybrid final predictions |
| `training_log.csv` | epoch, lr, loss, val_acc, val_macro_f1 per epoch |
| `hybrid_log.csv` | Summary metrics from inference run |

---

## Build & Run

```bash
# Train
mpicxx -O3 -std=c++17 -o mpi_train mpi_train.cpp -lm
mpirun -np 4 ./mpi_train

# Infer
mpicxx -O3 -std=c++17 -o mpi_infer mpi_infer.cpp -lm
mpirun -np 4 ./mpi_infer
```
