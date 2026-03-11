import numpy as np
import time
import sys
import os
import pickle

try:
    from pyspark.sql import SparkSession
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

# ===================== Constants =====================
MAX_DBSCAN = 2000  # Cap uncertain samples for DBSCAN

# ===================== SVM (Vectorized NumPy) =====================
class SimpleSVM:
    def __init__(self, n_classes=7, gamma=0.1, max_iter=200, lr=0.01):
        self.n_classes = n_classes
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.models = []
        self.epoch_logs = []  # [(ca, cb, epoch, train_err, n_train, val_err, n_val), ...]

    def _rbf_kernel_matrix(self, X, Y):
        X_norm = np.sum(X**2, axis=-1)
        Y_norm = np.sum(Y**2, axis=-1)
        dist = X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * np.maximum(dist, 0))

    def _train_binary(self, X, y_binary, ca, cb):
        n = len(X)
        # ---- Train/Validation split (80/20) ----
        use_val = (n >= 10)
        n_train = max(2, int(n * 0.8)) if use_val else n
        n_val = n - n_train

        X_train, y_train = X[:n_train], y_binary[:n_train]
        X_val, y_val = X[n_train:], y_binary[n_train:]

        alphas = np.zeros(n_train)
        bias = 0.0

        # Pre-compute kernel matrices
        K_train = self._rbf_kernel_matrix(X_train, X_train)
        K_val = self._rbf_kernel_matrix(X_val, X_train) if use_val else None

        decay = 0.001   # L2 weight decay
        patience = 15
        best_val_errors = n_val + 1
        wait = 0
        best_alphas = alphas.copy()
        best_bias = bias

        for epoch in range(self.max_iter):
            # --- Train phase ---
            scores = K_train @ alphas + bias
            margins = y_train * scores
            misclassified = margins <= 0
            errors = int(np.sum(misclassified))

            alphas[misclassified] += self.lr * y_train[misclassified]
            bias += self.lr * np.sum(y_train[misclassified])

            # --- Weight decay ---
            alphas *= (1.0 - decay)

            # --- Validation phase ---
            val_errors = 0
            if use_val:
                val_scores = K_val @ alphas + bias
                val_margins = y_val * val_scores
                val_errors = int(np.sum(val_margins <= 0))

            msg = f"  SVM({ca}v{cb}) Epoch {epoch+1}/{self.max_iter} | Train Errors: {errors}/{n_train}"
            if use_val:
                msg += f" | Val Errors: {val_errors}/{n_val}"
            print(msg)

            self.epoch_logs.append((ca, cb, epoch+1, errors, n_train, val_errors, n_val))

            # --- Early stopping ---
            if use_val:
                if val_errors < best_val_errors:
                    best_val_errors = val_errors
                    best_alphas = alphas.copy()
                    best_bias = bias
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"  Early stopping at epoch {epoch+1} (best val errors={best_val_errors})")
                        alphas = best_alphas
                        bias = best_bias
                        break
            else:
                if errors == 0:
                    break

        return X_train.copy(), alphas, bias

    def train(self, X, labels):
        self.models = []
        X = np.array(X, dtype=np.float64)
        labels = np.array(labels)
        flops = 0
        D = X.shape[1]
        # kernel matrix: max_samples^2 * 8 bytes on driver RAM
        # 5000^2 * 8 = 200MB per pair — safe on 8GB master
        max_samples = 5000

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                mask = (labels == i) | (labels == j)
                if not np.any(mask):
                    continue

                X_sub = X[mask]
                y_sub = np.where(labels[mask] == i, 1.0, -1.0)

                # Subsample if too many
                if len(X_sub) > max_samples:
                    step = len(X_sub) // max_samples
                    idx = np.arange(0, len(X_sub), step)[:max_samples]
                    X_sub = X_sub[idx]
                    y_sub = y_sub[idx]

                sv, alphas, bias = self._train_binary(X_sub, y_sub, i, j)
                self.models.append((sv, alphas, bias, i, j))

                n_sv = len(sv)
                flops += self.max_iter * n_sv * n_sv * (3 * D + 2)

        return flops

# ...existing code...

def run_spark_pipeline(n_workers=4):
    # ...existing code for data loading, SVM training, prediction, DBSCAN...

    save_svm_model(svm, "pyspark_svm_model.pkl")
    save_predictions(np.array(final_preds), "pyspark_predictions.csv")

    # ---- Overfitting analysis: save epoch errors + train accuracy ----
    # Save epoch errors CSV
    with open("pyspark_epoch_errors.csv", "w") as f:
        f.write("class_a,class_b,epoch,train_errors,n_train,val_errors,n_val\n")
        for row in svm.epoch_logs:
            f.write(",".join(str(x) for x in row) + "\n")
    print("Epoch errors saved to pyspark_epoch_errors.csv")

    # Compute train accuracy (subsample 10K)
    max_eval = 10000
    step_eval = max(1, len(train_data) // max_eval)
    idx_eval = np.arange(0, len(train_data), step_eval)[:max_eval]
    sample_data = train_data[idx_eval]
    sample_labels = train_labels[idx_eval]

    # Predict on training subsample using 1-vs-1 voting
    train_preds = np.zeros(len(sample_data), dtype=int)
    for si in range(len(sample_data)):
        votes = np.zeros(n_classes, dtype=int)
        for sv_x, alphas_m, bias_m, ci, cj in svm.models:
            K_row = svm._rbf_kernel_matrix(sample_data[si:si+1], sv_x)
            score = float(K_row @ alphas_m + bias_m)
            pred = ci if score >= 0 else cj
            votes[pred] += 1
        train_preds[si] = int(np.argmax(votes))

    train_acc = 100.0 * np.sum(train_preds == sample_labels) / len(sample_labels)
    print(f"Train Accuracy: {train_acc:.2f}%")

    with open("pyspark_accuracy.csv", "w") as f:
        f.write(f"{train_acc:.4f},{acc:.4f}\n")

if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_spark_pipeline(n_workers=n_workers)