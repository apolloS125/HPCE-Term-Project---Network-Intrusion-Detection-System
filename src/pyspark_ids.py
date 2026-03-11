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

    def _rbf_kernel_matrix(self, X, Y):
        X_norm = np.sum(X**2, axis=-1)
        Y_norm = np.sum(Y**2, axis=-1)
        dist = X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * np.maximum(dist, 0))

    def _train_binary(self, X, y_binary, ca, cb):
        n = len(X)
        alphas = np.zeros(n)
        bias = 0.0

        # Pre-compute kernel matrix on numpy (vectorized)
        K = self._rbf_kernel_matrix(X, X)

        for epoch in range(self.max_iter):
            scores = K @ alphas + bias
            margins = y_binary * scores
            misclassified = margins <= 0

            errors = int(np.sum(misclassified))
            print(f"  SVM({ca}v{cb}) Epoch {epoch+1}/{self.max_iter} | Errors: {errors}/{n}")

            if errors == 0:
                break

            alphas[misclassified] += self.lr * y_binary[misclassified]
            bias += self.lr * np.sum(y_binary[misclassified])

        return X.copy(), alphas, bias

    def train(self, X, labels):
        self.models = []
        X = np.array(X, dtype=np.float64)
        labels = np.array(labels)
        flops = 0
        D = X.shape[1]

        # With 1.76M samples, kernel matrix 5K^2 * 8B = 200MB per pair — keep cap
        max_samples = 5000

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                mask = (labels == i) | (labels == j)
                if not np.any(mask):
                    continue

                X_sub = X[mask]
                y_sub = np.where(labels[mask] == i, 1.0, -1.0)

                # Subsample support vectors only (not test/inference data)
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

# ===================== Data Loading =====================
def load_csv(filename):
    return np.loadtxt(filename, delimiter=',', dtype=np.float64)

def load_labels(filename):
    return np.loadtxt(filename, dtype=int)

# ===================== DBSCAN (sklearn) =====================
def dbscan_numpy(data, eps=1.5, min_pts=35):
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
        labels = db.labels_
    except ImportError:
        labels = np.full(len(data), -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    N, D = data.shape
    flops = N * (N - 1) // 2 * (3 * D + 1)
    return labels, n_clusters, n_noise, flops

# ===================== Save Functions =====================
def save_svm_model(svm, filename):
    with open(filename, 'wb') as f:
        pickle.dump(svm, f)
    print(f"Model saved to {filename}")

def save_predictions(preds, filename):
    np.savetxt(filename, preds, fmt='%d')
    print(f"Predictions saved to {filename}")

def save_dbscan_model(labels, data, eps, min_pts, filename):
    N, D = data.shape
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(np.array(labels) == -1))
    with open(filename, 'w') as f:
        f.write(f"{eps} {min_pts}\n")
        f.write(f"{n_clusters} {n_noise}\n")
        f.write(f"{N} {D}\n")
        for i in range(N):
            row = str(labels[i]) + ' ' + ' '.join(map(str, data[i]))
            f.write(row + '\n')
    print(f"DBSCAN model saved to {filename}")

# ===================== MAIN PIPELINE =====================
def run_spark_pipeline(n_workers=4):
    print(f"=== Network IDS: PySpark ({n_workers} Workers) — FULL DATA ===")
    total_start = time.time()

    # Stage 1: Load Full Data
    print("\n--- Loading Data ---")
    train_data   = load_csv("/root/HPCE-Term-Project---Network-Intrusion-Detection-System/scripts/data/train_data.csv")
    train_labels = load_labels("/root/HPCE-Term-Project---Network-Intrusion-Detection-System/scripts/data/train_labels.csv")
    test_data    = load_csv("/root/HPCE-Term-Project---Network-Intrusion-Detection-System/scripts/data/test_data.csv")
    test_labels  = load_labels("/root/HPCE-Term-Project---Network-Intrusion-Detection-System/scripts/data/test_labels.csv")

    # NO subsampling — use full 1.76M train / 756K test
    N_train, D = train_data.shape
    N_test = len(test_data)
    n_classes = int(np.max(train_labels)) + 1
    print(f"Train: {N_train} | Test: {N_test} | Features: {D} | Classes: {n_classes}")

    # Stage 2: Train SVM on Driver (vectorized numpy)
    # Note: SVM binary subsamples to 5K support vectors per pair to keep
    # kernel matrix ≤ 200MB (5K^2 * 8B). Full train data feeds class distribution.
    print("\n--- SVM Training ---")
    svm = SimpleSVM(n_classes=n_classes, gamma=0.1, max_iter=200, lr=0.01)
    t_train = time.time()
    train_flops = svm.train(train_data, train_labels)
    svm_train_time = time.time() - t_train
    print(f"SVM training: {svm_train_time*1000:.1f} ms")

    # Stage 3: Distributed Prediction via Spark
    # Tuned for: master 8GB/6-core + 3 workers x 4GB/2-core
    # Increase partitions to 60 to spread 756K test samples (12.6K per partition)
    print("\n--- SVM Prediction (Distributed) ---")
    spark = SparkSession.builder \
        .appName("NetworkIDS") \
        .master("spark://Spark-Master:7077") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.cores.max", "6") \
        .config("spark.sql.shuffle.partitions", "60") \
        .config("spark.default.parallelism", "60") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    sc = spark.sparkContext
    model_data = [(m[0].tolist(), m[1].tolist(), float(m[2]), int(m[3]), int(m[4])) for m in svm.models]
    b_model = sc.broadcast(model_data)
    b_gamma = sc.broadcast(float(svm.gamma))
    b_nclasses = sc.broadcast(n_classes)

    # Use 60 partitions for 756K samples (~12.6K per partition)
    test_rdd = sc.parallelize(list(enumerate(test_data.tolist())), numSlices=60)

    def predict_partition(iterator):
        import numpy as _np
        models = b_model.value
        gamma = b_gamma.value
        nc = b_nclasses.value
        results = []
        for idx, x in iterator:
            x_arr = _np.array(x)
            votes = _np.zeros(nc)
            scores_sum = _np.zeros(nc)
            for sv, alphas, bias, ca, cb in models:
                sv_arr = _np.array(sv)
                al_arr = _np.array(alphas)
                dists = _np.sum((sv_arr - x_arr)**2, axis=1)
                kernel_vals = _np.exp(-gamma * dists)
                score = _np.dot(kernel_vals, al_arr) + bias
                p = ca if score >= 0 else cb
                votes[p] += 1
                scores_sum[p] += abs(score)
            best = int(_np.argmax(votes))
            conf = float(scores_sum[best] / max(1, votes[best]))
            results.append((idx, best, conf))
        return iter(results)

    t_pred = time.time()
    raw_results = sorted(test_rdd.mapPartitions(predict_partition).collect(), key=lambda x: x[0])
    svm_pred_time = time.time() - t_pred
    predictions = [r[1] for r in raw_results]
    confidences  = [r[2] for r in raw_results]

    total_sv = sum(len(m[0]) for m in svm.models)
    pred_flops = N_test * total_sv * (3 * D + 2)
    print(f"SVM prediction: {svm_pred_time*1000:.1f} ms")

    # Stage 4: DBSCAN on uncertain samples
    confidence_threshold = 0.05
    final_preds = list(predictions)
    unc_idx  = [i for i in range(N_test) if confidences[i] < confidence_threshold]
    unc_data = test_data[unc_idx]

    confident_count = N_test - len(unc_idx)
    print(f"Confident: {confident_count} | Uncertain: {len(unc_idx)}")

    dbscan_flops = 0
    dbscan_time = 0.0
    if len(unc_data) > MAX_DBSCAN:
        unc_data = unc_data[:MAX_DBSCAN]
        unc_idx  = unc_idx[:MAX_DBSCAN]

    if len(unc_data) > 1:
        print(f"\n--- DBSCAN on {len(unc_data)} samples ---")
        t_db = time.time()
        db_labels, n_clusters, n_noise, dbscan_flops = dbscan_numpy(unc_data, eps=1.5, min_pts=D+1)
        dbscan_time = time.time() - t_db
        print(f"DBSCAN: {dbscan_time*1000:.1f} ms | Clusters: {n_clusters} | Noise: {n_noise}")

        for i, idx in enumerate(unc_idx):
            final_preds[idx] = -1 if db_labels[i] == -1 else predictions[idx]

        save_dbscan_model(db_labels, unc_data, 1.5, D+1, "pyspark_dbscan_model.txt")

    spark.stop()

    total_time = time.time() - total_start
    total_flops = train_flops + pred_flops + dbscan_flops

    # Results
    final_arr = np.array(final_preds)
    classified_mask = final_arr >= 0
    correct = int(np.sum(final_arr[classified_mask] == test_labels[classified_mask]))
    classified = int(np.sum(classified_mask))
    acc = 100.0 * correct / classified if classified > 0 else 0.0

    print(f"\nAccuracy: {acc:.2f}%")
    print(f"Unknown attacks: {N_test - classified}")

    gflops = (total_flops / 1e9) / total_time
    print(f"\n{'='*40}")
    print(f"  Technique:  PySpark ({n_workers} workers)")
    print(f"  Samples:    {N_test}")
    print(f"  Features:   {D}")
    print(f"  Time:       {total_time*1000:.3f} ms")
    print(f"  FLOP Count: {total_flops}")
    print(f"  GFLOPS:     {gflops:.4f}")
    print(f"{'='*40}")
    print(f"Timing: Train={svm_train_time*1000:.1f}ms Predict={svm_pred_time*1000:.1f}ms DBSCAN={dbscan_time*1000:.1f}ms Total={total_time*1000:.1f}ms")

    save_svm_model(svm, "pyspark_svm_model.pkl")
    save_predictions(np.array(final_preds), "pyspark_predictions.csv")

if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_spark_pipeline(n_workers=n_workers)
