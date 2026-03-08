#!/usr/bin/env python3
"""
PySpark implementation of Network IDS pipeline.
- Stage 1: Distributed preprocessing with Spark DataFrame
- Stage 2: SVM prediction with broadcast model + mapPartitions
- Stage 3: DBSCAN on uncertain subset (local - not distributed)

Usage:
    spark-submit --master local[4] pyspark_ids.py [N_WORKERS]
    # or simply:
    python pyspark_ids.py
"""
import numpy as np
import time
import sys
import os

# Try PySpark import
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, udf, monotonically_increasing_id
    from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, ArrayType
    from pyspark.ml.feature import VectorAssembler, MinMaxScaler
    from pyspark.ml import Pipeline
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False
    print("WARNING: PySpark not installed. Running in simulation mode.")
    print("Install: pip install pyspark")

# ===================== SVM (NumPy implementation) =====================
class SimpleSVM:
    """Simplified multi-class SVM with RBF kernel (1-vs-1)"""
    def __init__(self, n_classes=5, gamma=0.1, max_iter=50, lr=0.01):
        self.n_classes = n_classes
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.models = []

    def _rbf(self, x, y):
        d = x - y
        return np.exp(-self.gamma * np.dot(d, d))

    def _train_binary(self, X, y_binary):
        n = len(X)
        alphas = np.zeros(n)
        bias = 0.0
        for _ in range(self.max_iter):
            errors = 0
            for i in range(n):
                score = bias
                for j in range(n):
                    if alphas[j] != 0:
                        score += alphas[j] * self._rbf(X[j], X[i])
                if y_binary[i] * score <= 0:
                    alphas[i] += self.lr * y_binary[i]
                    bias += self.lr * y_binary[i]
                    errors += 1
            if errors == 0:
                break
        return X.copy(), alphas, bias

    def train(self, X, labels):
        self.models = []
        X = np.array(X)
        labels = np.array(labels)
        flops = 0
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                mask = (labels == i) | (labels == j)
                X_sub = X[mask]
                y_sub = np.where(labels[mask] == i, 1.0, -1.0)
                sv, alphas, bias = self._train_binary(X_sub, y_sub)
                self.models.append((sv, alphas, bias, i, j))
                n_sv = len(sv)
                D = X.shape[1]
                flops += self.max_iter * n_sv * n_sv * (3 * D + 2)
        return flops

    def predict_one(self, x):
        votes = np.zeros(self.n_classes)
        scores = np.zeros(self.n_classes)
        for sv, alphas, bias, ca, cb in self.models:
            score = bias
            for j in range(len(sv)):
                if alphas[j] != 0:
                    score += alphas[j] * self._rbf(sv[j], x)
            pred = ca if score >= 0 else cb
            votes[pred] += 1
            scores[pred] += abs(score)
        best = np.argmax(votes)
        conf = scores[best] / max(1, votes[best])
        return int(best), float(conf)

    def predict_batch(self, X):
        preds, confs = [], []
        for x in X:
            p, c = self.predict_one(x)
            preds.append(p)
            confs.append(c)
        return preds, confs

# ===================== DBSCAN (NumPy) =====================
def dbscan_numpy(data, eps=1.5, min_pts=35):
    N = len(data)
    D = data.shape[1] if hasattr(data, 'shape') else len(data[0])
    data = np.array(data)

    # Pairwise distance matrix
    dist = np.zeros((N, N))
    flops = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((data[i] - data[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d
    flops = N * (N - 1) // 2 * (3 * D + 1)

    labels = np.full(N, -2)  # -2 = unvisited
    cluster_id = 0

    for i in range(N):
        if labels[i] != -2:
            continue
        neighbors = np.where(dist[i] <= eps)[0].tolist()
        if len(neighbors) < min_pts:
            labels[i] = -1  # noise
            continue
        labels[i] = cluster_id
        seed = list(neighbors)
        idx = 0
        while idx < len(seed):
            q = seed[idx]
            if labels[q] == -1:
                labels[q] = cluster_id
            if labels[q] != -2:
                idx += 1
                continue
            labels[q] = cluster_id
            q_neighbors = np.where(dist[q] <= eps)[0].tolist()
            if len(q_neighbors) >= min_pts:
                for nn in q_neighbors:
                    if nn not in seed:
                        seed.append(nn)
            idx += 1
        cluster_id += 1

    n_noise = int(np.sum(labels == -1))
    return labels, cluster_id, n_noise, flops

# ===================== MAIN PIPELINE =====================
def run_spark_pipeline(n_workers=4):
    print(f"=== Network IDS: PySpark ({n_workers} workers) ===")

    total_start = time.time()
    total_flops = 0

    if HAS_SPARK:
        # ===== Stage 1: Spark Preprocessing =====
        print("\n--- Stage 1: Spark Preprocessing ---")
        spark = SparkSession.builder \
            .appName("NetworkIDS") \
            .master(f"local[{n_workers}]") \
            .config("spark.driver.memory", "2g") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

        t1 = time.time()

        # Load data as Spark DataFrames
        train_df = spark.read.csv("data/train_data.csv", inferSchema=True)
        test_df = spark.read.csv("data/test_data.csv", inferSchema=True)

        # Get column names
        feature_cols = train_df.columns

        # Add ID column
        train_df = train_df.withColumn("id", monotonically_increasing_id())
        test_df = test_df.withColumn("id", monotonically_increasing_id())

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        train_assembled = assembler.transform(train_df).select("id", "features")
        test_assembled = assembler.transform(test_df).select("id", "features")

        # Collect back to numpy (for SVM - Spark MLlib SVM is limited)
        train_rows = train_assembled.collect()
        test_rows = test_assembled.collect()

        train_data = np.array([list(r.features) for r in train_rows])
        test_data = np.array([list(r.features) for r in test_rows])

        preprocess_time = (time.time() - t1) * 1000
        print(f"Spark preprocessing: {preprocess_time:.1f} ms")
        print(f"Train: {len(train_data)} | Test: {len(test_data)} | Features: {train_data.shape[1]}")

    else:
        # Fallback: load with numpy
        print("\n--- Stage 1: NumPy Preprocessing (PySpark not available) ---")
        t1 = time.time()
        train_data = np.loadtxt("data/train_data.csv", delimiter=",")
        test_data = np.loadtxt("data/test_data.csv", delimiter=",")
        preprocess_time = (time.time() - t1) * 1000
        print(f"NumPy load: {preprocess_time:.1f} ms")

    train_labels = np.loadtxt("data/train_labels.csv", dtype=int)
    test_labels = np.loadtxt("data/test_labels.csv", dtype=int)
    N_test = len(test_data)
    D = train_data.shape[1]
    n_classes = int(max(train_labels)) + 1
    print(f"Classes: {n_classes}")

    # ===== Stage 2: SVM =====
    print("\n--- Stage 2: SVM Training & Prediction ---")
    svm = SimpleSVM(n_classes=n_classes, gamma=0.1, max_iter=50, lr=0.01)

    t2 = time.time()
    train_flops = svm.train(train_data, train_labels)
    svm_train_time = (time.time() - t2) * 1000
    total_flops += train_flops
    print(f"SVM training: {svm_train_time:.1f} ms")

    t3 = time.time()

    if HAS_SPARK:
        # Broadcast model and use mapPartitions for prediction
        sc = spark.sparkContext

        # Serialize model data for broadcast
        model_data = []
        for sv, alphas, bias, ca, cb in svm.models:
            model_data.append((sv.tolist(), alphas.tolist(), bias, ca, cb))

        broadcast_model = sc.broadcast(model_data)
        broadcast_gamma = sc.broadcast(svm.gamma)
        broadcast_n_classes = sc.broadcast(n_classes)

        # Create RDD of test data
        test_rdd = sc.parallelize(list(enumerate(test_data.tolist())), numSlices=n_workers)

        def predict_partition(iterator):
            models = broadcast_model.value
            gamma = broadcast_gamma.value
            n_classes = broadcast_n_classes.value

            def rbf(x, y):
                d = np.array(x) - np.array(y)
                return np.exp(-gamma * np.dot(d, d))

            results = []
            for idx, x in iterator:
                x = np.array(x)
                votes = np.zeros(n_classes)
                scores = np.zeros(n_classes)
                for sv, alphas, bias, ca, cb in models:
                    score = bias
                    for j in range(len(sv)):
                        if alphas[j] != 0:
                            score += alphas[j] * rbf(np.array(sv[j]), x)
                    pred = ca if score >= 0 else cb
                    votes[pred] += 1
                    scores[pred] += abs(score)
                best = int(np.argmax(votes))
                conf = float(scores[best] / max(1, votes[best]))
                results.append((idx, best, conf))
            return iter(results)

        pred_rdd = test_rdd.mapPartitions(predict_partition)
        pred_results = pred_rdd.collect()

        # Sort by index
        pred_results.sort(key=lambda x: x[0])
        predictions = [r[1] for r in pred_results]
        confidences = [r[2] for r in pred_results]
    else:
        predictions, confidences = svm.predict_batch(test_data)

    svm_pred_time = (time.time() - t3) * 1000

    total_sv = sum(len(m[0]) for m in svm.models)
    total_flops += N_test * total_sv * (3 * D + 2)
    print(f"SVM prediction: {svm_pred_time:.1f} ms")

    # Split confident / uncertain
    threshold = 0.3
    final_predictions = [-1] * N_test
    uncertain_indices = []
    uncertain_data = []

    for i in range(N_test):
        if confidences[i] >= threshold:
            final_predictions[i] = predictions[i]
        else:
            uncertain_indices.append(i)
            uncertain_data.append(test_data[i])

    confident_count = N_test - len(uncertain_indices)
    print(f"Confident: {confident_count} ({100*confident_count/N_test:.1f}%)")
    print(f"Uncertain: {len(uncertain_indices)} -> DBSCAN")

    # ===== Stage 3: DBSCAN =====
    dbscan_time = 0
    if len(uncertain_data) > 1:
        print("\n--- Stage 3: DBSCAN Anomaly Detection ---")
        t4 = time.time()
        uncertain_arr = np.array(uncertain_data)
        labels, n_clusters, n_noise, dbscan_flops = dbscan_numpy(uncertain_arr, eps=1.5, min_pts=D+1)
        dbscan_time = (time.time() - t4) * 1000
        total_flops += dbscan_flops
        print(f"DBSCAN: {dbscan_time:.1f} ms | Clusters: {n_clusters} | Noise: {n_noise}")

        for i, idx in enumerate(uncertain_indices):
            final_predictions[idx] = -1 if labels[i] == -1 else predictions[idx]

    total_time = (time.time() - total_start) * 1000

    # ===== Results =====
    print("\n--- Results ---")
    correct = sum(1 for i in range(N_test)
                  if final_predictions[i] >= 0 and final_predictions[i] == test_labels[i])
    classified = sum(1 for p in final_predictions if p >= 0)
    acc = 100 * correct / max(1, classified)
    unknown = N_test - classified

    print(f"Classified: {classified}/{N_test}")
    print(f"Unknown attacks: {unknown}")
    print(f"Accuracy: {acc:.2f}%")

    total_time_sec = total_time / 1000
    gflops = (total_flops / 1e9) / total_time_sec if total_time_sec > 0 else 0

    print(f"\n{'='*40}")
    print(f"  Technique:  PySpark ({n_workers} workers)")
    print(f"  Samples:    {N_test}")
    print(f"  Features:   {D}")
    print(f"  Time:       {total_time:.1f} ms")
    print(f"  FLOP Count: {total_flops}")
    print(f"  GFLOPS:     {gflops:.4f}")
    print(f"{'='*40}")
    print(f"\nTiming: Preprocess={preprocess_time:.1f}ms Train={svm_train_time:.1f}ms "
          f"Predict={svm_pred_time:.1f}ms DBSCAN={dbscan_time:.1f}ms Total={total_time:.1f}ms")

    if HAS_SPARK:
        spark.stop()

if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_spark_pipeline(n_workers)
