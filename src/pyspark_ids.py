import numpy as np
import time
import sys
import os
import pickle

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, monotonically_increasing_id
    from pyspark.ml.feature import VectorAssembler
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

# ===================== SVM (Optimized with Vectorization) =====================
class SimpleSVM:
    def __init__(self, n_classes=5, gamma=0.1, max_iter=50, lr=0.01):
        self.n_classes = n_classes
        self.gamma = gamma
        self.max_iter = max_iter
        self.lr = lr
        self.models = []

    def _rbf_kernel_matrix(self, X, Y):
        # คำนวณ RBF Kernel แบบ Vectorized (Matrix Operation)
        # เร็วกว่าการวนลูปทีละตัวมหาศาล
        X_norm = np.sum(X**2, axis=-1)
        Y_norm = np.sum(Y**2, axis=-1)
        dist = X_norm[:, None] + Y_norm[None, :] - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * dist)

    def _train_binary(self, X, y_binary):
        n = len(X)
        alphas = np.zeros(n)
        bias = 0.0
        
        # Pre-compute Kernel Matrix เพื่อลดการคำนวณซ้ำในลูป
        K = self._rbf_kernel_matrix(X, X)
        
        for epoch in range(self.max_iter):
            # คำนวณ Score ของทุกจุดพร้อมกันด้วย Matrix Multiplication
            scores = np.dot(K, alphas) + bias
            # หาจุดที่ทำนายผิด (Violation)
            margins = y_binary * scores
            misclassified = margins <= 0
            
            if not np.any(misclassified):
                break
                
            # Update weights เฉพาะจุดที่ผิด
            alphas[misclassified] += self.lr * y_binary[misclassified]
            bias += self.lr * np.sum(y_binary[misclassified])
            
        return X.copy(), alphas, bias

    def train(self, X, labels):
        self.models = []
        X = np.array(X)
        labels = np.array(labels)
        flops = 0
        D = X.shape[1]
        
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                mask = (labels == i) | (labels == j)
                if not np.any(mask): continue
                
                X_sub = X[mask]
                y_sub = np.where(labels[mask] == i, 1.0, -1.0)
                
                sv, alphas, bias = self._train_binary(X_sub, y_sub)
                self.models.append((sv, alphas, bias, i, j))
                
                # คำนวณ FLOP Count แบบ Vectorized
                n_sv = len(sv)
                flops += self.max_iter * (n_sv * n_sv * D) 
        return flops

# ===================== DBSCAN (NumPy) =====================
def dbscan_numpy(data, eps=1.5, min_pts=35):
    from sklearn.cluster import DBSCAN # ใช้ sklearn ถ้ามี เพื่อความเร็วบน Worker
    db = DBSCAN(eps=eps, min_samples=min_pts).fit(data)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    # ประมาณการ FLOP สำหรับ DBSCAN
    flops = len(data)**2 * data.shape[1]
    return labels, n_clusters, n_noise, flops

# ===================== MAIN PIPELINE =====================
def run_spark_pipeline(n_workers=3):
    print(f"=== Optimized Network IDS: PySpark ({n_workers} Workers) ===")
    total_start = time.time()
    
    spark = SparkSession.builder.appName("NetworkIDS-Optimized").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    # Path ข้อมูล (ใช้ Full Path ตามที่ตั้งค่าไว้)
    base_path = "/root/HPCE-Term-Project---Network-Intrusion-Detection-System/scripts/data"
    
    # Stage 1: Load Data
    train_df = spark.read.csv(f"{base_path}/train_data.csv", inferSchema=True)
    test_df = spark.read.csv(f"{base_path}/test_data.csv", inferSchema=True)
    
    assembler = VectorAssembler(inputCols=train_df.columns, outputCol="features")
    
    # ดึงข้อมูลมาเป็น NumPy (ระวัง OOM ถ้าข้อมูลใหญ่มาก)
    train_data = np.array([list(r.features) for r in assembler.transform(train_df).select("features").collect()])
    test_data = np.array([list(r.features) for r in assembler.transform(test_df).select("features").collect()])
    train_labels = np.loadtxt(f"{base_path}/train_labels.csv", dtype=int)
    test_labels = np.loadtxt(f"{base_path}/test_labels.csv", dtype=int)

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    # เลือกมา 30,000 แถวที่คละกันแล้ว
    train_data = train_data[indices[:30000]]
    train_labels = train_labels[indices[:30000]]

    # ส่วน Test เอามาสัก 50,000 แถวเพื่อโชว์พลัง Cluster
    test_data = test_data[:50000]
    test_labels = test_labels[:50000]

    print(f"Optimized Test Run: Train={len(train_data)}, Test={len(test_data)}") 

    n_classes = int(max(train_labels)) + 1
    D = train_data.shape[1]

    # Stage 2: Train SVM (Master)
    svm = SimpleSVM(n_classes=n_classes, max_iter=50)
    print("Training SVM (Vectorized)...")
    t_train = time.time()
    train_flops = svm.train(train_data, train_labels)
    print(f"Training finished in {time.time() - t_train:.2f}s")

    with open('nids_svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
    print("Model saved to nids_svm_model.pkl")

    # Stage 2: Predict (Distributed)
    sc = spark.sparkContext
    model_data = [(m[0].tolist(), m[1].tolist(), m[2], m[3], m[4]) for m in svm.models]
    b_model = sc.broadcast(model_data)
    b_gamma = sc.broadcast(svm.gamma)
    
    test_rdd = sc.parallelize(list(enumerate(test_data.tolist())), numSlices=n_workers * 2)

    def predict_partition(iterator):
        models = b_model.value
        gamma = b_gamma.value
        results = []
        for idx, x in iterator:
            x_arr = np.array(x)
            votes = np.zeros(n_classes)
            scores_sum = np.zeros(n_classes)
            for sv, alphas, bias, ca, cb in models:
                sv_arr = np.array(sv)
                al_arr = np.array(alphas)
                # Vectorized RBF for single point vs all SVs
                dists = np.sum((sv_arr - x_arr)**2, axis=1)
                kernel_vals = np.exp(-gamma * dists)
                score = np.dot(kernel_vals, al_arr) + bias
                p = ca if score >= 0 else cb
                votes[p] += 1
                scores_sum[p] += abs(score)
            best = int(np.argmax(votes))
            results.append((idx, best, float(scores_sum[best]/max(1, votes[best]))))
        return iter(results)

    print("Distributing Prediction to Workers...")
    results = sorted(test_rdd.mapPartitions(predict_partition).collect(), key=lambda x: x[0])
    predictions = [r[1] for r in results]
    confidences = [r[2] for r in results]

    # Stage 3: DBSCAN
    uncertain_data = np.array([test_data[i] for i in range(len(test_data)) if confidences[i] < 0.3])
    total_flops = train_flops + (len(test_data) * len(train_data) * D) # ประมาณการกว้างๆ
    
    if len(uncertain_data) > 1:
        _, _, _, db_flops = dbscan_numpy(uncertain_data)
        total_flops += db_flops

    total_time = (time.time() - total_start) * 1000
    print(f"\n{'='*40}")
    print(f"Total Time: {total_time:.1f} ms")
    print(f"Final GFLOPS: {(total_flops/1e9)/(total_time/1000):.4f}")
    print(f"{'='*40}")
    
    spark.stop()

if __name__ == "__main__":
    run_spark_pipeline(n_workers=3)
