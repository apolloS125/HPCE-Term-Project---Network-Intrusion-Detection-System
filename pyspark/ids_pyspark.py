#!/usr/bin/env python3
"""
Network Intrusion Detection System — PySpark (Fast Version)
============================================================
Speed improvements:
  1. SVM Training  — parallelize 6 binary pairs across workers (6x faster)
  2. SVM Predict   — distributed via Spark RDD (unchanged)
  3. DBSCAN        — runs on benign-predicted samples only

Pipeline:
  Load data
    → Spark: train 6 binary pairs in parallel (1 pair/core)
    → Spark: predict 756K test samples in parallel
    → SVM = Known Attack  → label directly
      SVM = Benign        → DBSCAN
                              cluster >= 0 → Unknown Attack (-1)
                              noise  == -1 → Normal Traffic (-2)
    → Metrics + GFLOPS

Usage:
    spark-submit --master spark://Spark-Master:7077 \
                 --driver-memory 8g \
                 --executor-memory 4g \
                 --total-executor-cores 6 \
                 ids_pyspark.py [--test]
"""

import numpy as np
import time, sys, os, pickle, argparse

try:
    from pyspark.sql import SparkSession
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

# ===================== Constants =====================
DATA_DIR           = "/root/project/scripts/data/processed"
GAMMA              = 0.1
LR                 = 0.01    # unused (kept for FLOP ref)
BATCH_SIZE         = 500     # unused (kept for FLOP ref)
MAX_SV             = 2000    # unused (kept for FLOP ref)
MAX_ITER           = 5000    # sklearn SVC max_iter
MAX_TRAIN_PER_PAIR = 5000    # samples per class per pair
MAX_DBSCAN         = 5000    # cap benign samples for DBSCAN
DBSCAN_EPS         = 0.2
DBSCAN_MIN_PTS     = 20


# ===================== Config loader =====================
def load_holdout_config(data_dir):
    cfg = {}
    with open(os.path.join(data_dir, "holdout_config.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()

    # รองรับทั้ง format เก่าและใหม่
    holdout_ids  = list(map(int, cfg["holdout_class_ids"].split(",")))
    svm_ids      = list(map(int, cfg["svm_class_ids"].split(",")))
    holdout_names = cfg.get("holdout_class_names", "").split(",")

    return {
        "holdout_ids":       holdout_ids,
        "holdout_names":     holdout_names,
        "svm_class_ids":     svm_ids,
        # optional fields — ใช้ default ถ้าไม่มี
        "n_classes_svm":     int(cfg.get("n_classes_svm",  len(svm_ids))),
        "n_features":        int(cfg.get("n_features",     0)),
        "n_train":           int(cfg.get("n_train",        cfg.get("n_train_raw", 0))),
        "n_test":            int(cfg.get("n_test",         0)),
        "n_holdout_in_test": int(cfg.get("n_holdout_in_test", 0)),
    }

def load_label_map(data_dir):
    lm = {}
    path = os.path.join(data_dir, "label_mapping.txt")
    if not os.path.exists(path):
        return lm
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 1)
            if len(parts) == 2:
                lm[int(parts[0])] = parts[1].strip().replace(" [HOLDOUT]", "")
    return lm


# ===================== Data Loading =====================
def load_csv(f):    return np.loadtxt(f, delimiter=",", dtype=np.float64)
def load_labels(f): return np.loadtxt(f, dtype=int)


# ===================== Binary SVM (sklearn SVC on Worker) =====================
def train_binary_pair(task):
    """
    รันบน Worker — train 1 binary SVM pair ด้วย sklearn SVC
    task = (ca, cb, X_list, y_list, gamma, max_iter)
    returns (ca, cb, sv_list, alpha_list, bias, n_flops)
    """
    import numpy as _np
    from sklearn.svm import SVC

    ca, cb, X_list, y_list, gamma, max_iter = task
    X = _np.array(X_list, dtype=_np.float64)
    y = _np.array(y_list, dtype=_np.float64)
    n, D = X.shape

    clf = SVC(kernel='rbf', gamma=gamma, C=10.0,   # C=10 แทน 1.0
              max_iter=max_iter, cache_size=500,
              probability=False)
    clf.fit(X, y)

    sv_arr = clf.support_vectors_
    al_arr = clf.dual_coef_.ravel()
    bias   = float(clf.intercept_[0])
    n_sv   = len(sv_arr)
    flops  = n_sv * n_sv * (3*D + 2)
    return (ca, cb, sv_arr.tolist(), al_arr.tolist(), bias, flops)


# ===================== DBSCAN =====================
def run_dbscan(data, eps=1.5, min_pts=None):
    if min_pts is None:
        min_pts = data.shape[1] + 1
    try:
        from sklearn.cluster import DBSCAN
        # ball_tree เร็วกว่า brute force สำหรับ large dataset
        algo = 'ball_tree' if len(data) > 10000 else 'auto'
        labels = DBSCAN(eps=eps, min_samples=min_pts,
                        algorithm=algo, n_jobs=-1).fit(data).labels_
    except ImportError:
        labels = np.full(len(data), -1)
    N, D       = data.shape
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    flops      = N*(N-1)//2 * (3*D + 1)
    return labels, n_clusters, n_noise, flops


# ===================== Metrics =====================
def compute_metrics(y_true, y_pred, svm_class_ids, holdout_ids, label_map):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy — known classes only
    known_mask = np.isin(y_true, svm_class_ids)
    acc = (100.0 * np.sum(y_pred[known_mask] == y_true[known_mask]) / known_mask.sum()
           if known_mask.sum() > 0 else 0.0)

    # DBSCAN detection rate — holdout samples ที่ถูกจับเป็น -1
    holdout_mask = np.isin(y_true, holdout_ids)
    n_holdout    = int(holdout_mask.sum())
    detected     = int(np.sum(y_pred[holdout_mask] == -1))
    det_rate     = 100.0 * detected / n_holdout if n_holdout > 0 else 0.0

    # False alarm rate — normal ที่ถูกบอกว่า unknown
    benign_id = next((cid for cid, name in label_map.items()
                      if "normal" in name.lower()), None)
    if benign_id is not None:
        norm_mask    = y_true == benign_id
        n_normal     = int(norm_mask.sum())
        false_alarms = int(np.sum(y_pred[norm_mask] == -1))
        far          = 100.0 * false_alarms / n_normal if n_normal > 0 else 0.0
    else:
        n_normal, false_alarms, far = 0, 0, 0.0

    # Per-class P/R/F1
    per_class = {}
    for c in svm_class_ids:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        per_class[c] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)

    return dict(accuracy=acc, detection_rate=det_rate, detected=detected,
                n_holdout=n_holdout, false_alarm_rate=far,
                false_alarms=false_alarms, n_normal=n_normal,
                per_class=per_class)


# ===================== Save helpers =====================
def save_dbscan_model(labels, data, eps, min_pts, fname):
    N, D       = data.shape
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(np.array(labels) == -1))
    with open(fname, "w") as f:
        f.write(f"{eps} {min_pts}\n{n_clusters} {n_noise}\n{N} {D}\n")
        for i in range(N):
            f.write(str(labels[i]) + " " + " ".join(map(str, data[i])) + "\n")
    print(f"  DBSCAN model -> {fname}")


# ===================== MAIN PIPELINE =====================
def run_pipeline(test_mode=False, n_workers=3):
    tag = " [TEST MODE]" if test_mode else ""
    print(f"\n{'='*60}")
    print(f"  Network IDS — PySpark Fast{tag}")
    print(f"  SVM training: parallel pairs | Predict: distributed RDD")
    print(f"{'='*60}")
    t_total = time.time()

    # ── Stage 1: Load ────────────────────────────────────────
    print("\n[Stage 1] Loading data...")
    cfg       = load_holdout_config(DATA_DIR)
    label_map = load_label_map(DATA_DIR)

    holdout_ids   = cfg["holdout_ids"]
    holdout_names = cfg["holdout_names"]
    svm_class_ids = cfg["svm_class_ids"]
    n_classes_svm = cfg["n_classes_svm"]

    train_data   = load_csv(os.path.join(DATA_DIR, "train_data.csv"))
    train_labels = load_labels(os.path.join(DATA_DIR, "train_labels.csv"))
    test_data    = load_csv(os.path.join(DATA_DIR, "test_data.csv"))
    test_labels  = load_labels(os.path.join(DATA_DIR, "test_labels.csv"))

    # Test mode subsample
    if test_mode:
        rng = np.random.default_rng(42)
        # train: 5000 samples/class
        tr_idx = np.concatenate([
            rng.choice(np.where(train_labels == c)[0],
                       min(5000, int(np.sum(train_labels == c))), replace=False)
            for c in svm_class_ids])
        train_data, train_labels = train_data[tr_idx], train_labels[tr_idx]
        # test: 500 samples/class รวม holdout
        te_idx = np.concatenate([
            rng.choice(np.where(test_labels == c)[0],
                       min(500, int(np.sum(test_labels == c))), replace=False)
            for c in svm_class_ids + holdout_ids])
        test_data, test_labels = test_data[te_idx], test_labels[te_idx]
        print(f"  TEST MODE: train={len(train_data):,} | test={len(test_data):,}")
        print(f"  Test distribution:")
        for c in svm_class_ids + holdout_ids:
            cnt = int(np.sum(test_labels == c))
            tag = "[HOLDOUT]" if c in holdout_ids else "[SVM]"
            print(f"    class {c} ({label_map.get(c,'?')}) {tag}: {cnt:,}")

    N_train, D = train_data.shape
    N_test     = len(test_data)
    print(f"  Train={N_train:,} | Test={N_test:,} | Features={D}")
    print(f"  SVM classes : {[label_map.get(i,i) for i in svm_class_ids]}")
    print(f"  Holdout     : {holdout_names}")

    # ── Stage 2: Spark init ──────────────────────────────────
    spark = (SparkSession.builder
             .appName("NetworkIDS_PySpark_Fast")
             .config("spark.sql.shuffle.partitions", "60")
             .config("spark.default.parallelism",    "60")
             .config("spark.driver.maxResultSize",   "4g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    # ── Stage 3: Parallel SVM Training ──────────────────────
    # key insight: กระจาย 6 pairs ไปรันบน 6 cores พร้อมกัน
    print("\n[Stage 2] Parallel SVM Training (6 pairs → 6 cores)...")
    pairs = [(svm_class_ids[i], svm_class_ids[j])
             for i in range(len(svm_class_ids))
             for j in range(i+1, len(svm_class_ids))]
    print(f"  {len(pairs)} pairs: {pairs}")

    # เตรียม task list — แต่ละ task คือ 1 binary pair
    tasks = []
    rng_b = np.random.default_rng(42)
    for ca, cb in pairs:
        idx_a = np.where(train_labels == ca)[0]
        idx_b = np.where(train_labels == cb)[0]

        # balance 50/50 — เลือก min(n_per_class, available) จากแต่ละ class
        n_per_class = min(MAX_TRAIN_PER_PAIR // 2, len(idx_a), len(idx_b))
        pick_a = rng_b.choice(idx_a, n_per_class, replace=False)
        pick_b = rng_b.choice(idx_b, n_per_class, replace=False)
        sel    = np.concatenate([pick_a, pick_b])

        X_sub  = train_data[sel]
        y_sub  = np.where(train_labels[sel] == ca, 1.0, -1.0)

        print(f"  Pair ({ca}v{cb}): {n_per_class:,} samples/class "
              f"→ {len(X_sub):,} total")

        tasks.append((ca, cb,
                      X_sub.tolist(), y_sub.tolist(),
                      GAMMA, MAX_ITER))

    t0 = time.time()
    # parallelize tasks — 1 task per partition = 1 pair per core
    tasks_rdd    = sc.parallelize(tasks, numSlices=len(tasks))
    results_list = tasks_rdd.map(train_binary_pair).collect()
    svm_train_t  = time.time() - t0

    # รวมโมเดลจากทุก worker
    models     = []
    train_flops = 0
    for ca, cb, sv_l, al_l, bias, flops in results_list:
        sv  = np.array(sv_l)
        al  = np.array(al_l)
        models.append((sv, al, bias, ca, cb))
        train_flops += flops
        print(f"  Pair ({ca}v{cb}): {len(sv)} support vectors")

    total_sv = sum(len(m[0]) for m in models)
    print(f"  Training done: {svm_train_t*1000:.1f} ms | total_sv={total_sv:,}")

    # ── Stage 4: Distributed Prediction ─────────────────────
    print("\n[Stage 3] Distributed Prediction...")

    model_bc   = sc.broadcast(
        [(m[0].tolist(), m[1].tolist(), float(m[2]), int(m[3]), int(m[4]))
         for m in models])
    gamma_bc   = sc.broadcast(float(GAMMA))
    classid_bc = sc.broadcast(svm_class_ids)

    # เพิ่ม partition ให้ทำงานคู่ขนานมากขึ้น
    n_partitions = max(60, n_workers * 20)
    test_rdd = sc.parallelize(list(enumerate(test_data.tolist())),
                              numSlices=n_partitions)

    def predict_partition(iterator):
        import numpy as _np
        _models    = model_bc.value
        _gamma     = gamma_bc.value
        _class_ids = classid_bc.value

        # รวม batch ทั้ง partition แล้วคำนวณ vectorized
        batch = list(iterator)
        if not batch:
            return iter([])

        idxs  = [b[0] for b in batch]
        X_bat = _np.array([b[1] for b in batch], dtype=_np.float64)  # (N, D)
        N     = len(idxs)

        votes     = {c: _np.zeros(N, dtype=_np.int32)   for c in _class_ids}
        score_sum = {c: _np.zeros(N, dtype=_np.float64) for c in _class_ids}

        for sv, alphas, bias, ca, cb in _models:
            sv_arr = _np.array(sv,     dtype=_np.float64)  # (n_sv, D)
            al_arr = _np.array(alphas, dtype=_np.float64)  # (n_sv,)

            # vectorized: (N, D) vs (n_sv, D) → (N, n_sv)
            X_n  = _np.sum(X_bat**2, axis=1, keepdims=True)   # (N,1)
            S_n  = _np.sum(sv_arr**2, axis=1)                  # (n_sv,)
            dist = X_n + S_n - 2 * X_bat @ sv_arr.T            # (N, n_sv)
            K    = _np.exp(-_gamma * _np.maximum(dist, 0))     # (N, n_sv)
            scores = K @ al_arr + bias                          # (N,)

            pos = scores >= 0
            score_sum[ca] += _np.where(pos,  scores,  0)
            score_sum[cb] += _np.where(~pos, -scores, 0)
            votes[ca] += pos.astype(_np.int32)
            votes[cb] += (~pos).astype(_np.int32)

        # หา best class ต่อ sample
        vote_mat  = _np.stack([votes[c]     for c in _class_ids], axis=1)
        score_mat = _np.stack([score_sum[c] for c in _class_ids], axis=1)
        best_pos  = _np.lexsort((score_mat.T, vote_mat.T))[-1]  # argmax votes then score
        # lexsort คืน index ที่ sort แบบ ascending ดังนั้นเอาตัวสุดท้าย
        best_pos  = (vote_mat * 1000 + score_mat).argmax(axis=1)
        best_cls  = [_class_ids[b] for b in best_pos]
        confs     = score_mat[_np.arange(N), best_pos] / _np.maximum(
                        vote_mat[_np.arange(N), best_pos], 1)

        return iter([(idxs[i], best_cls[i], float(confs[i])) for i in range(N)])

    t0 = time.time()
    raw         = sorted(test_rdd.mapPartitions(predict_partition).collect(),
                         key=lambda x: x[0])
    svm_pred_t  = time.time() - t0
    predictions = [r[1] for r in raw]
    pred_flops  = N_test * total_sv * (3*D + 2)

    # debug — แสดงการกระจาย prediction
    pred_arr = np.array(predictions)
    print(f"  Prediction done: {svm_pred_t*1000:.1f} ms")
    print(f"  Prediction distribution:")
    for cid in svm_class_ids:
        cnt = int(np.sum(pred_arr == cid))
        print(f"    class {cid} ({label_map.get(cid,'?')}): {cnt:,} samples ({100*cnt/N_test:.1f}%)")

    # ── Stage 5: DBSCAN ──────────────────────────────────────
    print("\n[Stage 4] DBSCAN on Benign samples...")

    benign_id = next((cid for cid, name in label_map.items()
                      if "normal" in name.lower()),
                     int(svm_class_ids[np.argmax(
                         [np.sum(train_labels == c) for c in svm_class_ids])]))
    print(f"  Benign class: {benign_id} ({label_map.get(benign_id,'?')})")

    final_preds  = list(predictions)
    dbscan_flops = 0
    dbscan_t     = 0.0

    benign_idx_arr = np.where(pred_arr == benign_id)[0].tolist()
    n_benign       = len(benign_idx_arr)
    print(f"  SVM→benign: {n_benign:,} samples → send to DBSCAN")

    if n_benign > 1:
        if n_benign > MAX_DBSCAN:
            rng_d    = np.random.default_rng(42)
            sel      = rng_d.choice(n_benign, MAX_DBSCAN, replace=False)
            cap_idx  = [benign_idx_arr[s] for s in sel]
            print(f"  Random-sampled {MAX_DBSCAN:,} from {n_benign:,}")
        else:
            cap_idx = benign_idx_arr

        cap_data = test_data[cap_idx]

        t0 = time.time()
        db_labels, n_clusters, n_noise, dbscan_flops = run_dbscan(
            cap_data, eps=DBSCAN_EPS, min_pts=DBSCAN_MIN_PTS)
        dbscan_t = time.time() - t0
        print(f"  DBSCAN done: {dbscan_t*1000:.1f} ms | "
              f"clusters={n_clusters} | noise={n_noise}")

        for i, orig in enumerate(cap_idx):
            final_preds[orig] = -1 if db_labels[i] >= 0 else -2

        save_dbscan_model(db_labels, cap_data, DBSCAN_EPS, DBSCAN_MIN_PTS,
                          "pyspark_dbscan_model.txt")

    spark.stop()

    # ── Stage 6: Metrics + Results ───────────────────────────
    total_t     = time.time() - t_total
    total_flops = train_flops + pred_flops + dbscan_flops
    gflops      = (total_flops / 1e9) / total_t if total_t > 0 else 0
    throughput  = N_test / total_t

    metrics = compute_metrics(test_labels, final_preds,
                              svm_class_ids, holdout_ids, label_map)

    print(f"\n{'='*60}")
    print(f"  RESULTS — PySpark Fast ({n_workers} workers)")
    print(f"{'='*60}")
    print(f"\n  [SVM — Known Attack Classification]")
    print(f"  {'ID':<5} {'Name':<30} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*65}")
    for cid in svm_class_ids:
        m = metrics["per_class"][cid]
        print(f"  {cid:<5} {label_map.get(cid,str(cid)):<30} "
              f"{m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
    print(f"\n  Accuracy (known classes) : {metrics['accuracy']:.2f}%")

    print(f"\n  [DBSCAN — Unknown Attack Detection]")
    print(f"  Holdout classes  : {holdout_names}")
    print(f"  Holdout in test  : {metrics['n_holdout']:,}")
    print(f"  Detected         : {metrics['detected']:,}")
    print(f"  Detection Rate   : {metrics['detection_rate']:.2f}%")
    print(f"  False Alarm Rate : {metrics['false_alarm_rate']:.2f}%"
          f" ({metrics['false_alarms']:,}/{metrics['n_normal']:,} normal)")

    print(f"\n  [Throughput]")
    print(f"  Train (parallel) : {svm_train_t*1000:>10.1f} ms")
    print(f"  Predict          : {svm_pred_t*1000:>10.1f} ms")
    print(f"  DBSCAN           : {dbscan_t*1000:>10.1f} ms")
    print(f"  Total            : {total_t*1000:>10.1f} ms")
    print(f"  FLOP count       : {total_flops:>20,}")
    print(f"  GFLOPS           : {gflops:>10.4f}")
    print(f"  Throughput       : {throughput:>10.1f} samples/sec")
    print(f"{'='*60}")

    # Save
    with open("pyspark_svm_model.pkl", "wb") as f:
        pickle.dump({"models": [(m[0].tolist(), m[1].tolist(),
                                 float(m[2]), int(m[3]), int(m[4]))
                                for m in models],
                     "gamma": GAMMA, "svm_class_ids": svm_class_ids}, f)
    print("  SVM model -> pyspark_svm_model.pkl")

    np.savetxt("pyspark_predictions.csv", np.array(final_preds), fmt="%d")
    print("  Predictions -> pyspark_predictions.csv")

    with open("pyspark_summary.csv", "w") as f:
        f.write("technique,n_train,n_test,features,svm_classes,"
                "accuracy,detection_rate,false_alarm_rate,"
                "train_ms,predict_ms,dbscan_ms,total_ms,flops,gflops,throughput\n")
        f.write(f"PySpark_{n_workers}w,"
                f"{N_train},{N_test},{D},{n_classes_svm},"
                f"{metrics['accuracy']:.4f},{metrics['detection_rate']:.4f},"
                f"{metrics['false_alarm_rate']:.4f},"
                f"{svm_train_t*1000:.1f},{svm_pred_t*1000:.1f},"
                f"{dbscan_t*1000:.1f},{total_t*1000:.1f},"
                f"{total_flops},{gflops:.4f},{throughput:.1f}\n")
    print("  Summary -> pyspark_summary.csv")

    return metrics


# ===================== Entry Point =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 500 samples/class train, 200/class test")
    args = parser.parse_args()

    if not HAS_SPARK:
        print("ERROR: pyspark not installed")
        sys.exit(1)

    run_pipeline(test_mode=args.test, n_workers=args.workers)
