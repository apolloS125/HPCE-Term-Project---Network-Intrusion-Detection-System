"""
DBSCAN Clustering
=================
DBSCAN-based anomaly detection for uncertain SVM predictions and pure clustering mode.
"""

import struct
from typing import Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

from config import PROJECT_ROOT, NORMAL_LABELS, MODEL_CONF_THRESHOLDS, HOLDOUT_CLASS_NAMES


def load_dbscan_model(model_id: str) -> dict:
    """Load pre-trained DBSCAN model from model folder.

    Format: n_clusters(int), eps(float), min_samples(int), threshold(float), n_dim(int),
            centroids(n_clusters x n_dim floats), radii(n_clusters floats)
    """
    dbscan_path = PROJECT_ROOT / model_id / "model" / "dbscan_model.bin"

    if not dbscan_path.exists():
        return None

    try:
        with open(dbscan_path, "rb") as f:
            # Read header
            n_clusters = struct.unpack('i', f.read(4))[0]
            eps = struct.unpack('f', f.read(4))[0]
            min_samples = struct.unpack('i', f.read(4))[0]
            threshold = struct.unpack('f', f.read(4))[0]
            n_dim = struct.unpack('i', f.read(4))[0]

            # Read centroids (in score/vote space, not feature space)
            centroid_size = n_clusters * n_dim
            centroids_flat = struct.unpack(f'{centroid_size}f', f.read(4 * centroid_size))
            centroids = np.array(centroids_flat, dtype=np.float32).reshape(n_clusters, n_dim)

            # Read cluster radii (optional - remaining bytes)
            remaining = f.read()
            radii = None
            if len(remaining) >= n_clusters * 4:
                radii = np.array(struct.unpack(f'{n_clusters}f', remaining[:n_clusters * 4]), dtype=np.float32)

            return {
                "n_clusters": n_clusters,
                "eps": eps,
                "min_samples": min_samples,
                "threshold": threshold,
                "n_dim": n_dim,
                "centroids": centroids,
                "radii": radii
            }
    except Exception as e:
        print(f"[WARN] Failed to load DBSCAN model for {model_id}: {e}")
        return None


def apply_dbscan_only(predictor, X: np.ndarray) -> Tuple[list, dict]:
    """
    Pure DBSCAN-only classification (NO SVM at all):

    1. Run DBSCAN directly on raw 52-D features
    2. Auto-tune eps from pairwise distances
    3. Assign generic cluster labels (Cluster 0, Cluster 1, etc.)

    This is pure unsupervised clustering without any SVM involvement.
    """
    N = X.shape[0]

    # Normalize features for better clustering
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Auto-tune eps from pairwise distances (30th percentile)
    if N > 1:
        pairwise_dists = pdist(X_normalized, metric='euclidean')
        eps = float(np.percentile(pairwise_dists, 30)) if pairwise_dists.size else 1.0
    else:
        eps = 1.0

    eps = max(eps, 1e-6)

    # Run DBSCAN on raw features
    min_samples = max(2, min(5, N // 20))  # Adaptive min_samples
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

    try:
        cluster_labels = dbscan.fit_predict(X_normalized)
    except Exception as e:
        return [], {
            "error": f"DBSCAN clustering failed: {str(e)}",
            "n_clusters": 0,
            "eps": eps,
        }

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outlier = int(np.sum(cluster_labels == -1))
    n_in_cluster = N - n_outlier

    # Build results
    results = []
    for i in range(N):
        cluster_id = int(cluster_labels[i])

        if cluster_id == -1:
            # Noise/outlier
            pred_label = "Unknown (Outlier)"
            is_outlier = True
            confidence = 0.0
        else:
            # In cluster
            pred_label = f"Unknown (Cluster)"
            is_outlier = False
            # Confidence based on distance to cluster centroid
            cluster_mask = cluster_labels == cluster_id
            cluster_points = X_normalized[cluster_mask]
            centroid = np.mean(cluster_points, axis=0)
            dist = float(np.linalg.norm(X_normalized[i] - centroid))
            max_dist = float(np.max([np.linalg.norm(p - centroid) for p in cluster_points]))
            confidence = 1.0 - min(dist / (max_dist + 1e-6), 1.0)

        results.append({
            "predicted_label": pred_label,
            "cluster_id": cluster_id,
            "is_outlier": is_outlier,
            "confidence": confidence,
        })

    stats = {
        "mode": "dbscan_only",
        "n_clusters": n_clusters,
        "eps": eps,
        "min_samples": min_samples,
        "n_in_cluster": n_in_cluster,
        "n_outlier": n_outlier,
        "using_pretrained": False,
        "note": "Pure DBSCAN on raw 52-D features (no SVM)"
    }

    return results, stats


def apply_dbscan_hybrid(results: list, predictor, X: np.ndarray, conf_threshold: float = None) -> Tuple[list, dict]:
    """
    Hybrid SVM+DBSCAN pipeline matching cuda_infer.cu / mpi_infer.cpp logic:

      conf > threshold                      → SVM prediction accepted (confident)
      conf <= threshold                     → Sent to DBSCAN uncertain pool

    DBSCAN clusters uncertain attack samples in raw OvO score space.
    Clustered samples are labelled by centroid OvO simulation.
    Noise (isolated) points stay with their SVM label marked Unverified.
    """
    model_id = predictor.model_id

    # Use model-specific threshold if not provided
    if conf_threshold is None:
        conf_threshold = MODEL_CONF_THRESHOLDS.get(model_id, 0.5)

    confident_idx         = []
    uncertain_normal_idx  = []   # low-conf Normal → keep as Normal, skip DBSCAN
    uncertain_attack_idx  = []   # low-conf attack  → DBSCAN

    for i, pred in enumerate(results):
        if pred["confidence"] > conf_threshold:
            confident_idx.append(i)
        else:
            # Ensure all uncertain samples, including Normal traffic, are processed by DBSCAN
            uncertain_attack_idx.append(i)

    n_confident        = len(confident_idx)
    n_uncertain_normal = len(uncertain_normal_idx)
    n_uncertain_attack = len(uncertain_attack_idx)

    stats = {
        "n_confident":        n_confident,
        "n_uncertain_normal": n_uncertain_normal,
        "n_uncertain":        n_uncertain_attack,
        "n_novel":            0,
        "n_unknown":          0,
        "n_clusters":         0,
        "eps":                0.0,
        "conf_threshold":     conf_threshold,
        "using_pretrained":   False,
    }

    # Mark confident predictions
    for i in confident_idx:
        results[i]["is_uncertain"] = False
        results[i]["is_novel"]     = False
        results[i]["cluster_id"]   = None

    # Uncertain Normal → stay Normal, just flag as low-confidence
    for i in uncertain_normal_idx:
        results[i]["is_uncertain"] = True
        results[i]["is_novel"]     = False
        results[i]["cluster_id"]   = None

    if n_uncertain_attack == 0:
        return results, stats

    X_uncertain = X[uncertain_attack_idx]

    # Build DBSCAN feature space — raw OvO scores (same space as C++ infer code)
    if hasattr(predictor, 'W') and predictor.W is not None:
        # Linear SVM → 6-D raw OvO decision values (mpi_infer.cpp / omp_infer.cpp)
        raw_scores = X_uncertain @ predictor.W.T + predictor.b
    elif hasattr(predictor, 'rff_W') and predictor.rff_W is not None:
        # RFF-SVM → RFF transform then 6-D OvO scores (cuda_infer.cu)
        D = predictor.Omega.shape[0]
        Z = np.sqrt(2.0 / D) * np.cos(X_uncertain @ predictor.Omega.T + predictor.phi)
        raw_scores = Z @ predictor.rff_W.T + predictor.rff_b
    else:
        # Kernel SVM fallback → 4-D vote-count space
        raw_scores = []
        for i in uncertain_attack_idx:
            v = results[i].get("votes", {})
            raw_scores.append([v.get(lbl, 0.0) for lbl in sorted(v.keys())])
        raw_scores = np.array(raw_scores, dtype=np.float32)

    # Try pre-trained DBSCAN model
    dbscan_model = load_dbscan_model(model_id)

    if dbscan_model is not None and dbscan_model["n_dim"] == raw_scores.shape[1]:
        stats["using_pretrained"] = True
        stats["n_clusters"]       = dbscan_model["n_clusters"]
        stats["eps"]              = dbscan_model["eps"]
        centroids = dbscan_model["centroids"]

        all_dists = [float(np.min(np.linalg.norm(centroids - sv, axis=1))) for sv in raw_scores]
        adaptive_threshold = float(np.median(all_dists)) if all_dists else dbscan_model["eps"] * 10

        n_novel = 0
        n_unknown = 0
        for local_idx, global_idx in enumerate(uncertain_attack_idx):
            dists = np.linalg.norm(centroids - raw_scores[local_idx], axis=1)
            min_dist = float(np.min(dists))
            nearest  = int(np.argmin(dists))
            if min_dist <= adaptive_threshold:
                orig = results[global_idx]["predicted_label"].replace(" (Unverified)", "").replace(" (DBSCAN verified)", "")
                results[global_idx]["predicted_label"] = f"{orig} (Cluster)"
                results[global_idx]["cluster_id"]      = nearest
                results[global_idx]["is_uncertain"]    = False
                results[global_idx]["is_novel"]        = False
                results[global_idx]["dbscan_distance"] = min_dist
                n_novel += 1
            else:
                orig = results[global_idx]["predicted_label"].replace(" (Unverified)", "").replace(" (DBSCAN verified)", "")
                results[global_idx]["predicted_label"] = f"{orig} (Outlier)"
                results[global_idx]["cluster_id"]      = -1
                results[global_idx]["is_uncertain"]    = True
                results[global_idx]["is_novel"]        = True
                results[global_idx]["dbscan_distance"] = min_dist
                n_unknown += 1

        stats["n_novel"]   = n_novel
        stats["n_unknown"] = n_unknown

    else:
        # Online DBSCAN — auto-tune eps from pairwise distances (30th percentile)
        n_db = len(uncertain_attack_idx)
        if n_db > 1:
            pairwise_dists = pdist(raw_scores, metric='euclidean')
            eps = float(np.percentile(pairwise_dists, 30)) if pairwise_dists.size else 0.5
        else:
            eps = 0.5
        eps = max(eps, 1e-6)
        stats["eps"] = eps

        min_samples = max(2, min(3, n_db))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        try:
            cluster_labels = dbscan.fit_predict(raw_scores)
        except Exception:
            return results, stats

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        stats["n_clusters"] = n_clusters
        stats["n_unknown"]  = int(np.sum(cluster_labels == -1))
        stats["n_novel"]    = n_db - stats["n_unknown"]

        # Simulate per-cluster attack label from centroid OvO scores
        cluster_vecs: dict = {}
        for local_idx, cid in enumerate(cluster_labels):
            cid = int(cid)
            if cid >= 0:
                cluster_vecs.setdefault(cid, []).append(raw_scores[local_idx])

        svm_names = ["DDoS", "DoS", "NormalTraffic", "PortScan"]
        ovo_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        cluster_label_map: dict = {}
        for cid, vecs in cluster_vecs.items():
            centroid = np.mean(vecs, axis=0)
            if centroid.shape[0] == 6:   # 6-D OvO score space
                vote_sim = [0] * 4
                for pi, (pa, pb) in enumerate(ovo_pairs):
                    if centroid[pi] >= 0:
                        vote_sim[pa] += 1
                    else:
                        vote_sim[pb] += 1
                cluster_label_map[cid] = svm_names[int(np.argmax(vote_sim))]
            else:                        # 4-D vote-count space
                cluster_label_map[cid] = svm_names[int(np.argmax(centroid))]

        for local_idx, cid in enumerate(cluster_labels):
            cid = int(cid)
            global_idx = uncertain_attack_idx[local_idx]
            if cid == -1:
                # Noise: isolated uncertain attack — flag as Outlier
                orig = results[global_idx]["predicted_label"].replace(" (Unverified)", "").replace(" (DBSCAN verified)", "")
                results[global_idx]["predicted_label"] = f"{orig} (Outlier)"
                results[global_idx]["cluster_id"]      = -1
                results[global_idx]["is_uncertain"]    = True
                results[global_idx]["is_novel"]        = False
            else:
                # Clustered: attack pattern identified by centroid simulation
                attack_lbl = cluster_label_map.get(cid, "Unknown")
                results[global_idx]["predicted_label"] = f"{attack_lbl} (Cluster)"
                results[global_idx]["cluster_id"]      = cid
                results[global_idx]["is_uncertain"]    = True
                results[global_idx]["is_novel"]        = True

    return results, stats
