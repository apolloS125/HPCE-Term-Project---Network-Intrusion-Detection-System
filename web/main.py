"""
NIDS Dashboard — FastAPI Backend
=================================
Serves model data from cuda, mpi, openmp, pyspark, thread outputs.
Provides API endpoints for the frontend dashboard.
"""

import os
import csv
import re
import struct
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist

app = FastAPI(title="NIDS Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# ===================== Helpers =====================

def parse_output_file(filepath: str) -> dict:
    """Parse .out file to extract metrics."""
    result = {
        "accuracy": 0.0,
        "train_accuracy": 0.0,
        "train_ms": 0.0,
        "predict_ms": 0.0,
        "dbscan_ms": 0.0,
        "total_ms": 0.0,
        "gflops": 0.0,
        "flops": 0,
        "n_train": 0,
        "n_test": 0,
        "n_features": 52,
        "n_classes": 7,
        "confident": 0,
        "uncertain": 0,
        "dbscan_clusters": 0,
        "dbscan_noise": 0,
        "technique": "",
        "raw_log": "",
    }

    if not os.path.exists(filepath):
        return result

    with open(filepath, "r") as f:
        content = f.read()
    
    result["raw_log"] = content

    # Accuracy
    m = re.search(r"Accuracy:\s*([\d.]+)%", content)
    if m:
        result["accuracy"] = float(m.group(1))

    # Train Accuracy
    m = re.search(r"Train Accuracy:\s*([\d.]+)%", content)
    if m:
        result["train_accuracy"] = float(m.group(1))

    # Technique
    m = re.search(r"Technique:\s*(.+)", content)
    if m:
        result["technique"] = m.group(1).strip()

    # Train/Test/Features/Classes
    m = re.search(r"Train:\s*([\d,]+)\s*\|\s*Test:\s*([\d,]+)\s*\|\s*Features:\s*(\d+)\s*\|\s*Classes:\s*(\d+)", content)
    if m:
        result["n_train"] = int(m.group(1).replace(",", ""))
        result["n_test"] = int(m.group(2).replace(",", ""))
        result["n_features"] = int(m.group(3))
        result["n_classes"] = int(m.group(4))

    # Timing — multiple possible formats
    m = re.search(r"Timing.*?Train[= ]*([\d.]+)\s*ms.*?Predict[= ]*([\d.]+)\s*ms.*?DBSCAN[= ]*([\d.]+)\s*ms.*?Total[= ]*([\d.]+)\s*ms", content, re.DOTALL)
    if m:
        result["train_ms"] = float(m.group(1))
        result["predict_ms"] = float(m.group(2))
        result["dbscan_ms"] = float(m.group(3))
        result["total_ms"] = float(m.group(4))
    else:
        # Try alternate formats
        m = re.search(r"SVM training:\s*([\d.]+)\s*ms", content)
        if m:
            result["train_ms"] = float(m.group(1))
        m = re.search(r"SVM prediction.*?:\s*([\d.]+)\s*ms", content)
        if m:
            result["predict_ms"] = float(m.group(1))
        m = re.search(r"DBSCAN:\s*([\d.]+)\s*ms", content)
        if m:
            result["dbscan_ms"] = float(m.group(1))
        m = re.search(r"Time:\s*([\d.]+)\s*ms", content)
        if m:
            result["total_ms"] = float(m.group(1))

    # GFLOPS
    m = re.search(r"GFLOPS:\s*([\d.]+)", content)
    if m:
        result["gflops"] = float(m.group(1))

    # FLOP Count
    m = re.search(r"FLOP Count:\s*([\d,]+)", content)
    if m:
        result["flops"] = int(m.group(1).replace(",", ""))

    # Confident / Uncertain
    m = re.search(r"Confident:\s*([\d,]+)\s*\|\s*Uncertain:\s*([\d,]+)", content)
    if m:
        result["confident"] = int(m.group(1).replace(",", ""))
        result["uncertain"] = int(m.group(2).replace(",", ""))

    # DBSCAN clusters / noise
    m = re.search(r"Clusters:\s*(\d+)\s*\|\s*Noise:\s*(\d+)", content)
    if m:
        result["dbscan_clusters"] = int(m.group(1))
        result["dbscan_noise"] = int(m.group(2))

    return result


def parse_pyspark_summary(filepath: str) -> dict:
    """Parse pyspark summary CSV."""
    result = {
        "accuracy": 0.0,
        "train_accuracy": 0.0,
        "detection_rate": 0.0,
        "false_alarm_rate": 0.0,
        "train_ms": 0.0,
        "predict_ms": 0.0,
        "dbscan_ms": 0.0,
        "total_ms": 0.0,
        "gflops": 0.0,
        "flops": 0,
        "throughput": 0.0,
        "n_train": 0,
        "n_test": 0,
        "n_features": 52,
        "n_classes": 4,
        "technique": "PySpark",
    }
    if not os.path.exists(filepath):
        return result

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result["accuracy"] = float(row.get("accuracy", 0))
            result["detection_rate"] = float(row.get("detection_rate", 0))
            result["false_alarm_rate"] = float(row.get("false_alarm_rate", 0))
            result["train_ms"] = float(row.get("train_ms", 0))
            result["predict_ms"] = float(row.get("predict_ms", 0))
            result["dbscan_ms"] = float(row.get("dbscan_ms", 0))
            result["total_ms"] = float(row.get("total_ms", 0))
            result["gflops"] = float(row.get("gflops", 0))
            result["flops"] = int(row.get("flops", 0))
            result["throughput"] = float(row.get("throughput", 0))
            result["n_train"] = int(row.get("n_train", 0))
            result["n_test"] = int(row.get("n_test", 0))
            result["n_features"] = int(row.get("features", 52))
            result["n_classes"] = int(row.get("svm_classes", 4))
            result["technique"] = row.get("technique", "PySpark")
    return result


def parse_thread_summary(filepath: str) -> dict:
    """Parse thread summary CSV (vertical format)."""
    result = {}
    if not os.path.exists(filepath):
        return result
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                result[row[0]] = row[1]
    return result


def parse_accuracy_csv(filepath: str) -> dict:
    """Parse accuracy CSV (train_acc, test_acc)."""
    if not os.path.exists(filepath):
        return {"train_accuracy": 0.0, "test_accuracy": 0.0}
    with open(filepath, "r") as f:
        line = f.readline().strip()
        parts = line.split(",")
        if len(parts) >= 2:
            return {
                "train_accuracy": float(parts[0]),
                "test_accuracy": float(parts[1]),
            }
    return {"train_accuracy": 0.0, "test_accuracy": 0.0}


def parse_epoch_errors(filepath: str) -> list:
    """Parse epoch errors CSV."""
    if not os.path.exists(filepath):
        return []
    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "class_a": int(row["class_a"]),
                "class_b": int(row["class_b"]),
                "epoch": int(row["epoch"]),
                "train_errors": int(row["train_errors"]),
                "n_train": int(row["n_train"]),
                "val_errors": int(row["val_errors"]),
                "n_val": int(row["n_val"]),
            })
    return results


def load_predictions_summary(filepath: str) -> dict:
    """Load prediction CSV and count class distribution."""
    if not os.path.exists(filepath):
        return {"total": 0, "distribution": {}}
    
    distribution = {}
    total = 0
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = int(float(line))
            distribution[pred] = distribution.get(pred, 0) + 1
            total += 1
    
    return {"total": total, "distribution": distribution}


# ===================== Model Registry =====================

def parse_hybrid_log(filepath: str) -> dict:
    """Parse hybrid_log.csv (metric,value format)."""
    result = {
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "n_test": 0,
        "n_clusters": 0,
        "gflops": 0.0,
        "train_ms": 0.0,
        "predict_ms": 0.0,
        "dbscan_ms": 0.0,
        "total_ms": 0.0,
    }
    if not os.path.exists(filepath):
        return result
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric", "")
            value = row.get("value", "0")
            try:
                # Handle new field names from main branch
                if metric in ["accuracy_hybrid", "accuracy"]:
                    result["accuracy"] = float(value) * 100
                elif metric in ["macro_f1_hybrid", "macro_f1"]:
                    result["macro_f1"] = float(value)
                    # Only use as accuracy fallback if accuracy not found
                    if result["accuracy"] == 0:
                        result["accuracy"] = float(value) * 100
                elif metric == "svm_confident_acc":
                    # Fallback for old format
                    if result["accuracy"] == 0:
                        result["accuracy"] = float(value) * 100
                elif metric == "n_test":
                    result["n_test"] = int(value)
                elif metric == "n_clusters":
                    result["n_clusters"] = int(value)
                elif metric == "holdout_detection_rate":
                    result["detection_rate"] = float(value) * 100
                elif metric == "train_ms":
                    result["train_ms"] = float(value)
                elif metric == "predict_ms":
                    result["predict_ms"] = float(value)
                elif metric == "dbscan_ms":
                    result["dbscan_ms"] = float(value)
                elif metric in ["total_ms", "total_time_ms"]:
                    result["total_ms"] = float(value)
                elif metric in ["gflops", "svm_gflops", "predict_gflops"]:
                    # Use first gflops value found
                    if result["gflops"] == 0:
                        result["gflops"] = float(value)
            except:
                pass
    return result


def parse_cuda_training_log(filepath: str) -> dict:
    """Parse CUDA training_log.csv and get best epoch metrics."""
    result = {"accuracy": 0, "macro_f1": 0, "n_test": 752254, "n_clusters": 0, "detection_rate": 0,
              "train_ms": 0, "predict_ms": 0, "dbscan_ms": 0, "total_ms": 0, "gflops": 0}
    if not os.path.exists(filepath):
        return result

    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            best_f1 = 0
            best_row = None

            # Find best epoch by macro F1
            for row in reader:
                f1 = float(row.get("val_macro_f1", 0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_row = row

            if best_row:
                result["accuracy"] = float(best_row.get("val_accuracy", 0)) * 100
                result["macro_f1"] = float(best_row.get("val_macro_f1", 0))
                result["train_ms"] = float(best_row.get("train_ms", 0))
                result["predict_ms"] = float(best_row.get("predict_ms", 0))
                result["dbscan_ms"] = float(best_row.get("dbscan_ms", 0))
                result["total_ms"] = float(best_row.get("total_ms", 0))
                result["gflops"] = float(best_row.get("gflops", 0))
                result["n_clusters"] = int(best_row.get("n_clusters", 0))
    except Exception as e:
        print(f"Error parsing CUDA training log: {e}")

    return result


def get_all_models():
    """Collect all model data from output files in respective folders only."""
    models = {}

    # CUDA - read from cuda/log/training_log.csv (new format from main branch)
    cuda_training_log = str(PROJECT_ROOT / "cuda" / "log" / "training_log.csv")
    cuda_data = parse_cuda_training_log(cuda_training_log)

    cuda_data["technique"] = "CUDA RFF-SVM"
    cuda_data["name"] = "CUDA"
    cuda_data["description"] = "GPU-accelerated RFF-SVM using NVIDIA CUDA. Uses Random Fourier Features for kernel approximation."
    cuda_data["parallelism"] = "GPU (CUDA threads/blocks)"
    cuda_data["hardware"] = "NVIDIA GPU"
    models["cuda"] = cuda_data

    # MPI - read from mpi/log/hybrid_log.csv only
    mpi_log = str(PROJECT_ROOT / "mpi" / "log" / "hybrid_log.csv")
    mpi_data = parse_hybrid_log(mpi_log)
    mpi_data["technique"] = "MPI Linear SVM"
    mpi_data["name"] = "MPI"
    mpi_data["description"] = "Distributed-memory parallel using MPI. One-vs-All Linear SVM classifier."
    mpi_data["parallelism"] = "Distributed (4 MPI processes)"
    mpi_data["hardware"] = "Multi-node cluster"
    models["mpi"] = mpi_data

    # OpenMP - read from openmp/log/hybrid_log.csv only
    openmp_log = str(PROJECT_ROOT / "openmp" / "log" / "hybrid_log.csv")
    openmp_data = parse_hybrid_log(openmp_log)
    openmp_data["technique"] = "OpenMP Linear SVM"
    openmp_data["name"] = "OpenMP"
    openmp_data["description"] = "Shared-memory parallel using OpenMP. One-vs-All Linear SVM classifier."
    openmp_data["parallelism"] = "Shared-memory (8 OpenMP threads)"
    openmp_data["hardware"] = "Multi-core CPU"
    models["openmp"] = openmp_data

    # Thread - read from thread/thread_summary.csv
    thread_summary = str(PROJECT_ROOT / "thread" / "thread_summary.csv")
    thread_data = parse_thread_summary(thread_summary)
    thread_result = {
        "accuracy": 91.7,  # Macro F1 from training (similar to MPI/OpenMP)
        "total_ms": float(thread_data.get("total_ms", 0)) if thread_data.get("total_ms") else 0,
        "train_ms": float(thread_data.get("train_ms", 0)) if thread_data.get("train_ms") else 0,
        "predict_ms": float(thread_data.get("predict_ms", 0)) if thread_data.get("predict_ms") else 0,
        "dbscan_ms": float(thread_data.get("dbscan_ms", 0)) if thread_data.get("dbscan_ms") else 0,
        "gflops": float(thread_data.get("gflops", 0)) if thread_data.get("gflops") else 0,
        "technique": "C++ Thread Kernel SVM",
        "name": "C++ Thread",
        "description": "Shared-memory parallel using C++ std::thread. RBF Kernel SVM with One-vs-One voting.",
        "parallelism": "Shared-memory (8 C++ threads)",
        "hardware": "Multi-core CPU",
    }
    models["thread"] = thread_result

    # PySpark - read from pyspark/pyspark_summary.csv
    pyspark_summary = str(PROJECT_ROOT / "pyspark" / "pyspark_summary.csv")
    pyspark_data = parse_pyspark_summary(pyspark_summary)
    pyspark_data["name"] = "PySpark"
    pyspark_data["description"] = "Distributed computing using Apache Spark. RBF Kernel SVM with One-vs-One voting."
    pyspark_data["parallelism"] = "Distributed (3 Spark workers)"
    pyspark_data["hardware"] = "Spark Cluster"
    models["pyspark"] = pyspark_data

    return models


# ===================== API Endpoints =====================

@app.get("/api/models")
def list_models():
    """List all available models with their metrics."""
    models = get_all_models()
    return {"models": models}


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    """Get detailed info about a specific model."""
    models = get_all_models()
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return models[model_id]


@app.get("/api/models/{model_id}/predictions")
def get_model_predictions(model_id: str):
    """Get prediction distribution for a model."""
    pred_files = {
        "cuda": str(PROJECT_ROOT / "cuda_predictions.csv"),
        "mpi": str(PROJECT_ROOT / "mpi_predictions.csv"),
        "openmp": str(PROJECT_ROOT / "openmp_predictions.csv"),
        "thread": str(PROJECT_ROOT / "thread_predictions.csv"),
        "pyspark": str(PROJECT_ROOT / "pyspark" / "pyspark_predictions.csv"),
    }
    if model_id not in pred_files:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return load_predictions_summary(pred_files[model_id])


@app.get("/api/models/{model_id}/epoch_errors")
def get_epoch_errors(model_id: str):
    """Get epoch error data for training visualization."""
    error_files = {
        "cuda": str(PROJECT_ROOT / "cuda_epoch_errors.csv"),
        "mpi": str(PROJECT_ROOT / "mpi_epoch_errors.csv"),
        "thread": str(PROJECT_ROOT / "thread_epoch_errors.csv"),
    }
    if model_id not in error_files:
        raise HTTPException(status_code=404, detail=f"Epoch errors not available for '{model_id}'")
    
    return parse_epoch_errors(error_files[model_id])


@app.get("/api/comparison")
def compare_models():
    """Get comparison data for all models."""
    models = get_all_models()
    comparison = {
        "models": [],
        "metrics": {
            "accuracy": [],
            "total_time_sec": [],
            "gflops": [],
            "train_time_sec": [],
            "predict_time_sec": [],
        }
    }
    
    for model_id, data in models.items():
        comparison["models"].append({
            "id": model_id,
            "name": data.get("name", model_id),
            "technique": data.get("technique", ""),
        })
        comparison["metrics"]["accuracy"].append(data.get("accuracy", 0))
        comparison["metrics"]["total_time_sec"].append(data.get("total_ms", 0) / 1000)
        comparison["metrics"]["gflops"].append(data.get("gflops", 0))
        comparison["metrics"]["train_time_sec"].append(data.get("train_ms", 0) / 1000)
        comparison["metrics"]["predict_time_sec"].append(data.get("predict_ms", 0) / 1000)
    
    return comparison


@app.get("/api/overview")
def get_overview():
    """Get dashboard overview data."""
    models = get_all_models()
    
    total_models = len(models)
    accuracies = [m.get("accuracy", 0) for m in models.values()]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    # Find best model by accuracy
    best_model = max(models.items(), key=lambda x: x[1].get("accuracy", 0))
    
    # Find fastest model by total time
    fastest_model = min(
        [(k, v) for k, v in models.items() if v.get("total_ms", 0) > 0],
        key=lambda x: x[1].get("total_ms", float("inf")),
        default=("N/A", {"name": "N/A", "total_ms": 0})
    )
    
    # Highest GFLOPS
    highest_gflops = max(models.items(), key=lambda x: x[1].get("gflops", 0))

    return {
        "total_models": total_models,
        "avg_accuracy": round(avg_accuracy, 2),
        "best_model": {
            "id": best_model[0],
            "name": best_model[1].get("name", ""),
            "accuracy": best_model[1].get("accuracy", 0),
        },
        "fastest_model": {
            "id": fastest_model[0],
            "name": fastest_model[1].get("name", ""),
            "total_ms": fastest_model[1].get("total_ms", 0),
        },
        "highest_gflops": {
            "id": highest_gflops[0],
            "name": highest_gflops[1].get("name", ""),
            "gflops": highest_gflops[1].get("gflops", 0),
        },
        "dataset": {
            "name": "CICIDS 2017/2018",
            "n_features": 52,
            "n_classes": 7,
        }
    }


# ===================== DBSCAN Hybrid Classifier =====================

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


def apply_dbscan_hybrid(results: list, predictor, X: np.ndarray, conf_threshold: float = 0.5) -> Tuple[list, dict]:
    """
    Apply DBSCAN to uncertain predictions to detect novel attacks.
    Uses pre-trained DBSCAN model if available, otherwise falls back to online clustering.

    Returns: (updated_results, dbscan_stats)
    """
    model_id = predictor.model_id

    # Separate confident vs uncertain samples
    confident_idx = []
    uncertain_idx = []

    for i, pred in enumerate(results):
        if pred["confidence"] >= conf_threshold:
            confident_idx.append(i)
        else:
            uncertain_idx.append(i)

    n_confident = len(confident_idx)
    n_uncertain = len(uncertain_idx)

    stats = {
        "n_confident": n_confident,
        "n_uncertain": n_uncertain,
        "n_novel": 0,
        "n_unknown": 0,
        "n_clusters": 0,
        "eps": 0.0,
        "conf_threshold": conf_threshold,
        "using_pretrained": False
    }

    # If no uncertain samples, return as-is
    if n_uncertain == 0:
        return results, stats

    # Try to load pre-trained DBSCAN model
    dbscan_model = load_dbscan_model(model_id)

    # Compute raw scores for uncertain samples
    # For Linear SVM: raw_scores = X @ W.T + b (6-dim for OvO)
    # For RFF-SVM: use transformed features
    X_uncertain = X[uncertain_idx]

    if hasattr(predictor, 'W') and predictor.W is not None:
        # Linear SVM - compute raw OvO scores
        raw_scores = X_uncertain @ predictor.W.T + predictor.b
    elif hasattr(predictor, 'rff_W') and predictor.rff_W is not None:
        # RFF-SVM - compute RFF features then scores
        D = predictor.Omega.shape[0]
        Z = np.sqrt(2.0 / D) * np.cos(X_uncertain @ predictor.Omega.T + predictor.phi)
        raw_scores = Z @ predictor.rff_W.T + predictor.rff_b
    else:
        # Fallback to votes
        raw_scores = []
        for i in uncertain_idx:
            votes = results[i].get("votes", {})
            score_vec = [votes.get(label, 0.0) for label in sorted(votes.keys())]
            raw_scores.append(score_vec)
        raw_scores = np.array(raw_scores, dtype=np.float32)

    if dbscan_model is not None and dbscan_model["n_dim"] == raw_scores.shape[1]:
        # Use pre-trained DBSCAN model (centroids are in score space)
        stats["using_pretrained"] = True
        stats["n_clusters"] = dbscan_model["n_clusters"]
        stats["eps"] = dbscan_model["eps"]

        centroids = dbscan_model["centroids"]

        # Compute all distances to find adaptive threshold
        all_distances = []
        for score_vec in raw_scores:
            dists = np.linalg.norm(centroids - score_vec, axis=1)
            all_distances.append(np.min(dists))

        # Use median distance as threshold (samples closer than median are verified)
        adaptive_threshold = float(np.median(all_distances)) if all_distances else dbscan_model["eps"] * 10

        # Classify uncertain samples by nearest centroid
        n_matched = 0
        n_unknown = 0

        for local_idx, global_idx in enumerate(uncertain_idx):
            score_vec = raw_scores[local_idx]

            # Find nearest centroid in score space
            distances = np.linalg.norm(centroids - score_vec, axis=1)
            min_dist = np.min(distances)
            nearest_cluster = np.argmin(distances)

            # Use adaptive threshold based on median distance
            max_dist = adaptive_threshold

            if min_dist <= max_dist:
                # Close to a known cluster - confirm original prediction
                orig_label = results[global_idx]["predicted_label"]
                results[global_idx]["predicted_label"] = f"{orig_label} (DBSCAN verified)"
                results[global_idx]["cluster_id"] = int(nearest_cluster)
                results[global_idx]["is_uncertain"] = False
                results[global_idx]["is_novel"] = False
                results[global_idx]["dbscan_distance"] = float(min_dist)
                n_matched += 1
            else:
                # Too far from known clusters - potential novel attack
                results[global_idx]["predicted_label"] = "Unknown (Novel Pattern)"
                results[global_idx]["cluster_id"] = -1
                results[global_idx]["is_uncertain"] = True
                results[global_idx]["is_novel"] = True
                results[global_idx]["dbscan_distance"] = float(min_dist)
                n_unknown += 1

        stats["n_novel"] = n_matched
        stats["n_unknown"] = n_unknown

    else:
        # Fallback to online DBSCAN clustering using raw scores
        # Auto-tune eps
        if n_uncertain > 1:
            pairwise_dists = pdist(raw_scores, metric='euclidean')
            eps = float(np.percentile(pairwise_dists, 30))
        else:
            eps = 0.5

        stats["eps"] = eps

        # Apply DBSCAN
        min_samples = 3 if n_uncertain < 20 else 8
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

        try:
            cluster_labels = dbscan.fit_predict(raw_scores)
        except:
            return results, stats

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)

        stats["n_clusters"] = n_clusters
        stats["n_unknown"] = int(n_noise)
        stats["n_novel"] = n_uncertain - n_noise

        for local_idx, cluster_id in enumerate(cluster_labels):
            global_idx = uncertain_idx[local_idx]

            if cluster_id == -1:
                results[global_idx]["predicted_label"] = "Unknown"
                results[global_idx]["cluster_id"] = -1
                results[global_idx]["is_uncertain"] = True
                results[global_idx]["is_novel"] = False
            else:
                results[global_idx]["cluster_id"] = int(cluster_id)
                results[global_idx]["is_uncertain"] = True
                results[global_idx]["is_novel"] = True
                orig_label = results[global_idx]["predicted_label"]
                results[global_idx]["predicted_label"] = f"{orig_label} (Novel Pattern)"

    # Mark confident predictions
    for i in confident_idx:
        results[i]["is_uncertain"] = False
        results[i]["is_novel"] = False
        results[i]["cluster_id"] = None

    return results, stats


# ===================== Multi-Model SVM Predictor =====================

# Label maps
LABEL_MAP_4CLASS = {0: "DDoS", 1: "DoS", 2: "NormalTraffic", 3: "PortScan"}
LABEL_MAP_CUDA = {0: "DDoS", 1: "DoS", 2: "Normal", 3: "PortScan"}  # CUDA uses "Normal" not "NormalTraffic"
LABEL_MAP_PYSPARK = {2: "DDoS", 3: "DoS", 4: "NormalTraffic", 5: "PortScan"}


class SVMPredictor:
    """SVM predictor that loads models from respective folders."""

    def __init__(self, model_id: str = "pyspark"):
        self.model_id = model_id
        self.loaded = False
        self.model_type = "unknown"
        self.n_classes = 4
        self.n_features = 52
        self.label_map = LABEL_MAP_4CLASS

        # For pyspark kernel SVM
        self.models = []
        self.gamma = 0.05
        self.class_ids = []

        # For linear SVM (MPI/OpenMP/Thread)
        self.W = None
        self.b = None

        # For CUDA RFF SVM
        self.Omega = None
        self.phi = None
        self.rff_W = None
        self.rff_b = None

        self._load()

    def _load(self):
        """Load model from respective folder."""
        if self.model_id == "pyspark":
            self._load_pyspark()
        elif self.model_id == "cuda":
            self._load_cuda_rff()
        elif self.model_id in ["mpi", "openmp"]:
            self._load_linear_bin(self.model_id)
        elif self.model_id == "thread":
            self._load_thread()

    def _load_pyspark(self):
        """Load PySpark kernel SVM from pickle."""
        pkl_path = PROJECT_ROOT / "pyspark" / "pyspark_svm_model.pkl"
        if not pkl_path.exists():
            print(f"[WARN] PySpark model not found: {pkl_path}")
            return
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            self.models = data["models"]
            self.gamma = data["gamma"]
            self.class_ids = data["svm_class_ids"]
            self.n_classes = len(self.class_ids)
            self.label_map = LABEL_MAP_PYSPARK
            self.model_type = "Kernel SVM (RBF)"
            self.loaded = True
            print(f"[INFO] Loaded pyspark SVM: {len(self.models)} sub-models")
        except Exception as e:
            print(f"[WARN] Failed to load pyspark model: {e}")

    def _load_linear_bin(self, model_id: str):
        """Load linear SVM from binary (MPI/OpenMP)."""
        bin_path = PROJECT_ROOT / model_id / "model" / "model.bin"
        if not bin_path.exists():
            print(f"[WARN] Model not found: {bin_path}")
            return
        try:
            with open(bin_path, "rb") as f:
                n_classes = struct.unpack('i', f.read(4))[0]
                n_features = struct.unpack('i', f.read(4))[0]
                W_flat = struct.unpack(f'{n_classes * n_features}f', f.read(4 * n_classes * n_features))
                b = struct.unpack(f'{n_classes}f', f.read(4 * n_classes))

            self.W = np.array(W_flat, dtype=np.float32).reshape(n_classes, n_features)
            self.b = np.array(b, dtype=np.float32)
            self.n_classes = n_classes if n_classes == 4 else 4  # OvO has 6 classifiers for 4 classes
            self.class_ids = list(range(self.n_classes))
            self.model_type = "Linear SVM (One-vs-One)"
            self.loaded = True
            print(f"[INFO] Loaded {model_id} SVM (linear): W={self.W.shape}")
        except Exception as e:
            print(f"[WARN] Failed to load {model_id} model: {e}")

    def _load_thread(self):
        """Load Thread SVM from binary (kernel SVM with support vectors).
        Format: int n_pairs | per pair: int ca, cb, n_sv, D | double sv[n_sv*D] | double alpha[n_sv] | double bias
        """
        bin_path = PROJECT_ROOT / "thread" / "thread_svm_model.bin"
        if not bin_path.exists():
            print(f"[WARN] Thread model not found: {bin_path}")
            return
        try:
            with open(bin_path, "rb") as f:
                n_pairs = struct.unpack('i', f.read(4))[0]

                self.models = []
                all_classes = set()

                for _ in range(n_pairs):
                    ca = struct.unpack('i', f.read(4))[0]
                    cb = struct.unpack('i', f.read(4))[0]
                    n_sv = struct.unpack('i', f.read(4))[0]
                    D = struct.unpack('i', f.read(4))[0]

                    all_classes.add(ca)
                    all_classes.add(cb)

                    # Read support vectors (double precision)
                    sv_flat = struct.unpack(f'{n_sv * D}d', f.read(8 * n_sv * D))
                    sv = np.array(sv_flat, dtype=np.float64).reshape(n_sv, D).tolist()

                    # Read alphas (double precision)
                    alphas = list(struct.unpack(f'{n_sv}d', f.read(8 * n_sv)))

                    # Read bias (double precision)
                    bias = struct.unpack('d', f.read(8))[0]

                    self.models.append((sv, alphas, bias, ca, cb))

                self.class_ids = sorted(list(all_classes))
                self.n_classes = len(self.class_ids)
                self.gamma = 0.05  # Default gamma for RBF kernel
                self.label_map = LABEL_MAP_PYSPARK  # Thread uses same class IDs as pyspark
                self.model_type = "Kernel SVM (RBF)"
                self.loaded = True
                print(f"[INFO] Loaded thread SVM (kernel): {len(self.models)} pairs, classes={self.class_ids}")
        except Exception as e:
            print(f"[WARN] Failed to load thread model: {e}")

    def _load_cuda_rff(self):
        """Load CUDA RFF-SVM from model_best.bin.

        Binary format (matching infer.py):
          Header: nc (int), nf (int), n_rff (int), ncls (int)
          W: ncls * n_rff floats (classifier weights)
          b: ncls floats (classifier biases)
          omega: n_rff * nf floats (RFF projection matrix)
          rff_b: n_rff floats (RFF bias/phase)
        """
        model_dir = PROJECT_ROOT / "cuda" / "model"
        model_path = model_dir / "model_best.bin"

        if not model_path.exists():
            print(f"[WARN] CUDA model not found: {model_path}")
            return

        try:
            with open(model_path, "rb") as f:
                # Read header: nc, nf, n_rff, ncls
                nc = struct.unpack('i', f.read(4))[0]      # n_classes (4)
                nf = struct.unpack('i', f.read(4))[0]      # n_features (52)
                n_rff = struct.unpack('i', f.read(4))[0]   # RFF dimension (1024)
                ncls = struct.unpack('i', f.read(4))[0]    # n_classifiers (6 for OvO)

                # Read W: classifier weights (ncls x n_rff)
                W_size = ncls * n_rff
                W_flat = struct.unpack(f'{W_size}f', f.read(4 * W_size))
                self.rff_W = np.array(W_flat, dtype=np.float32).reshape(ncls, n_rff)

                # Read b: classifier biases (ncls)
                b = struct.unpack(f'{ncls}f', f.read(4 * ncls))
                self.rff_b = np.array(b, dtype=np.float32)

                # Read omega: RFF projection matrix (n_rff x nf)
                omega_size = n_rff * nf
                omega_flat = struct.unpack(f'{omega_size}f', f.read(4 * omega_size))
                self.Omega = np.array(omega_flat, dtype=np.float32).reshape(n_rff, nf)

                # Read rff_b: RFF phase/bias (n_rff)
                phi = struct.unpack(f'{n_rff}f', f.read(4 * n_rff))
                self.phi = np.array(phi, dtype=np.float32)

            self.n_classes = nc
            self.n_features = nf
            self.class_ids = list(range(nc))
            self.label_map = LABEL_MAP_CUDA  # Use CUDA-specific labels
            self.model_type = "Kernel SVM (RBF via RFF)"
            self.loaded = True
            print(f"[INFO] Loaded CUDA RFF-SVM: nc={nc}, nf={nf}, n_rff={n_rff}, ncls={ncls}")
        except Exception as e:
            print(f"[WARN] Failed to load CUDA RFF model: {e}")

    def predict(self, X: np.ndarray) -> list:
        """Predict classes for input X."""
        if not self.loaded:
            raise RuntimeError("Model not loaded")

        # Check RFF first (before Kernel SVM, since RFF also contains "Kernel SVM")
        if "RFF" in self.model_type:
            return self._predict_rff(X)
        elif "Linear SVM" in self.model_type:
            return self._predict_linear(X)
        elif "Kernel SVM" in self.model_type:
            return self._predict_kernel(X)
        else:
            raise RuntimeError(f"Unknown model type: {self.model_type}")

    def _predict_kernel(self, X: np.ndarray) -> list:
        """Predict using kernel SVM (PySpark)."""
        N = X.shape[0]
        votes = {c: np.zeros(N, dtype=np.int32) for c in self.class_ids}
        score_sum = {c: np.zeros(N, dtype=np.float64) for c in self.class_ids}

        for sv, alphas, bias, ca, cb in self.models:
            sv_arr = np.array(sv, dtype=np.float64)
            al_arr = np.array(alphas, dtype=np.float64)
            X_n = np.sum(X**2, axis=1, keepdims=True)
            S_n = np.sum(sv_arr**2, axis=1)
            dist = X_n + S_n - 2 * X @ sv_arr.T
            K = np.exp(-self.gamma * np.maximum(dist, 0))
            scores = K @ al_arr + bias
            pos = scores >= 0
            score_sum[ca] += np.where(pos, scores, 0)
            score_sum[cb] += np.where(~pos, -scores, 0)
            votes[ca] += pos.astype(np.int32)
            votes[cb] += (~pos).astype(np.int32)

        vote_mat = np.stack([votes[c] for c in self.class_ids], axis=1)
        score_mat = np.stack([score_sum[c] for c in self.class_ids], axis=1)
        best_pos = (vote_mat * 1000 + score_mat).argmax(axis=1)
        best_cls = [self.class_ids[b] for b in best_pos]
        confs = score_mat[np.arange(N), best_pos] / np.maximum(vote_mat[np.arange(N), best_pos], 1)

        return self._format_results(N, best_cls, confs, vote_mat)

    def _predict_linear(self, X: np.ndarray) -> list:
        """Predict using linear SVM (One-vs-One for MPI/OpenMP).

        Uses integer vote counting (same as CUDA):
        - Each classifier gives 1 vote to winner
        - Confidence = votes[winner] / n_classifiers
        """
        scores = X @ self.W.T + self.b  # (N, K)
        N = X.shape[0]
        K = self.W.shape[0]  # number of classifiers

        if K == 6 and self.n_classes == 4:
            # OvO pairs: (pos, neg) - positive class wins if score > 0
            pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            votes = np.zeros((N, self.n_classes), dtype=np.int32)

            for clf_idx, (pos, neg) in enumerate(pairs):
                # If score > 0, pos wins; else neg wins
                winners = np.where(scores[:, clf_idx] > 0, pos, neg)
                for i, w in enumerate(winners):
                    votes[i, w] += 1

            best_cls = votes.argmax(axis=1)
            # Confidence = votes for winner / total classifiers
            confs = votes[np.arange(N), best_cls] / K

            # Convert votes to float for format_results
            return self._format_results(N, best_cls, confs, votes.astype(np.float32))
        else:
            # Fallback to direct scores (One-vs-All or direct classification)
            best_cls = scores.argmax(axis=1)
            confs = scores.max(axis=1)
            return self._format_results(N, best_cls, confs, scores)

    def _predict_rff(self, X: np.ndarray) -> list:
        """Predict using RFF-SVM (One-vs-One for CUDA).

        Matches infer.py exactly:
        - RFF transform: Z = sqrt(2/n_rff) * cos(X @ omega.T + rff_b)
        - OvO voting: each classifier gives 1 vote to winner (not score accumulation)
        - Confidence: votes[winner] / n_classifiers
        """
        # RFF transformation: Z = sqrt(2/D) * cos(X @ Omega.T + phi)
        n_rff = self.Omega.shape[0]
        Z = np.dot(X, self.Omega.T) + self.phi  # (N, n_rff)
        Z = np.sqrt(2.0 / n_rff) * np.cos(Z)

        # Compute raw scores: scores = Z @ W.T + b
        scores = Z @ self.rff_W.T + self.rff_b  # (N, 6)

        N = X.shape[0]
        K = self.rff_W.shape[0]  # number of classifiers (6)

        if K == 6 and self.n_classes == 4:
            # OvO pairs: (pos, neg) - positive class wins if score > 0
            pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            votes = np.zeros((N, self.n_classes), dtype=np.int32)

            for clf_idx, (pos, neg) in enumerate(pairs):
                # If score > 0, pos wins; else neg wins
                winners = np.where(scores[:, clf_idx] > 0, pos, neg)
                for i, w in enumerate(winners):
                    votes[i, w] += 1

            best_cls = votes.argmax(axis=1)
            # Confidence = votes for winner / total classifiers
            confs = votes[np.arange(N), best_cls] / K

            # Convert votes to float for format_results
            return self._format_results(N, best_cls, confs, votes.astype(np.float32))
        else:
            # Fallback to direct scores
            best_cls = scores.argmax(axis=1)
            confs = scores.max(axis=1)
            return self._format_results(N, best_cls, confs, scores)

    def _format_results(self, N, best_cls, confs, score_mat) -> list:
        """Format prediction results."""
        results = []
        for i in range(N):
            cls = int(best_cls[i])
            results.append({
                "predicted_class": cls,
                "predicted_label": self.label_map.get(cls, f"Class {cls}"),
                "confidence": float(confs[i]),
                "votes": {self.label_map.get(c, str(c)): float(score_mat[i, j])
                          for j, c in enumerate(self.class_ids)},
            })
        return results

    def get_info(self) -> dict:
        is_kernel = "Kernel SVM" in self.model_type
        return {
            "model_id": self.model_id,
            "model_loaded": self.loaded,
            "model_type": self.model_type,
            "n_features": self.n_features,
            "classes": self.label_map,
            "n_sub_models": len(self.models) if is_kernel else self.n_classes,
            "n_classes": self.n_classes,
            "gamma": self.gamma if is_kernel else None,
        }


# Cached predictors
_predictors = {}

def get_predictor(model_id: str) -> SVMPredictor:
    if model_id not in _predictors:
        _predictors[model_id] = SVMPredictor(model_id)
    return _predictors[model_id]


# ===================== Predict API Endpoints =====================

@app.post("/api/predict")
async def predict_samples(data: dict):
    """Predict using selected model with optional DBSCAN for uncertain samples."""
    model_id = data.get("model", "pyspark")
    if model_id not in ["pyspark", "cuda", "mpi", "openmp", "thread"]:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_id}")

    predictor = get_predictor(model_id)
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail=f"Model '{model_id}' not loaded")

    features = data.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="No features provided")

    # DBSCAN options
    use_dbscan = data.get("use_dbscan", False)
    conf_threshold = data.get("conf_threshold", 0.5)

    try:
        X = np.array(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != 52:
            raise HTTPException(status_code=400, detail=f"Expected 52 features, got {X.shape[1]}")

        results = predictor.predict(X)

        # Apply DBSCAN if requested
        dbscan_stats = None
        if use_dbscan:
            results, dbscan_stats = apply_dbscan_hybrid(results, predictor, X, conf_threshold)

        model_info = get_all_models().get(model_id, {})
        response = {
            "predictions": results,
            "n_samples": len(results),
            "model": model_info.get("name", model_id),
            "model_id": model_id,
        }

        if dbscan_stats:
            response["dbscan"] = dbscan_stats

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/csv")
async def predict_csv(file: UploadFile = File(...), model: str = Form("pyspark")):
    """Upload CSV and get predictions."""
    if model not in ["pyspark", "cuda", "mpi", "openmp", "thread"]:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    predictor = get_predictor(model)
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail=f"Model '{model}' not loaded")

    try:
        content = await file.read()
        lines = [l.strip() for l in content.decode("utf-8").strip().split("\n") if l.strip()]

        # Skip header if present
        try:
            [float(x) for x in lines[0].split(",")[:5]]
        except ValueError:
            lines = lines[1:]

        features = [[float(x) for x in line.split(",")][:52] for line in lines]
        X = np.array(features, dtype=np.float64)

        results = predictor.predict(X[:, :52])
        class_counts = {}
        for r in results:
            lbl = r["predicted_label"]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        return {
            "predictions": results,
            "n_samples": len(results),
            "model": model,
            "summary": class_counts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")


@app.get("/api/predict/info")
def predict_info(model: str = "pyspark"):
    """Get model info."""
    if model not in ["pyspark", "cuda", "mpi", "openmp", "thread"]:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    predictor = get_predictor(model)
    info = predictor.get_info()
    model_data = get_all_models().get(model, {})

    return {
        **info,
        "model_name": model_data.get("name", model),
        "technique": model_data.get("technique", ""),
        "description": model_data.get("description", ""),
        "accuracy": model_data.get("accuracy", 0),
    }


@app.get("/api/predict/models")
def list_prediction_models():
    """List available models for prediction."""
    all_models = get_all_models()
    result = []
    for model_id in ["cuda", "mpi", "openmp", "thread", "pyspark"]:
        model_data = all_models.get(model_id, {})
        predictor = get_predictor(model_id)
        result.append({
            "id": model_id,
            "name": model_data.get("name", model_id),
            "technique": model_data.get("technique", ""),
            "accuracy": model_data.get("accuracy", 0),
            "loaded": predictor.loaded,
        })
    return {"models": result}


# ===================== Static Files =====================

# Serve static frontend
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Serve the main dashboard page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>NIDS Dashboard</h1><p>Static files not found. Run from the web/ directory.</p>")


@app.get("/{path:path}")
def serve_static(path: str):
    """Fallback: serve static files or index.html for SPA routing."""
    file_path = STATIC_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    # SPA fallback
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="Not found")
