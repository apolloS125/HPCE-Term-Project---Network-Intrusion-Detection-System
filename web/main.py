"""
NIDS Dashboard — FastAPI Backend
=================================
Serves model data from cuda, mpi, openmp, pyspark, thread outputs.
Provides API endpoints for the frontend dashboard.
"""

import os
import csv
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

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

def get_all_models():
    """Collect all model data from output files."""
    models = {}

    # CUDA
    cuda_out = str(PROJECT_ROOT / "cuda_ids_3696.out")
    cuda_data = parse_output_file(cuda_out)
    cuda_acc = parse_accuracy_csv(str(PROJECT_ROOT / "cuda_accuracy.csv"))
    cuda_data["train_accuracy"] = cuda_acc["train_accuracy"]
    if cuda_data["accuracy"] == 0:
        cuda_data["accuracy"] = cuda_acc["test_accuracy"]
    cuda_data["technique"] = cuda_data["technique"] or "CUDA C++"
    cuda_data["name"] = "CUDA"
    cuda_data["description"] = "GPU-accelerated using NVIDIA CUDA. Prediction parallelized on RTX 2080 Ti."
    cuda_data["parallelism"] = "GPU (CUDA threads/blocks)"
    cuda_data["hardware"] = "NVIDIA GeForce RTX 2080 Ti"
    models["cuda"] = cuda_data

    # MPI
    mpi_out = str(PROJECT_ROOT / "mpi_ids_3704.out")
    mpi_data = parse_output_file(mpi_out)
    mpi_acc = parse_accuracy_csv(str(PROJECT_ROOT / "mpi_accuracy.csv"))
    mpi_data["train_accuracy"] = mpi_acc["train_accuracy"]
    if mpi_data["accuracy"] == 0:
        mpi_data["accuracy"] = mpi_acc["test_accuracy"]
    mpi_data["technique"] = mpi_data["technique"] or "MPI (4 processes)"
    mpi_data["name"] = "MPI"
    mpi_data["description"] = "Distributed-memory parallel using MPI. Training and prediction distributed across 4 processes."
    mpi_data["parallelism"] = "Distributed (4 MPI processes)"
    mpi_data["hardware"] = "Multi-node cluster"
    models["mpi"] = mpi_data

    # OpenMP
    openmp_out = str(PROJECT_ROOT / "openmp_ids_3703.out")
    openmp_data = parse_output_file(openmp_out)
    openmp_data["technique"] = openmp_data["technique"] or "OpenMP (8 threads)"
    openmp_data["name"] = "OpenMP"
    openmp_data["description"] = "Shared-memory parallel using OpenMP. Prediction parallelized with 8 threads."
    openmp_data["parallelism"] = "Shared-memory (8 OpenMP threads)"
    openmp_data["hardware"] = "Multi-core CPU"
    models["openmp"] = openmp_data

    # Thread
    thread_out = str(PROJECT_ROOT / "thread_ids_3705.out")
    thread_data = parse_output_file(thread_out)
    thread_acc = parse_accuracy_csv(str(PROJECT_ROOT / "thread_accuracy.csv"))
    thread_data["train_accuracy"] = thread_acc["train_accuracy"]
    if thread_data["accuracy"] == 0:
        thread_data["accuracy"] = thread_acc["test_accuracy"]
    thread_data["technique"] = thread_data["technique"] or "C++ Thread (8 threads)"
    thread_data["name"] = "C++ Thread"
    thread_data["description"] = "Shared-memory parallel using C++ std::thread. Prediction and DBSCAN parallelized with 8 threads."
    thread_data["parallelism"] = "Shared-memory (8 C++ threads)"
    thread_data["hardware"] = "Multi-core CPU"
    models["thread"] = thread_data

    # PySpark
    pyspark_summary = str(PROJECT_ROOT / "pyspark" / "pyspark_summary.csv")
    pyspark_data = parse_pyspark_summary(pyspark_summary)
    pyspark_data["name"] = "PySpark"
    pyspark_data["description"] = "Distributed computing using Apache Spark. SVM pairs trained in parallel across workers."
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
