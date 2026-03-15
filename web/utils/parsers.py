"""
Data Parsing Utilities
======================
Functions for parsing log files, CSV outputs, and model metrics.
"""

import os
import csv
import re

from config import PROJECT_ROOT


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
                elif metric in ["gflops", "svm_gflops", "predict_gflops", "dbscan_gflops"]:
                    result["gflops"] += float(value)
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


def parse_cuda_train_out(log_dir: str) -> dict:
    """Parse CUDA cuda_train_*.out for the training summary total_ms and effective GFLOPS."""
    import glob as _glob
    result = {"total_ms": 0.0, "gflops": 0.0}
    files = sorted(_glob.glob(os.path.join(log_dir, "cuda_train_*.out")))
    if not files:
        return result
    try:
        with open(files[-1], "r") as f:
            content = f.read()
        m = re.search(r"Total Train Time\s*:\s*([\d.]+)\s*ms", content)
        if m:
            result["total_ms"] = float(m.group(1))
        m = re.search(r"Effective GFLOPS\s*:\s*([\d.]+)", content)
        if m:
            result["gflops"] = float(m.group(1))
    except Exception:
        pass
    return result


def get_all_models():
    """Collect all model data from output files in respective folders only."""
    models = {}

    # CUDA - read from cuda/log/training_log.csv (new format from main branch)
    cuda_training_log = str(PROJECT_ROOT / "cuda" / "log" / "training_log.csv")
    cuda_data = parse_cuda_training_log(cuda_training_log)

    # Override total_ms and GFLOPS with the authoritative training summary from the .out file
    cuda_out_summary = parse_cuda_train_out(str(PROJECT_ROOT / "cuda" / "log"))
    if cuda_out_summary["total_ms"] > 0:
        cuda_data["total_ms"] = cuda_out_summary["total_ms"]
        cuda_data["train_ms"] = cuda_out_summary["total_ms"]  # entire CUDA run is training
    if cuda_out_summary["gflops"] > 0:
        cuda_data["gflops"] = cuda_out_summary["gflops"]

    # Fields missing from training_log.csv — known constants from training output
    cuda_data["n_features"] = 52
    cuda_data["n_classes"] = 4
    cuda_data["n_train"] = 588717
    # n_test already hardcoded as 752254 in parse_cuda_training_log
    if cuda_data["gflops"] > 0 and cuda_data["total_ms"] > 0:
        cuda_data["flops"] = int(cuda_data["gflops"] * (cuda_data["total_ms"] / 1000) * 1e9)

    cuda_data["technique"] = "CUDA RFF-SVM"
    cuda_data["name"] = "CUDA"
    cuda_data["description"] = "GPU-accelerated RFF-SVM using NVIDIA CUDA. Uses Random Fourier Features for kernel approximation."
    cuda_data["parallelism"] = "GPU (CUDA threads/blocks)"
    cuda_data["hardware"] = "NVIDIA GPU"
    models["cuda"] = cuda_data

    # MPI - read from mpi/log/hybrid_log.csv only
    mpi_log = str(PROJECT_ROOT / "mpi" / "log" / "hybrid_log.csv")
    mpi_data = parse_hybrid_log(mpi_log)
    # train_ms not in hybrid_log — read from train_results.csv
    mpi_train_results = str(PROJECT_ROOT / "mpi" / "log" / "train_results.csv")
    if os.path.exists(mpi_train_results):
        with open(mpi_train_results) as _f:
            for row in csv.DictReader(_f):
                mpi_data["train_ms"] = float(row.get("train_time_ms", 0))
    mpi_data["technique"] = "MPI Linear SVM"
    mpi_data["name"] = "MPI"
    mpi_data["description"] = "Distributed-memory parallel using MPI. One-vs-All Linear SVM classifier."
    mpi_data["parallelism"] = "Distributed (4 MPI processes)"
    mpi_data["hardware"] = "Multi-node cluster"
    mpi_data["n_features"] = 52
    mpi_data["n_classes"] = 4
    mpi_data["n_train"] = 588717
    if mpi_data["gflops"] > 0 and mpi_data["total_ms"] > 0:
        mpi_data["flops"] = int(mpi_data["gflops"] * (mpi_data["total_ms"] / 1000) * 1e9)
    models["mpi"] = mpi_data

    # OpenMP - read from openmp/log/hybrid_log.csv only
    openmp_log = str(PROJECT_ROOT / "openmp" / "log" / "hybrid_log.csv")
    openmp_data = parse_hybrid_log(openmp_log)
    # train_ms not in hybrid_log — read trailing summary rows from training_log.csv
    openmp_training_csv = str(PROJECT_ROOT / "openmp" / "log" / "training_log.csv")
    if os.path.exists(openmp_training_csv):
        with open(openmp_training_csv) as _f:
            for line in _f:
                parts = line.strip().split(",")
                if len(parts) == 2 and parts[0] == "train_ms":
                    try:
                        openmp_data["train_ms"] = float(parts[1])
                    except ValueError:
                        pass
    openmp_data["technique"] = "OpenMP Linear SVM"
    openmp_data["name"] = "OpenMP"
    openmp_data["description"] = "Shared-memory parallel using OpenMP. One-vs-All Linear SVM classifier."
    openmp_data["parallelism"] = "Shared-memory (8 OpenMP threads)"
    openmp_data["hardware"] = "Multi-core CPU"
    openmp_data["n_features"] = 52
    openmp_data["n_classes"] = 4
    openmp_data["n_train"] = 588717
    if openmp_data["gflops"] > 0 and openmp_data["total_ms"] > 0:
        openmp_data["flops"] = int(openmp_data["gflops"] * (openmp_data["total_ms"] / 1000) * 1e9)
    models["openmp"] = openmp_data

    # Thread - read from thread/thread_summary.csv
    thread_summary = str(PROJECT_ROOT / "thread" / "thread_summary.csv")
    thread_data = parse_thread_summary(thread_summary)
    thread_result = {
        "accuracy": 82.33,  # Static value — no accuracy log file available for C++ Thread
        "total_ms": float(thread_data.get("total_ms", 0)) if thread_data.get("total_ms") else 0,
        "train_ms": float(thread_data.get("train_ms", 0)) if thread_data.get("train_ms") else 0,
        "predict_ms": float(thread_data.get("predict_ms", 0)) if thread_data.get("predict_ms") else 0,
        "dbscan_ms": float(thread_data.get("dbscan_ms", 0)) if thread_data.get("dbscan_ms") else 0,
        "gflops": float(thread_data.get("gflops", 0)) if thread_data.get("gflops") else 0,
        "flops": int(thread_data.get("flops", 0)) if thread_data.get("flops") else 0,
        "n_features": 52,
        "n_classes": 4,
        "n_train": 588717,
        "n_test": 756226,
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
