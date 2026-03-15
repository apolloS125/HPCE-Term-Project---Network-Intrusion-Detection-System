"""
NIDS Dashboard — FastAPI Backend
=================================
Serves model data from cuda, mpi, openmp, pyspark, thread outputs.
Provides API endpoints for the frontend dashboard.

Refactored into modular components:
- config.py: Constants & thresholds
- utils/parsers.py: Data parsing functions
- models/predictor.py: SVM prediction
- models/dbscan.py: DBSCAN clustering
"""

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import from local modules
from config import PROJECT_ROOT, MODEL_CONF_THRESHOLDS, HOLDOUT_CLASS_NAMES
from utils.parsers import (
    parse_epoch_errors,
    load_predictions_summary,
    get_all_models,
)
from models.predictor import get_predictor
from models.dbscan import apply_dbscan_only, apply_dbscan_hybrid


# FastAPI app
app = FastAPI(title="NIDS Dashboard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disable browser caching for JS/CSS so changes are picked up immediately on refresh
@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") and (path.endswith(".js") or path.endswith(".css")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


# ===================== Model Data API Endpoints =====================

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


# ===================== Prediction API Endpoints =====================

@app.post("/api/predict")
async def predict_samples(data: dict):
    """Predict using selected model with optional DBSCAN for uncertain samples."""
    model_id = data.get("model", "pyspark")
    if model_id not in ["pyspark", "cuda", "mpi", "openmp", "thread", "dbscan"]:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_id}")

    if model_id != "dbscan":
        predictor = get_predictor(model_id)
        if not predictor.loaded:
            raise HTTPException(status_code=503, detail=f"Model '{model_id}' not loaded")
    else:
        predictor = None

    features = data.get("features", [])
    if not features:
        raise HTTPException(status_code=400, detail="No features provided")

    # DBSCAN options
    use_dbscan = data.get("use_dbscan", False)
    dbscan_mode = data.get("dbscan_mode", "hybrid")  # "hybrid" or "only"
    conf_threshold = data.get("conf_threshold", None)  # None = use MODEL_CONF_THRESHOLDS

    try:
        X = np.array(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != 52:
            raise HTTPException(status_code=400, detail=f"Expected 52 features, got {X.shape[1]}")

        # Apply DBSCAN if requested
        dbscan_stats = None
        if model_id == "dbscan" or (use_dbscan and dbscan_mode == "only"):
            # DBSCAN-only mode: skip SVM voting entirely
            results, dbscan_stats = apply_dbscan_only(predictor, X)
        elif use_dbscan:
            # Hybrid mode: SVM first, then DBSCAN on uncertain samples
            results = predictor.predict(X)
            results, dbscan_stats = apply_dbscan_hybrid(results, predictor, X, conf_threshold)
        else:
            # SVM only (no DBSCAN)
            results = predictor.predict(X)

        if model_id == "dbscan":
            model_info = {"name": "Pure DBSCAN (Unsupervised)"}
        else:
            model_info = get_all_models().get(model_id, {})
            
        response = {
            "predictions": results,
            "n_samples": len(results),
            "model": model_info.get("name", model_id),
            "model_id": model_id,
        }

        if dbscan_stats:
            response["dbscan"] = dbscan_stats

            # Add detailed DBSCAN result breakdown
            if dbscan_stats.get("mode") == "dbscan_only":
                # DBSCAN-only mode breakdown
                in_cluster = [r for r in results if not r.get("is_outlier", False)]
                outliers = [r for r in results if r.get("is_outlier", False)]

                response["dbscan_breakdown"] = {
                    "mode": "dbscan_only",
                    "holdout_classes": HOLDOUT_CLASS_NAMES,
                    "clustered_samples": {
                        "count": len(in_cluster),
                        "percentage": round(len(in_cluster) / len(results) * 100, 2) if results else 0,
                        "samples": in_cluster,
                        "description": "Samples classified by DBSCAN clusters"
                    },
                    "outliers": {
                        "count": len(outliers),
                        "percentage": round(len(outliers) / len(results) * 100, 2) if results else 0,
                        "samples": outliers,
                        "description": f"Novel patterns (potential holdout attacks: {', '.join(HOLDOUT_CLASS_NAMES.values())})"
                    }
                }
            else:
                # Hybrid mode breakdown
                confident_preds = [r for r in results if not r.get("is_uncertain", False)]
                uncertain_normal_preds = [r for r in results if r.get("is_uncertain", False) and not r.get("is_novel", False) and "Normal" in r["predicted_label"]]
                dbscan_verified = [r for r in results if "DBSCAN verified" in r["predicted_label"]]
                novel_unknown = [r for r in results if r.get("is_novel", False)]

                response["dbscan_breakdown"] = {
                    "mode": "hybrid",
                    "holdout_classes": HOLDOUT_CLASS_NAMES,
                    "confident_predictions": {
                        "count": len(confident_preds),
                        "percentage": round(len(confident_preds) / len(results) * 100, 2) if results else 0,
                        "samples": confident_preds
                    },
                    "uncertain_normal": {
                        "count": len(uncertain_normal_preds),
                        "percentage": round(len(uncertain_normal_preds) / len(results) * 100, 2) if results else 0,
                        "samples": uncertain_normal_preds
                    },
                    "dbscan_verified_attacks": {
                        "count": len(dbscan_verified),
                        "percentage": round(len(dbscan_verified) / len(results) * 100, 2) if results else 0,
                        "samples": dbscan_verified
                    },
                    "novel_unknown_attacks": {
                        "count": len(novel_unknown),
                        "percentage": round(len(novel_unknown) / len(results) * 100, 2) if results else 0,
                        "samples": novel_unknown,
                        "description": f"Potential holdout attacks: {', '.join(HOLDOUT_CLASS_NAMES.values())}"
                    }
                }

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/api/predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    model: str = Form("pyspark"),
    use_dbscan: str = Form("false"),
    dbscan_mode: str = Form("hybrid"),  # "hybrid" or "only"
    conf_threshold: float = Form(None),  # None = use MODEL_CONF_THRESHOLDS
):
    """Upload CSV and get predictions, with optional DBSCAN detection (hybrid or only mode)."""
    if model not in ["pyspark", "cuda", "mpi", "openmp", "thread", "dbscan"]:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

    if model != "dbscan":
        predictor = get_predictor(model)
        if not predictor.loaded:
            raise HTTPException(status_code=503, detail=f"Model '{model}' not loaded")
    else:
        predictor = None

    dbscan_enabled = use_dbscan.strip().lower() in ("true", "1", "yes")

    # Use model-specific threshold if not provided
    threshold = conf_threshold if conf_threshold is not None else MODEL_CONF_THRESHOLDS.get(model, 0.5)

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

        # Apply DBSCAN if requested
        dbscan_stats = None
        if model == "dbscan" or (dbscan_enabled and dbscan_mode == "only"):
            # DBSCAN-only mode
            results, dbscan_stats = apply_dbscan_only(predictor, X[:, :52])
        elif dbscan_enabled:
            # Hybrid mode
            results = predictor.predict(X[:, :52])
            results, dbscan_stats = apply_dbscan_hybrid(results, predictor, X[:, :52], threshold)
        else:
            # SVM only
            results = predictor.predict(X[:, :52])

        class_counts = {}
        for r in results:
            lbl = r["predicted_label"]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        response = {
            "predictions": results,
            "n_samples": len(results),
            "model": model,
            "summary": class_counts,
        }
        if dbscan_stats:
            response["dbscan"] = dbscan_stats

            # Add detailed DBSCAN result breakdown
            if dbscan_stats.get("mode") == "dbscan_only":
                # DBSCAN-only mode breakdown
                in_cluster = [r for r in results if not r.get("is_outlier", False)]
                outliers = [r for r in results if r.get("is_outlier", False)]

                response["dbscan_breakdown"] = {
                    "mode": "dbscan_only",
                    "holdout_classes": HOLDOUT_CLASS_NAMES,
                    "clustered_samples": {
                        "count": len(in_cluster),
                        "percentage": round(len(in_cluster) / len(results) * 100, 2) if results else 0,
                        "description": "Samples classified by DBSCAN clusters"
                    },
                    "outliers": {
                        "count": len(outliers),
                        "percentage": round(len(outliers) / len(results) * 100, 2) if results else 0,
                        "description": f"Novel patterns (potential holdout attacks: {', '.join(HOLDOUT_CLASS_NAMES.values())})"
                    }
                }
            else:
                # Hybrid mode breakdown
                confident_preds = [r for r in results if not r.get("is_uncertain", False)]
                uncertain_normal_preds = [r for r in results if r.get("is_uncertain", False) and not r.get("is_novel", False) and "Normal" in r["predicted_label"]]
                dbscan_verified = [r for r in results if "DBSCAN verified" in r["predicted_label"]]
                novel_unknown = [r for r in results if r.get("is_novel", False)]

                response["dbscan_breakdown"] = {
                    "mode": "hybrid",
                    "holdout_classes": HOLDOUT_CLASS_NAMES,
                    "confident_predictions": {
                        "count": len(confident_preds),
                        "percentage": round(len(confident_preds) / len(results) * 100, 2) if results else 0
                    },
                    "uncertain_normal": {
                        "count": len(uncertain_normal_preds),
                        "percentage": round(len(uncertain_normal_preds) / len(results) * 100, 2) if results else 0
                    },
                    "dbscan_verified_attacks": {
                        "count": len(dbscan_verified),
                        "percentage": round(len(dbscan_verified) / len(results) * 100, 2) if results else 0
                    },
                    "novel_unknown_attacks": {
                        "count": len(novel_unknown),
                        "percentage": round(len(novel_unknown) / len(results) * 100, 2) if results else 0,
                        "description": f"Potential holdout attacks: {', '.join(HOLDOUT_CLASS_NAMES.values())}"
                    }
                }

        return response
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
        "model_name":    model_data.get("name", model),
        "technique":     model_data.get("technique", ""),
        "description":   model_data.get("description", ""),
        "accuracy":      model_data.get("accuracy", 0),
        "conf_threshold": MODEL_CONF_THRESHOLDS.get(model, 0.5),
        "holdout_classes": list(HOLDOUT_CLASS_NAMES.values()),
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
