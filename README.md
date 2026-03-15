# HPCE Term Project - Advanced Network Intrusion Detection System (NIDS)

An Enterprise-grade Hybrid Anomaly Detection system built with C++, CUDA, MPI, OpenMP, Threading, PySpark, and an Interactive Web Dashboard.

## Architecture Highlights

- **Parallel Inference Engines**: 5 distinct parallelization strategies evaluating Random Fourier Features (RFF) and One-vs-Rest (OvR) Support Vector Machines.
- **Hybrid Threat Detection**: Combines Supervised Learning (SVM) for known threats and Unsupervised Clustering (DBSCAN) for novel zero-day attacks.
- **FastAPI / Python Backend**: Serves predictions and clustering analytics to the client.
- **Dynamic Vanilla JS Dashboard**: Aesthetic, glassmorphism UI for visualizing real-time predictions and insights.
- **Novelty Detection (Zero-Day)**: Uses `dbscan_model.bin` thresholds to cluster uncertain behavioral patterns into new potential threats (Bots, BruteForce, WebAttacks).

## Project Structure

```text
hpce_code/
├── cuda/                       # CUDA GPU Accelerated inference
├── mpi/                        # MPI Distributed message-passing inference
├── openmp/                     # OpenMP Multi-core parallel inference
├── thread/                     # C++ std::thread POSIX parallel inference
├── pyspark/                    # PySpark Distributed inference cluster
└── web/                        # Dashboard & Backend API Server
    ├── config.py               # Constants, Thresholds, & Label Encoders
    ├── main.py                 # FastAPI Application Entrypoint
    ├── models/
    │   ├── dbscan.py           # Hybrid & Pure DBSCAN Logic
    │   └── predictor.py        # Centralized Model Wrapper API
    ├── static/                 # Frontend UI
    │   ├── index.html          # Dashboard Markup
    │   ├── app.js              # Application Logic & UI Interactions
    │   └── style.css           # Styling
```

## Dashboard Features

- **Inference Engine Selection**: Switch seamlessly between CUDA, MPI, OpenMP, Thread, and PySpark backends.
- **DBSCAN Anomaly Clustering**:
  - **Hybrid Mode**: Let the SVM classify high-confidence data, and only pass low-confidence (Uncertain) predictions down to DBSCAN for behavioral clustering.
  - **Pure DBSCAN (Unsupervised)**: Completely bypass the SVM. Run an auto-tuned, pure Unsupervised Learning algorithm directly on the raw 52-dimensional features to observe intrinsic cluster distributions.
- **Real-Time Threshold Tuning**: Interactive Confidence Threshold slider to control how strictly the SVM votes must agree before offloading to DBSCAN.
- **Detailed Threat Table**: View prediction results with explicit labels for `(Cluster)` identified patterns and `(Outlier)` unverified behavior.

## Quick Start (Web Dashboard)

**1. Install Backend Dependencies:**

```bash
cd web
pip install -r requirements.txt
```

**2. Start the FastAPI Server:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Access the Application:**

Open your browser and navigate to `http://localhost:8000`.

## Hybrid vs Pure DBSCAN Pipeline

1. **Preprocessing**: 52-dimensional features are normalized using StandardScaler mechanisms matching the C++ inference tools.
2. **SVM Classification (Hybrid Mode)**: Evaluates the features using pre-computed One-vs-Rest weights.
    - **High Confidence**: Handled directly as the targeted threat (or Normal traffic).
    - **Low Confidence**: Forwarded to the DBSCAN layer.
3. **DBSCAN Clustering**:
    - Analyzes uncertain data or raw payloads (if Pure mode is enabled).
    - Data points forming new dense structures are flagged as `Unknown (Cluster)` indicating a novel attack pattern.
    - Isolated data points are flagged as `Unknown (Outlier)` representing pure noise or unpredictable anomaly states.
