"""
Configuration & Constants
=========================
Central configuration for model thresholds, label mappings, and constants.
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Normal label variants across all model flavours
NORMAL_LABELS = {"Normal", "NormalTraffic"}

# Per-architecture confidence thresholds matching the C++ infer code:
#   cuda_infer.cu  CONF_THRESHOLD = 0.5   (conf = votes/3, uncertain if ≤1/3 pairs won)
#   mpi_infer.cpp  CONF_THRESHOLD = 0.8   (only perfect 3/3 wins are confident)
#   omp_infer.cpp  CONF_THRESHOLD = 0.67  (only perfect 3/3 wins are confident)
#   thread/pyspark CONF_THRESHOLD = 0.5
MODEL_CONF_THRESHOLDS = {
    "cuda":    0.5,
    "mpi":     0.8,
    "openmp":  0.6,
    "thread":  1.0,
    "pyspark": 0.5,
}

# Holdout class names (alphabetically sorted LabelEncoder IDs 0,1,6 from CICIDS2017)
HOLDOUT_CLASS_NAMES = {0: "Bots", 1: "BruteForce", 6: "WebAttacks"}

# Label maps for different models
LABEL_MAP_4CLASS = {0: "DDoS", 1: "DoS", 2: "NormalTraffic", 3: "PortScan"}
LABEL_MAP_CUDA = {0: "DDoS", 1: "DoS", 2: "Normal", 3: "PortScan"}  # CUDA uses "Normal" not "NormalTraffic"
LABEL_MAP_PYSPARK = {2: "DDoS", 3: "DoS", 4: "NormalTraffic", 5: "PortScan"}
