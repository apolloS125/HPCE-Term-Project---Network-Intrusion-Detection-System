"""
SVM Predictor
=============
Multi-model SVM predictor supporting linear, kernel, and RFF-based models.
"""

import struct
import pickle
from pathlib import Path

import numpy as np

from config import (
    PROJECT_ROOT,
    LABEL_MAP_4CLASS,
    LABEL_MAP_CUDA,
    LABEL_MAP_PYSPARK,
)


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

        return self._format_results(N, best_cls, confs, score_mat)

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
            votes     = np.zeros((N, self.n_classes), dtype=np.int32)
            score_sum = np.zeros((N, self.n_classes), dtype=np.float32)

            for clf_idx, (pos, neg) in enumerate(pairs):
                clf_scores = scores[:, clf_idx]
                pos_wins   = clf_scores > 0
                votes[:, pos] += pos_wins.astype(np.int32)
                votes[:, neg] += (~pos_wins).astype(np.int32)
                # Accumulate signed decision margin: winner gets |score|, loser gets 0
                score_sum[:, pos] += np.where(pos_wins,  clf_scores, 0).astype(np.float32)
                score_sum[:, neg] += np.where(~pos_wins, -clf_scores, 0).astype(np.float32)

            best_cls = votes.argmax(axis=1)
            # Confidence = votes for winner / max possible votes per class (n_classes - 1).
            # Each class only appears in (n_classes-1) OvO pairs, so max votes = 3, not K=6.
            # Dividing by K would cap confidence at 0.5; divide by (n_classes-1) gives [0, 1].
            confs = votes[np.arange(N), best_cls] / (self.n_classes - 1)

            # Pass accumulated decision margins so frontend shows real SVM scores
            return self._format_results(N, best_cls, confs, score_sum)
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
            votes     = np.zeros((N, self.n_classes), dtype=np.int32)
            score_sum = np.zeros((N, self.n_classes), dtype=np.float32)

            for clf_idx, (pos, neg) in enumerate(pairs):
                clf_scores = scores[:, clf_idx]
                pos_wins   = clf_scores > 0
                votes[:, pos] += pos_wins.astype(np.int32)
                votes[:, neg] += (~pos_wins).astype(np.int32)
                score_sum[:, pos] += np.where(pos_wins,  clf_scores, 0).astype(np.float32)
                score_sum[:, neg] += np.where(~pos_wins, -clf_scores, 0).astype(np.float32)

            best_cls = votes.argmax(axis=1)
            # Confidence = votes for winner / max possible votes per class (n_classes - 1).
            # Each class only appears in (n_classes-1) OvO pairs, so max votes = 3, not K=6.
            # Dividing by K would cap confidence at 0.5; divide by (n_classes-1) gives [0, 1].
            confs = votes[np.arange(N), best_cls] / (self.n_classes - 1)

            # Pass accumulated decision margins so frontend shows real SVM scores
            return self._format_results(N, best_cls, confs, score_sum)
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
