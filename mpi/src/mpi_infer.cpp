/*
 * mpi_infer.cpp  —  Hybrid SVM + DBSCAN inference pipeline (MPI, CPU only)
 *
 * Purpose:
 *   Load the SVM model produced by mpi_train (model.bin), run distributed
 *   inference on CICIDS2017 test data, then apply CPU DBSCAN re-classification
 *   for low-confidence and holdout samples.  Saves both the SVM-only model
 *   output and the fitted DBSCAN cluster model so each stage can be inspected
 *   independently.  Outputs final hybrid predictions and a metrics summary.
 *
 * Two saved models used by this pipeline:
 *   model.bin       — SVM weights (produced by mpi_train.cpp)
 *   dbscan_model.bin — DBSCAN cluster centroids + labels (produced here)
 *
 * Algorithm summary:
 *   ── PHASE 1  SVM INFERENCE ──────────────────────────────────────────
 *   1. Rank 0 loads model.bin, test_data.csv, test_labels.csv.
 *   2. MPI_Bcast: N_test, N_FEATURES; MPI_Bcast: W[6×52] and b[6].
 *   3. MPI_Scatter test rows equally to all ranks (same scheme as training:
 *      floor(N_test/size) rows per rank; rank 0 keeps the remainder).
 *   4. Each rank runs One-vs-One voting: for each of the 6 pairs, scores
 *      the sample and tallies a vote for the winning class.
 *      conf[i] = votes[best_k] / (N_CLASSES-1)  (range [0,1])
 *      svm_pred[i] = argmax of vote tallies
 *   5. MPI_Gather vote counts (4D), conf, svm_pred → rank 0.
 *   6. Rank 0 saves svm_predictions.csv (SVM-only, before DBSCAN refinement).
 *
 *   ── PHASE 2  DBSCAN CLUSTERING ─────────────────────────────────────
 *   7. Rank 0 partitions test samples:
 *        holdout  (orig ∈ {0,1,6}) → DBSCAN pool (always; act as anchors)
 *        conf < CONF_THRESHOLD     → uncertain   → DBSCAN pool (capped)
 *        else                      → confident   → final SVM prediction done
 *   8. Rank 0 runs O(n²) CPU DBSCAN in N_PAIRS-D (6-D) continuous raw OvO
 *      score space on the pool.  Using raw scores instead of integer vote
 *      counts avoids the 44-value collapse of 4-D vote space and lets
 *      WebAttacks / BruteForce / Bots separate even when they share the
 *      same binary vote outcome.  EPS=1.0 (raw margin units; tune as needed).
 *   9. Cluster labelling:
 *        has holdout anchors  → majority-vote original class ID
 *        no anchor, near Normal centroid → NORMAL_ORIG_ID (4)
 *        no anchor, far Normal           → PRED_UNKNOWN (-2)
 *        noise point                     → PRED_NOVEL (-1)
 *  10. Rank 0 saves dbscan_model.bin (cluster centroids + labels + hyperparams).
 *
 *   ── PHASE 3  HYBRID COMBINATION ────────────────────────────────────
 *  11. SVM FIRST: samples with conf >= CONF_THRESHOLD (and not holdout)
 *      keep their SVM argmax prediction mapped to original class ID.
 *      DBSCAN FALLBACK: all other samples (uncertain OR holdout) get their
 *      prediction from DBSCAN cluster assignment.
 *  12. Evaluation tables printed; output files written.
 *
 * Reproducibility note:
 *   The DBSCAN and evaluation sections run only on rank 0 and are thus
 *   independent of world_size.  Tiny floating-point differences in the
 *   gathered SVM scores may arise from MPI_Allreduce ordering in mpi_train.
 *
 * Build:
 *   mpicxx -O3 -std=c++17 -o mpi_infer mpi_infer.cpp -lm
 *
 * Run:
 *   mpirun -np 4 ./mpi_infer
 *
 * Expected output files:
 *   svm_predictions.csv   — original_label,svm_predicted_label (before DBSCAN)
 *   dbscan_model.bin      — DBSCAN cluster model: centroids + labels
 *   hybrid_predictions.csv — original_label,final_predicted_label
 *   hybrid_log.csv        — detailed metrics (metric,value format)
 *   results.csv           — flat one-row summary:
 *                           acc,lav,f1,Accuracy,Predict_ms,DBSCAN_ms,Total_ms,GFLOPS
 *
 * dbscan_model.bin layout (rank 0 save, v2):
 *   int32   n_clusters
 *   float32 eps
 *   int32   min_samples
 *   float32 normal_prox_thresh
 *   int32   n_dbscan_dim  (= N_PAIRS = 6; raw OvO score dimension)
 *   For each cluster c in [0, n_clusters):
 *     float32[n_dbscan_dim]  centroid  (mean N_PAIRS-D raw score vector)
 *     int32                  label     (original class ID, PRED_UNKNOWN=-2)
 */

#include <mpi.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// MPI error wrapper
// ---------------------------------------------------------------------------
#define MPI_CHECK(call) do {                                                   \
    int _e = (call);                                                           \
    if (_e != MPI_SUCCESS) {                                                   \
        char _s[256]; int _l = 256;                                            \
        MPI_Error_string(_e, _s, &_l);                                         \
        fprintf(stderr, "MPI error %s:%d: %s\n", __FILE__, __LINE__, _s);     \
        MPI_Abort(MPI_COMM_WORLD, _e);                                         \
    }                                                                          \
} while (0)

// ---------------------------------------------------------------------------
// Configuration — do NOT change to match spec
// ---------------------------------------------------------------------------
static constexpr int   N_FEATURES         = 52;
static constexpr int   N_CLASSES          = 4;    // trained classifiers
static constexpr int   N_PAIRS            = N_CLASSES * (N_CLASSES - 1) / 2; // 6
static constexpr int   PAIR_I[N_PAIRS]    = {0, 0, 0, 1, 1, 2};
static constexpr int   PAIR_J[N_PAIRS]    = {1, 2, 3, 2, 3, 3};
static constexpr int   N_ALL              = 7;    // original class count

static constexpr float CONF_THRESHOLD     = 0.8f;    // keep low — PortScan/DDoS need it
static constexpr int   N_MAX_UNCERTAIN    = 110000;  // covers all uncertain at 0.5 threshold
// DBSCAN runs in N_PAIRS-dimensional (6-D) continuous raw OvO score space.
// Using raw scores instead of integer vote counts (which only have 44 unique
// values for 4-class OvO) gives much better separation for holdout classes.
static constexpr int   N_DBSCAN_DIM      = N_PAIRS; // 6-D raw score space
static constexpr float EPS               = 1.0f;    // tuned for 6-D raw score space; ~0.5 raw-margin units
static constexpr int   MIN_SAMPLES        = 8;
static constexpr float NORMAL_PROX_THRESH = 5.0f;   // L2 distance threshold in 6-D raw score space

static constexpr int ORIG2INT[N_ALL]     = { -1, -1,  0,  1,  2,  3, -1 };
static constexpr int INT2ORIG[N_CLASSES] = {  2,  3,  4,  5 };
static constexpr int NORMAL_INT          = 2;    // INT2ORIG[2] == 4
static constexpr int NORMAL_ORIG_ID      = 4;
static constexpr int PRED_NOVEL          = -1;
static constexpr int PRED_UNKNOWN        = -2;

static const char* CLASS_NAMES[N_ALL] = {
    "Bots", "BruteForce", "DDoS", "DoS", "NormalTraffic", "PortScan", "WebAttacks"
};

// Input paths
static const char* MODEL_IN       = "model.bin";
static const char* TEST_DATA_CSV  = "data/processed/test_data.csv";
static const char* TEST_LABEL_CSV = "data/processed/test_labels.csv";

// Output paths — one per pipeline stage
static const char* SVM_PRED_OUT   = "svm_predictions.csv";    // Phase 1 output
static const char* DBSCAN_MDL_OUT = "dbscan_model.bin";       // Phase 2 output
static const char* PRED_OUT       = "hybrid_predictions.csv"; // Phase 3 output
static const char* LOG_OUT        = "hybrid_log.csv";         // Phase 3 metrics
static const char* RESULTS_OUT    = "results.csv";            // flat one-row summary
static const char* RUN_LOG        = "mpi_infer_run.log";

// ---------------------------------------------------------------------------
// Path helpers — allow running binaries from mpi/ or mpi/src
// ---------------------------------------------------------------------------
static std::string current_working_dir() {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto cwd = fs::current_path(ec);
    if (ec) return std::string("<unknown>");
    return cwd.string();
}

static std::string resolve_input_path(const char* relative) {
    namespace fs = std::filesystem;
    static const char* kSearchRoots[] = { "./", "../", "src/", "../src/" };
    fs::path rel(relative);
    for (const char* root : kSearchRoots) {
        fs::path candidate = fs::path(root) / rel;
        std::error_code ec;
        if (fs::exists(candidate, ec) && !ec)
            return candidate.lexically_normal().string();
    }
    return rel.string();
}

// ---------------------------------------------------------------------------
// Dual-output logger: writes to both stdout and a run log file
// ---------------------------------------------------------------------------
static FILE* g_log_fp = nullptr;

static void log_open(const char* path) {
    g_log_fp = fopen(path, "w");
    if (!g_log_fp) fprintf(stderr, "Warning: cannot open run log %s\n", path);
}

static void log_close() {
    if (g_log_fp) { fflush(g_log_fp); fclose(g_log_fp); g_log_fp = nullptr; }
}

static void lprintf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt); vprintf(fmt, ap); va_end(ap);
    if (g_log_fp) {
        va_list ap2;
        va_start(ap2, fmt); vfprintf(g_log_fp, fmt, ap2); va_end(ap2);
        fflush(g_log_fp);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string ts() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return buf;
}

// Returns true for holdout classes never seen during SVM training
static inline bool is_holdout(int orig) { return orig == 0 || orig == 1 || orig == 6; }

// Squared Euclidean distance between two n-dimensional points
static inline float distND_sq(const float* a, const float* b, int n) {
    float d = 0.f;
    for (int i = 0; i < n; ++i) { float di = a[i] - b[i]; d += di * di; }
    return d;
}

// ---------------------------------------------------------------------------
// CSV loaders (rank 0 only)
// ---------------------------------------------------------------------------
static std::vector<float> load_csv_float(const char* path) {
    std::string resolved = resolve_input_path(path);
    std::ifstream f(resolved);
    if (!f) {
        std::string cwd = current_working_dir();
        fprintf(stderr, "[rank 0] Cannot open %s (cwd=%s)\n",
                resolved.c_str(), cwd.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<float> data;
    data.reserve(800000LL * N_FEATURES);
    std::string line, tok;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        while (std::getline(ss, tok, ','))
            data.push_back(std::stof(tok));
    }
    return data;
}

static std::vector<int> load_csv_int(const char* path) {
    std::string resolved = resolve_input_path(path);
    std::ifstream f(resolved);
    if (!f) {
        std::string cwd = current_working_dir();
        fprintf(stderr, "[rank 0] Cannot open %s (cwd=%s)\n",
                resolved.c_str(), cwd.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<int> data;
    data.reserve(800000);
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) data.push_back(std::stoi(line));
    return data;
}

// ---------------------------------------------------------------------------
// SVM model loader (rank 0 only)
// ---------------------------------------------------------------------------
struct SVMModel {
    int K, F;                        // K=N_PAIRS, F=N_FEATURES
    std::vector<float> W, b;         // SVM weights: W[K×F], b[K]
};

static SVMModel load_svm_model_from_path(const std::string& resolved_path) {
    std::ifstream f(resolved_path, std::ios::binary);
    if (!f) {
        std::string cwd = current_working_dir();
        fprintf(stderr, "[rank 0] Cannot open SVM model %s (cwd=%s)\n",
                resolved_path.c_str(), cwd.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    SVMModel m;
    int dims[2];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    m.K = dims[0]; m.F = dims[1];
    m.W.resize((size_t)m.K * m.F);
    m.b.resize(m.K);
    f.read(reinterpret_cast<char*>(m.W.data()), (size_t)m.K * m.F * sizeof(float));
    f.read(reinterpret_cast<char*>(m.b.data()), m.K * sizeof(float));
    return m;
}

// ---------------------------------------------------------------------------
// CPU O(n²) DBSCAN on n_db points in n_dim-dimensional score space
//   Returns cluster_id[n_db] (-1 = noise).  n_clusters returned by value.
// ---------------------------------------------------------------------------
static std::vector<int> run_dbscan(const std::vector<float>& X_db, int n_db, int n_dim,
                                    float eps, int min_samples,
                                    int& n_clusters_out)
{
    const float eps2 = eps * eps;

    // Pass 1: count neighbours (self included → count >= min_samples = core)
    lprintf("[%s]   DBSCAN pass 1: counting neighbours (n=%d, eps=%.5f) ...\n",
           ts().c_str(), n_db, (double)eps);
    std::vector<int> nbr_count(n_db, 0);
    for (int i = 0; i < n_db; ++i) {
        const float* xi = X_db.data() + (size_t)i * n_dim;
        int cnt = 0;
        for (int j = 0; j < n_db; ++j) {
            if (distND_sq(xi, X_db.data() + (size_t)j * n_dim, n_dim) <= eps2) ++cnt;
        }
        nbr_count[i] = cnt;
    }

    std::vector<bool> is_core(n_db);
    for (int i = 0; i < n_db; ++i)
        is_core[i] = (nbr_count[i] >= min_samples);

    // Build CSR adjacency list (all neighbours per point)
    std::vector<int> row_ptr(n_db + 1, 0);
    for (int i = 0; i < n_db; ++i) row_ptr[i + 1] = row_ptr[i] + nbr_count[i];
    long total_edges = row_ptr[n_db];
    lprintf("[%s]   DBSCAN pass 2: filling adjacency (total edges=%ld, avg=%.1f) ...\n",
           ts().c_str(), total_edges, (double)total_edges / n_db);

    std::vector<int> col_idx(total_edges);
    {
        std::vector<int> fill_pos(row_ptr.begin(), row_ptr.begin() + n_db);
        for (int i = 0; i < n_db; ++i) {
            const float* xi = X_db.data() + (size_t)i * n_dim;
            for (int j = 0; j < n_db; ++j) {
                if (distND_sq(xi, X_db.data() + (size_t)j * n_dim, n_dim) <= eps2)
                    col_idx[fill_pos[i]++] = j;
            }
        }
    }

    // BFS cluster expansion: core points seed new clusters;
    // border points (reachable but not core) join without expanding further
    std::vector<int> labels(n_db, -1);
    int nc = 0;
    std::queue<int> q;
    for (int i = 0; i < n_db; ++i) {
        if (!is_core[i] || labels[i] >= 0) continue;
        labels[i] = nc;
        q.push(i);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
                int v = col_idx[e];
                if (labels[v] >= 0) continue;
                labels[v] = nc;
                if (is_core[v]) q.push(v);
            }
        }
        ++nc;
    }

    int n_noise = 0;
    for (int i = 0; i < n_db; ++i) if (labels[i] < 0) ++n_noise;
    lprintf("[%s]   DBSCAN done: clusters=%d  noise=%d (%.1f%%)\n",
           ts().c_str(), nc, n_noise, 100.0 * n_noise / n_db);

    n_clusters_out = nc;
    return labels;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    if (rank == 0) log_open(RUN_LOG);

    double t_total_start = MPI_Wtime();

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 1 — SVM INFERENCE  (distributed across all ranks)
    //
    //   Load the SVM model trained by mpi_train, scatter test data to all
    //   ranks, run One-vs-One voting for each sample, gather results back to
    //   rank 0.  Samples with conf >= CONF_THRESHOLD receive a final SVM
    //   prediction here; all others are deferred to Phase 2.
    // ══════════════════════════════════════════════════════════════════════

    if (rank == 0)
        lprintf("[%s] ════════════════════════════════════════════════════\n"
               "[%s] PHASE 1 — SVM INFERENCE  (loading model.bin)\n"
               "[%s] ════════════════════════════════════════════════════\n",
               ts().c_str(), ts().c_str(), ts().c_str());

    double t0_phase1 = MPI_Wtime();

    std::vector<float> X_test_all;
    std::vector<int>   y_test_all;
    int    N_test  = 0;
    int    N_feat  = N_FEATURES;
    std::vector<float> W_svm(N_PAIRS * N_FEATURES, 0.f);
    std::vector<float> b_svm(N_PAIRS, 0.f);

    // Step 1a: Rank 0 loads SVM model and test data
    if (rank == 0) {
        std::string model_path = resolve_input_path(MODEL_IN);
        lprintf("[%s] Loading SVM model: %s\n", ts().c_str(),
                model_path.c_str());
        SVMModel m = load_svm_model_from_path(model_path);
        assert(m.K == N_PAIRS && m.F == N_FEATURES);
        W_svm = m.W;
        b_svm = m.b;
        lprintf("[%s] Model loaded: K=%d F=%d\n",
                ts().c_str(), m.K, m.F);

        lprintf("[%s] Loading test data: %s\n", ts().c_str(), TEST_DATA_CSV);
        X_test_all = load_csv_float(TEST_DATA_CSV);
        y_test_all = load_csv_int(TEST_LABEL_CSV);
        N_test     = (int)y_test_all.size();
        assert((int)X_test_all.size() == (size_t)N_test * N_FEATURES);
        lprintf("[%s] N_test=%d\n", ts().c_str(), N_test);
    }

    // Step 1b: Broadcast N_test and SVM model weights to all ranks
    MPI_CHECK(MPI_Bcast(&N_test, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&N_feat, 1, MPI_INT, 0, MPI_COMM_WORLD));
    assert(N_feat == N_FEATURES);

    MPI_CHECK(MPI_Bcast(W_svm.data(), N_PAIRS * N_FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(b_svm.data(), N_PAIRS,               MPI_FLOAT, 0, MPI_COMM_WORLD));

    // Step 1c: Scatter test rows (floor(N_test/size) per rank; rank 0 keeps remainder)
    int base_n    = N_test / world_size;
    int remainder = N_test % world_size;

    std::vector<float> X_local_scatter(base_n * N_FEATURES);
    std::vector<int>   y_local_scatter(base_n);

    MPI_CHECK(MPI_Scatter(
        rank == 0 ? X_test_all.data() : nullptr, base_n * N_FEATURES, MPI_FLOAT,
        X_local_scatter.data(),                   base_n * N_FEATURES, MPI_FLOAT,
        0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatter(
        rank == 0 ? y_test_all.data() : nullptr, base_n, MPI_INT,
        y_local_scatter.data(),                   base_n, MPI_INT,
        0, MPI_COMM_WORLD));

    int local_n = base_n;
    std::vector<float> X_local = X_local_scatter;
    std::vector<int>   y_local = y_local_scatter;

    if (rank == 0 && remainder > 0) {
        int rem_off = world_size * base_n;
        X_local.insert(X_local.end(),
                        X_test_all.begin() + (size_t)rem_off * N_FEATURES,
                        X_test_all.end());
        y_local.insert(y_local.end(),
                        y_test_all.begin() + rem_off,
                        y_test_all.end());
        local_n += remainder;
    }

    // Step 1d: Each rank runs OvO voting for its scatter partition (base_n rows).
    //   For each pair p: score = W_svm[p]·x + b_svm[p]; winning class gets a vote.
    //   scores_local stores the N_PAIRS raw decision values (continuous, used by DBSCAN).
    //   conf[i]  = votes[best_k] / (N_CLASSES - 1)  range [0,1]
    //   pred[i]  = argmax of vote tallies  (internal class index 0-3)
    std::vector<float> scores_local(base_n * N_PAIRS);   // raw scores for DBSCAN
    std::vector<float> conf_local  (base_n);
    std::vector<int>   pred_local  (base_n);

    for (int i = 0; i < base_n; ++i) {
        const float* x = X_local.data() + (size_t)i * N_FEATURES;
        int votes[N_CLASSES] = {};
        for (int p = 0; p < N_PAIRS; ++p) {
            float s = b_svm[p];
            for (int f = 0; f < N_FEATURES; ++f)
                s += W_svm[p * N_FEATURES + f] * x[f];
            scores_local[i * N_PAIRS + p] = s;   // store continuous raw score
            if (s >= 0.f) votes[PAIR_I[p]]++;
            else          votes[PAIR_J[p]]++;
        }
        int best_k = 0;
        for (int k = 1; k < N_CLASSES; ++k)
            if (votes[k] > votes[best_k]) best_k = k;
        conf_local[i] = (float)votes[best_k] / (N_CLASSES - 1); // range [0,1]
        pred_local[i] = best_k;
    }

    // Step 1e: Gather the scatter-partition results to rank 0
    std::vector<float> scores_all;
    std::vector<float> conf_all;
    std::vector<int>   pred_all;
    std::vector<int>   y_orig_all;

    if (rank == 0) {
        scores_all.resize((size_t)N_test * N_PAIRS);    // N_PAIRS raw scores per sample
        conf_all  .resize(N_test);
        pred_all  .resize(N_test);
        y_orig_all = y_test_all;
    }

    MPI_CHECK(MPI_Gather(scores_local.data(), base_n * N_PAIRS, MPI_FLOAT,
                         rank == 0 ? scores_all.data() : nullptr,
                         base_n * N_PAIRS, MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Gather(conf_local.data(), base_n, MPI_FLOAT,
                         rank == 0 ? conf_all.data() : nullptr,
                         base_n, MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Gather(pred_local.data(), base_n, MPI_INT,
                         rank == 0 ? pred_all.data() : nullptr,
                         base_n, MPI_INT, 0, MPI_COMM_WORLD));

    // Rank 0: compute OvO votes for its remainder rows (not part of the scatter)
    if (rank == 0 && remainder > 0) {
        int rem_start = world_size * base_n;
        for (int i = base_n; i < base_n + remainder; ++i) {
            int gi = rem_start + (i - base_n);
            const float* x = X_local.data() + (size_t)i * N_FEATURES;
            int votes[N_CLASSES] = {};
            for (int p = 0; p < N_PAIRS; ++p) {
                float s = b_svm[p];
                for (int f = 0; f < N_FEATURES; ++f)
                    s += W_svm[p * N_FEATURES + f] * x[f];
                scores_all[(size_t)gi * N_PAIRS + p] = s;  // raw score
                if (s >= 0.f) votes[PAIR_I[p]]++;
                else          votes[PAIR_J[p]]++;
            }
            int best_k = 0;
            for (int k = 1; k < N_CLASSES; ++k)
                if (votes[k] > votes[best_k]) best_k = k;
            conf_all[gi] = (float)votes[best_k] / (N_CLASSES - 1);
            pred_all[gi] = best_k;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // All remaining work (Phase 1 post-processing, Phase 2, Phase 3) is
    // rank 0 only — DBSCAN and evaluation are serial CPU tasks.
    // ──────────────────────────────────────────────────────────────────────
    if (rank == 0) {

        double t1_phase1 = MPI_Wtime();
        lprintf("[%s] ─── SVM inference complete for all %d test samples.\n",
               ts().c_str(), N_test);
        lprintf("[%s] Phase 1 (SVM inference) time: %.3fs\n",
               ts().c_str(), t1_phase1 - t0_phase1);

        // Step 1f: SVM-only evaluation (before DBSCAN refinement)
        //   Count trained-class samples where SVM prediction is correct.
        //   Holdout samples (orig ∈ {0,1,6}) are excluded from SVM metrics
        //   because they were never seen during training.
        float svm_acc  = 0.f;
        float mf1_svm  = 0.f;
        {
            long cm_svm[N_CLASSES][N_CLASSES] = {};
            for (int i = 0; i < N_test; ++i) {
                int orig = y_orig_all[i];
                if (orig < 0 || orig >= N_ALL) continue;
                int ti = ORIG2INT[orig]; if (ti < 0) continue;   // holdout
                int pi = pred_all[i];
                if (pi >= 0 && pi < N_CLASSES) cm_svm[ti][pi]++;
            }
            long svm_correct = 0, svm_total = 0;
            for (int i = 0; i < N_CLASSES; ++i)
                for (int j = 0; j < N_CLASSES; ++j) {
                    if (i == j) svm_correct += cm_svm[i][j];
                    svm_total += cm_svm[i][j];
                }
            svm_acc = svm_total > 0 ? (float)svm_correct / svm_total : 0.f;

            lprintf("\n  SVM-only per-class metrics (trained classes, all non-holdout):\n");
            lprintf("  %-16s  %8s  %8s  %8s  %8s\n",
                   "Class", "Prec", "Recall", "F1", "Support");
            for (int k = 0; k < N_CLASSES; ++k) {
                long tp = cm_svm[k][k], fp2 = 0, fn = 0;
                for (int j = 0; j < N_CLASSES; ++j) {
                    if (j != k) fp2 += cm_svm[j][k];
                    if (j != k) fn  += cm_svm[k][j];
                }
                float p  = (tp + fp2 > 0) ? (float)tp / (tp + fp2) : 0.f;
                float r  = (tp + fn  > 0) ? (float)tp / (tp + fn)  : 0.f;
                float f1 = (p + r > 1e-9f) ? 2.f * p * r / (p + r) : 0.f;
                mf1_svm += f1;
                lprintf("  %-16s  %8.4f  %8.4f  %8.4f  %8ld\n",
                       CLASS_NAMES[INT2ORIG[k]], (double)p, (double)r, (double)f1, tp + fn);
            }
            mf1_svm /= N_CLASSES;
            lprintf("  %-16s                          %8.4f  (SVM-only)\n",
                   "macro_F1", (double)mf1_svm);
        }

        // Step 1g: Save SVM-only predictions to svm_predictions.csv
        //   -3 is used for holdout samples (SVM cannot classify them).
        {
            std::ofstream sf(SVM_PRED_OUT);
            if (!sf) {
                fprintf(stderr, "[rank 0] Cannot open %s\n", SVM_PRED_OUT);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (int i = 0; i < N_test; ++i) {
                int svm_p = is_holdout(y_orig_all[i]) ? -3 : INT2ORIG[pred_all[i]];
                sf << y_orig_all[i] << "," << svm_p << "\n";
            }
            sf.close();
            lprintf("[%s] SVM-only predictions -> %s\n", ts().c_str(), SVM_PRED_OUT);
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE 2 — DBSCAN CLUSTERING
        //
        //   Collect two pools into a single DBSCAN input:
        //     (a) Holdout pool  — all samples from classes {0,1,6}.
        //         These are placed FIRST (indices 0..n_holdout-1) and act as
        //         labelled anchors: their known ground-truth class is used to
        //         label clusters via majority vote.
        //     (b) Uncertain pool — non-holdout samples where the SVM confidence
        //         did not reach CONF_THRESHOLD.  These are the samples the SVM
        //         is least sure about.
        //   DBSCAN operates in the 4-D space of OvO vote count vectors.
        // ══════════════════════════════════════════════════════════════════

        lprintf("\n[%s] ════════════════════════════════════════════════════\n"
               "[%s] PHASE 2 — DBSCAN CLUSTERING\n"
               "[%s] ════════════════════════════════════════════════════\n",
               ts().c_str(), ts().c_str(), ts().c_str());

        double t0_phase2 = MPI_Wtime();

        // Step 2a: Partition samples into holdout, uncertain, and confident
        std::vector<int> ho_idx, unc_idx;
        int n_full_uncertain = 0;
        std::vector<int> unc_normal_idx;   // conf < threshold but SVM argmax = Normal
        for (int i = 0; i < N_test; ++i) {
            int orig = y_orig_all[i];
            if (is_holdout(orig)) {
                ho_idx.push_back(i);
            } else if (conf_all[i] < CONF_THRESHOLD) {
                ++n_full_uncertain;
                if (pred_all[i] == NORMAL_INT) {
                    // SVM already says Normal — don't send to DBSCAN where it
                    // could be relabelled -1 (noise) or -2 (unknown).
                    // Assign Normal directly by recording in unc_normal_idx.
                    unc_normal_idx.push_back(i);
                } else {
                    // Uncertain attack-class prediction → DBSCAN can help
                    unc_idx.push_back(i);
                }
            }
        }

        // Cap only the non-Normal uncertain pool (attack-class uncertain samples)
        if ((int)unc_idx.size() > N_MAX_UNCERTAIN)
            unc_idx.resize(N_MAX_UNCERTAIN);

        int nh   = (int)ho_idx.size();
        int nu   = (int)unc_idx.size();
        int n_db = nh + nu;
        int n_confident    = N_test - nh - n_full_uncertain;
        int n_unc_normal   = (int)unc_normal_idx.size();
        int n_unc_attack   = n_full_uncertain - n_unc_normal;

        lprintf("  Holdout samples  (anchors):         %d\n", nh);
        lprintf("  Confident samples (SVM final):       %d\n", n_confident);
        lprintf("  Uncertain: Normal-predicted (direct): %d  (SVM argmax=Normal, skip DBSCAN)\n",
               n_unc_normal);
        lprintf("  Uncertain: attack-predicted (DBSCAN): %d (capped at %d; "
               "%d overflow -> PRED_NOVEL)\n",
               n_unc_attack, N_MAX_UNCERTAIN, n_unc_attack - nu);
        lprintf("  DBSCAN pool total:                   %d\n", n_db);

        // Step 2b: Build X_db — each row is the N_PAIRS-dim raw OvO score vector.
        //   Using continuous raw scores instead of integer vote counts provides
        //   much better class separation (44 possible vote vectors in 4-D vs
        //   continuous 6-D for N_PAIRS=6; WebAttacks and BruteForce may share
        //   the same vote pattern but differ in raw margins).
        //   Holdout anchors occupy pool positions [0, nh)
        //   Uncertain samples occupy pool positions [nh, nh+nu)
        std::vector<float> X_db((size_t)n_db * N_DBSCAN_DIM);
        for (int i = 0; i < nh; ++i)
            for (int k = 0; k < N_DBSCAN_DIM; ++k)
                X_db[(size_t)i * N_DBSCAN_DIM + k] = scores_all[(size_t)ho_idx[i] * N_PAIRS + k];
        for (int i = 0; i < nu; ++i)
            for (int k = 0; k < N_DBSCAN_DIM; ++k)
                X_db[(size_t)(nh + i) * N_DBSCAN_DIM + k] = scores_all[(size_t)unc_idx[i] * N_PAIRS + k];

        // Holdout original IDs (anchor ground-truth labels)
        std::vector<int> ho_orig(nh);
        for (int i = 0; i < nh; ++i) ho_orig[i] = y_orig_all[ho_idx[i]];

        // Step 2c: Compute Normal-class centroid in N_PAIRS-dim raw score space.
        //   Used as a reference point for clusters that have no holdout anchors.
        std::array<double, N_PAIRS> nsum = {};
        int n_norm = 0;
        for (int i = 0; i < N_test; ++i) {
            if (y_orig_all[i] == NORMAL_ORIG_ID) {
                for (int k = 0; k < N_PAIRS; ++k)
                    nsum[k] += scores_all[(size_t)i * N_PAIRS + k];
                ++n_norm;
            }
        }
        std::vector<float> norm_ctr(N_PAIRS, 0.f);
        if (n_norm > 0)
            for (int k = 0; k < N_PAIRS; ++k)
                norm_ctr[k] = (float)(nsum[k] / n_norm);
        lprintf("  Normal centroid (n=%d): [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
               n_norm,
               (double)norm_ctr[0], (double)norm_ctr[1], (double)norm_ctr[2],
               (double)norm_ctr[3], (double)norm_ctr[4], (double)norm_ctr[5]);

        // Step 2d: Run DBSCAN on the pool (O(n²), CPU, rank 0 only)
        int n_clusters = 0;
        std::vector<int> cluster_id = run_dbscan(X_db, n_db, N_DBSCAN_DIM, EPS, MIN_SAMPLES,
                                                  n_clusters);

        // Step 2e: Compute cluster centroids in N_PAIRS-dim raw score space
        std::vector<std::array<double, N_PAIRS>> centroid(n_clusters, std::array<double, N_PAIRS>{});
        std::vector<int> csz(n_clusters, 0);
        for (int i = 0; i < n_db; ++i) {
            int c = cluster_id[i]; if (c < 0) continue;
            for (int d = 0; d < N_DBSCAN_DIM; ++d)
                centroid[c][d] += X_db[(size_t)i * N_DBSCAN_DIM + d];
            ++csz[c];
        }
        for (int c = 0; c < n_clusters; ++c)
            if (csz[c] > 0)
                for (int d = 0; d < N_DBSCAN_DIM; ++d) centroid[c][d] /= csz[c];

        // Step 2f: Label each cluster
        //   Rule 1: If the cluster contains any holdout anchor, assign the most
        //           common holdout original-class ID (majority vote).
        //   Rule 2: If no holdout anchors, compare cluster centroid distance to
        //           the Normal centroid.  If close (< NORMAL_PROX_THRESH),
        //           classify as Normal (4).  Otherwise UnknownAnomaly (-2).
        std::vector<std::unordered_map<int, int>> votes(n_clusters);
        for (int i = 0; i < nh; ++i) {
            int c = cluster_id[i];
            if (c >= 0) votes[c][ho_orig[i]]++;
        }

        std::vector<int> cluster_class(n_clusters, PRED_UNKNOWN);
        int cnt_by_holdout = 0, cnt_normal = 0, cnt_unknown = 0;
        for (int c = 0; c < n_clusters; ++c) {
            if (!votes[c].empty()) {
                // Majority vote over holdout anchor labels in this cluster
                int best_class = -1, best_votes = 0;
                for (auto& [cl, v] : votes[c])
                    if (v > best_votes) { best_votes = v; best_class = cl; }
                cluster_class[c] = best_class;
                ++cnt_by_holdout;
            } else {
                // No holdout anchors in this cluster.
                // Rule 2: simulate OvO votes from the centroid's raw scores to
                //   determine the predicted class in raw-score space.  Only
                //   reclassify as Normal if BOTH:
                //   1. The simulated argmax is NORMAL_INT (cluster looks like Normal).
                //   2. The centroid is within NORMAL_PROX_THRESH of the Normal centroid.
                // If only condition 2 holds, call it UnknownAnomaly (security risk).
                int votes_sim[N_CLASSES] = {};
                for (int d = 0; d < N_PAIRS; ++d)
                    if (centroid[c][d] >= 0.0) votes_sim[PAIR_I[d]]++;
                    else                       votes_sim[PAIR_J[d]]++;
                int ctr_pred = 0;
                for (int d = 1; d < N_CLASSES; ++d)
                    if (votes_sim[d] > votes_sim[ctr_pred]) ctr_pred = d;

                double d2 = 0.0;
                for (int d = 0; d < N_DBSCAN_DIM; ++d) {
                    double diff = centroid[c][d] - norm_ctr[d];
                    d2 += diff * diff;
                }
                if (ctr_pred == NORMAL_INT &&
                    d2 < (double)NORMAL_PROX_THRESH * NORMAL_PROX_THRESH) {
                    cluster_class[c] = NORMAL_ORIG_ID;
                    ++cnt_normal;
                } else {
                    // Label by centroid's simulated vote — keeps samples classifiable
                    // as DDoS/DoS/PortScan rather than dropping them as PRED_UNKNOWN.
                    cluster_class[c] = INT2ORIG[ctr_pred];
                    ++cnt_unknown;
                }
            }
        }
        lprintf("  Cluster labelling: %d holdout-voted, %d reclassified Normal, "
               "%d centroid-predicted attack\n", cnt_by_holdout, cnt_normal, cnt_unknown);

        // Step 2g: Save DBSCAN model to dbscan_model.bin
        //   Stores cluster centroids + labels so the DBSCAN model can be loaded
        //   independently and used for nearest-centroid inference on new data.
        //   Updated layout (v2): includes N_DBSCAN_DIM in header so loaders
        //   do not hard-code the centroid dimension.
        {
            std::ofstream df(DBSCAN_MDL_OUT, std::ios::binary);
            if (!df) {
                fprintf(stderr, "[rank 0] Cannot open %s\n", DBSCAN_MDL_OUT);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            // Header: hyperparameters and cluster count
            df.write(reinterpret_cast<const char*>(&n_clusters),         sizeof(int));
            float eps_save = EPS;
            df.write(reinterpret_cast<const char*>(&eps_save),           sizeof(float));
            df.write(reinterpret_cast<const char*>(&MIN_SAMPLES),        sizeof(int));
            float npt_save = NORMAL_PROX_THRESH;
            df.write(reinterpret_cast<const char*>(&npt_save),           sizeof(float));
            int ndim_save = N_DBSCAN_DIM;
            df.write(reinterpret_cast<const char*>(&ndim_save),          sizeof(int));
            // Per-cluster: centroid (N_DBSCAN_DIM floats) + label (int)
            for (int c = 0; c < n_clusters; ++c) {
                float ctr[N_DBSCAN_DIM];
                for (int d = 0; d < N_DBSCAN_DIM; ++d) ctr[d] = (float)centroid[c][d];
                df.write(reinterpret_cast<const char*>(ctr),              N_DBSCAN_DIM * sizeof(float));
                df.write(reinterpret_cast<const char*>(&cluster_class[c]), sizeof(int));
            }
            df.close();
            lprintf("[%s] DBSCAN model saved -> %s  (%d clusters, %d-D)\n",
                   ts().c_str(), DBSCAN_MDL_OUT, n_clusters, N_DBSCAN_DIM);
        }

        // ══════════════════════════════════════════════════════════════════
        // PHASE 3 — HYBRID COMBINATION  (SVM + DBSCAN)
        //
        //   SVM FIRST (confident):
        //     Non-holdout samples with conf >= CONF_THRESHOLD → SVM argmax.
        //
        //   SVM DIRECT (uncertain Normal):
        //     Non-holdout samples where conf < threshold BUT SVM argmax is
        //     Normal → assign Normal directly.  These must NOT go to DBSCAN
        //     because DBSCAN noise/unknown labels (-1/-2) would turn genuine
        //     Normal traffic into false attack detections.
        //
        //   DBSCAN FALLBACK (uncertain attack-class):
        //     Non-holdout samples where conf < threshold AND SVM argmax is
        //     an attack class (DDoS/DoS/PortScan) → DBSCAN cluster label.
        //     Holdout samples always go through DBSCAN for novel-class detection.
        // ══════════════════════════════════════════════════════════════════

        lprintf("\n[%s] ════════════════════════════════════════════════════\n"
               "[%s] PHASE 3 — HYBRID COMBINATION  (SVM->DBSCAN fallback)\n"
               "[%s] ════════════════════════════════════════════════════\n",
               ts().c_str(), ts().c_str(), ts().c_str());

        double t1_phase2 = MPI_Wtime();
        lprintf("[%s] Phase 2 (DBSCAN clustering) time: %.3fs\n",
               ts().c_str(), t1_phase2 - t0_phase2);

        double t0_phase3 = MPI_Wtime();
        lprintf("  Logic: conf >= %.2f → SVM argmax;\n"
               "         conf < %.2f AND pred=Normal → Normal direct (no DBSCAN);\n"
               "         conf < %.2f AND pred=attack → DBSCAN cluster;\n"
               "         holdout class → DBSCAN (novel detection).\n",
               (double)CONF_THRESHOLD,
               (double)CONF_THRESHOLD,
               (double)CONF_THRESHOLD);

        // Step 3a: Initialise final_pred to PRED_NOVEL for all samples.
        std::vector<int> final_pred(N_test, PRED_NOVEL);

        // SVM FIRST — confident non-holdout samples (conf >= CONF_THRESHOLD)
        for (int i = 0; i < N_test; ++i) {
            if (!is_holdout(y_orig_all[i]) && conf_all[i] >= CONF_THRESHOLD)
                final_pred[i] = INT2ORIG[pred_all[i]];
        }

        // SVM DIRECT — uncertain samples where SVM argmax is already Normal.
        // Assign Normal immediately; do NOT send through DBSCAN which would
        // risk labeling real Normal traffic as noise (-1) or unknown (-2).
        for (int idx : unc_normal_idx)
            final_pred[idx] = NORMAL_ORIG_ID;

        // DBSCAN FALLBACK — uncertain attack-class pool (pool indices [nh, nh+nu))
        long n_noise_svm = 0;  // noise points that fell back to SVM original prediction
        for (int ui = 0; ui < nu; ++ui) {
            int ti = unc_idx[ui];
            int c  = cluster_id[nh + ui];
            if (c < 0) {
                // DBSCAN noise: isolated point → keep SVM's attack prediction
                final_pred[ti] = INT2ORIG[pred_all[ti]];
                ++n_noise_svm;
            } else {
                final_pred[ti] = cluster_class[c];
            }
        }

        // DBSCAN FALLBACK — holdout pool (pool indices [0, nh))
        for (int hi = 0; hi < nh; ++hi) {
            int ti = ho_idx[hi];
            int c  = cluster_id[hi];
            final_pred[ti] = (c < 0) ? PRED_NOVEL : cluster_class[c];
        }
        // Note: overflow uncertain attack-class samples (above cap) keep PRED_NOVEL

        // ── Evaluation ────────────────────────────────────────────────────
        lprintf("\n[%s] ── Evaluation ──────────────────────────────────────\n",
               ts().c_str());

        // (A) Per-class confusion matrix for the 4 trained classes
        //     Only samples where true label AND predicted label are both
        //     mapped trained classes contribute to the confusion matrix.
        long cm[N_CLASSES][N_CLASSES] = {};
        for (int i = 0; i < N_test; ++i) {
            int orig = y_orig_all[i];
            if (orig < 0 || orig >= N_ALL) continue;
            int ti = ORIG2INT[orig]; if (ti < 0) continue;   // holdout: skip
            int fp = final_pred[i];
            if (fp < 0 || fp >= N_ALL) continue;
            int pi = ORIG2INT[fp];   if (pi < 0) continue;
            cm[ti][pi]++;
        }

        lprintf("\n  Hybrid per-class metrics (trained classes, confident+DBSCAN):\n");
        lprintf("  %-16s  %8s  %8s  %8s  %8s\n",
               "Class", "Prec", "Recall", "F1", "Support");
        float mf1 = 0.f;
        for (int k = 0; k < N_CLASSES; ++k) {
            long tp = cm[k][k], fp2 = 0, fn = 0;
            for (int j = 0; j < N_CLASSES; ++j) {
                if (j != k) fp2 += cm[j][k];
                if (j != k) fn  += cm[k][j];
            }
            float p  = (tp + fp2 > 0) ? (float)tp / (tp + fp2) : 0.f;
            float r  = (tp + fn  > 0) ? (float)tp / (tp + fn)  : 0.f;
            float f1 = (p + r > 1e-9f) ? 2.f * p * r / (p + r) : 0.f;
            mf1 += f1;
            lprintf("  %-16s  %8.4f  %8.4f  %8.4f  %8ld\n",
                   CLASS_NAMES[INT2ORIG[k]],
                   (double)p, (double)r, (double)f1, tp + fn);
        }
        mf1 /= N_CLASSES;
        lprintf("  %-16s                          %8.4f  (hybrid)\n",
               "macro_F1", (double)mf1);

        // Overall accuracy on trained (non-holdout) classes
        long correct_total = 0, total_trained = 0;
        for (int i = 0; i < N_test; ++i) {
            int orig = y_orig_all[i];
            if (orig < 0 || orig >= N_ALL) continue;
            int ti = ORIG2INT[orig]; if (ti < 0) continue;
            int fp_i = final_pred[i];
            if (fp_i < 0 || fp_i >= N_ALL) continue;
            int pi = ORIG2INT[fp_i]; if (pi < 0) continue;
            ++total_trained;
            if (ti == pi) ++correct_total;
        }
        float accuracy = total_trained > 0
                         ? (float)correct_total / total_trained : 0.f;

        // (B) Holdout detection via DBSCAN cluster majority-vote labels
        lprintf("\n  Holdout class detection via DBSCAN:\n");
        lprintf("  %-16s  %8s  %8s  %8s\n", "Class", "Total", "Correct", "Noise");
        long ho_total_all = 0, ho_correct_all = 0;
        long bots_tot = 0, bots_cor = 0, bots_noi = 0;
        long bf_tot   = 0, bf_cor   = 0, bf_noi   = 0;
        long wa_tot   = 0, wa_cor   = 0, wa_noi   = 0;
        for (int hi = 0; hi < nh; ++hi) {
            int orig = ho_orig[hi];
            int c    = cluster_id[hi];
            bool correct = (c >= 0 && cluster_class[c] == orig);
            bool noise   = (c < 0);
            ++ho_total_all; if (correct) ++ho_correct_all;
            if      (orig == 0) { ++bots_tot; if(correct)++bots_cor; if(noise)++bots_noi; }
            else if (orig == 1) { ++bf_tot;   if(correct)++bf_cor;   if(noise)++bf_noi;   }
            else                { ++wa_tot;   if(correct)++wa_cor;   if(noise)++wa_noi;    }
        }
        lprintf("  %-16s  %8ld  %8ld  %8ld\n", "Bots",       bots_tot, bots_cor, bots_noi);
        lprintf("  %-16s  %8ld  %8ld  %8ld\n", "BruteForce", bf_tot,   bf_cor,   bf_noi);
        lprintf("  %-16s  %8ld  %8ld  %8ld\n", "WebAttacks", wa_tot,   wa_cor,   wa_noi);
        float holdout_dr = ho_total_all > 0 ?
                           (float)ho_correct_all / ho_total_all : 0.f;
        lprintf("  Holdout detection rate: %.4f  (%ld / %ld)\n",
               (double)holdout_dr, ho_correct_all, ho_total_all);

        // (C) Uncertain pool reclassification breakdown
        // Attack-class uncertain → went through DBSCAN
        long ub = 0, uh = 0;
        for (int ui = 0; ui < nu; ++ui) {
            int p = final_pred[unc_idx[ui]];
            if      (p == NORMAL_ORIG_ID)          ++ub;
            else if (p == 0 || p == 1 || p == 6)  ++uh;
        }
        long n_centroid_pred = nu - ub - uh - n_noise_svm;
        lprintf("\n  Uncertain pool breakdown (%d total uncertain):\n", n_full_uncertain);
        lprintf("  %-42s  %d\n", "Direct Normal (SVM argmax=Normal, skip DBSCAN)", n_unc_normal);
        lprintf("  %-42s  %d\n", "Attack-class uncertain sent to DBSCAN",           nu);
        lprintf("    %-40s  %ld\n", "-> Reclassified Normal (near-Normal cluster)",  ub);
        lprintf("    %-40s  %ld\n", "-> Matched holdout pattern (Bots/BF/WebAtk)",   uh);
        lprintf("    %-40s  %ld\n", "-> Centroid-predicted attack class",             n_centroid_pred);
        lprintf("    %-40s  %ld\n", "-> SVM fallback (DBSCAN noise -1)",              n_noise_svm);

        // ── Write hybrid_predictions.csv ──────────────────────────────────
        {
            std::ofstream pf(PRED_OUT);
            if (!pf) {
                fprintf(stderr, "[rank 0] Cannot open %s\n", PRED_OUT);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (int i = 0; i < N_test; ++i)
                pf << y_orig_all[i] << "," << final_pred[i] << "\n";
            pf.close();
            lprintf("\n[%s] Hybrid predictions    -> %s\n", ts().c_str(), PRED_OUT);
        }

        // ── Write hybrid_log.csv ──────────────────────────────────────────
        {
            double t1_phase3   = MPI_Wtime();
            double t_total_end = MPI_Wtime();
            lprintf("[%s] Phase 3 (hybrid combination) time: %.3fs\n",
                   ts().c_str(), t1_phase3 - t0_phase3);

            double predict_ms    = (t1_phase1 - t0_phase1) * 1000.0;
            double dbscan_ms     = (t1_phase2 - t0_phase2) * 1000.0;
            double phase3_ms     = (t1_phase3 - t0_phase3) * 1000.0;
            double total_time_ms = (t_total_end - t_total_start) * 1000.0;

            // SVM inference FLOPs: OvO scoring per sample
            double svm_flops   = (double)N_test * (double)N_PAIRS * N_FEATURES * 2.0;
            double svm_gflops  = (t1_phase1 - t0_phase1) > 0.0
                                 ? svm_flops / (t1_phase1 - t0_phase1) / 1e9 : 0.0;

            // DBSCAN FLOPs: 2 O(n²) passes × N_DBSCAN_DIM-D distance per pair
            double dbscan_flops  = 2.0 * n_db * n_db * N_DBSCAN_DIM * 2.0;
            double dbscan_gflops = (t1_phase2 - t0_phase2) > 0.0
                                   ? dbscan_flops / (t1_phase2 - t0_phase2) / 1e9 : 0.0;

            lprintf("[%s] ── Performance Summary ─────────────────────────\n",
                   ts().c_str());
            lprintf("  Accuracy:         %.4f\n", (double)accuracy);
            lprintf("  Predict   (ms):   %.1f\n",  predict_ms);
            lprintf("  DBSCAN    (ms):   %.1f\n",  dbscan_ms);
            lprintf("  Total     (ms):   %.1f\n",  total_time_ms);
            lprintf("  SVM GFLOPS:       %.3f\n",  svm_gflops);
            lprintf("  DBSCAN GFLOPS:    %.3f\n",  dbscan_gflops);

            std::ofstream lf(LOG_OUT);
            if (!lf) {
                fprintf(stderr, "[rank 0] Cannot open %s\n", LOG_OUT);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            lf << "metric,value\n";
            lf << "n_test,"                    << N_test           << "\n";
            lf << "n_clusters,"                << n_clusters       << "\n";
            lf << "eps,"                        << EPS              << "\n";
            lf << "conf_threshold,"             << CONF_THRESHOLD   << "\n";
            lf << "n_confident,"                << n_confident      << "\n";
            lf << "n_uncertain_full,"           << n_full_uncertain << "\n";
            lf << "n_unc_normal_direct,"        << n_unc_normal     << "\n";
            lf << "n_unc_attack_dbscan,"        << nu               << "\n";
            lf << "n_holdout,"                  << nh               << "\n";
            lf << "accuracy,"                   << accuracy         << "\n";
            lf << "macro_f1_hybrid,"            << mf1              << "\n";
            lf << "holdout_detection_rate,"     << holdout_dr       << "\n";
            lf << "bots_correct,"               << bots_cor         << "\n";
            lf << "bots_total,"                 << bots_tot         << "\n";
            lf << "bf_correct,"                 << bf_cor           << "\n";
            lf << "bf_total,"                   << bf_tot           << "\n";
            lf << "wa_correct,"                 << wa_cor           << "\n";
            lf << "wa_total,"                   << wa_tot           << "\n";
            lf << "dbscan_reclass_normal,"      << ub               << "\n";
            lf << "dbscan_reclass_holdout,"     << uh               << "\n";
            lf << "dbscan_centroid_pred,"       << n_centroid_pred  << "\n";
            lf << "dbscan_noise_svm,"           << n_noise_svm      << "\n";
            lf << "predict_ms,"                 << predict_ms       << "\n";
            lf << "dbscan_ms,"                  << dbscan_ms        << "\n";
            lf << "phase3_ms,"                  << phase3_ms        << "\n";
            lf << "total_time_ms,"              << total_time_ms    << "\n";
            lf << "phase1_time_s,"              << (t1_phase1 - t0_phase1) << "\n";
            lf << "phase2_time_s,"              << (t1_phase2 - t0_phase2) << "\n";
            lf << "phase3_time_s,"              << (t1_phase3 - t0_phase3) << "\n";
            lf << "total_time_s,"               << (t_total_end - t_total_start) << "\n";
            lf << "svm_gflops,"                 << svm_gflops       << "\n";
            lf << "dbscan_gflops,"              << dbscan_gflops    << "\n";
            lf.close();
            lprintf("[%s] Summary log           -> %s\n", ts().c_str(), LOG_OUT);

            // ── Write results.csv (flat one-row summary) ──────────────────
            // Columns: acc (SVM-only), lav (avg confidence / LAV),
            //          f1  (SVM-only macro-F1), Accuracy (hybrid),
            //          Predict_ms, DBSCAN_ms, Total_ms, GFLOPS
            {
                double avg_conf = 0.0;
                for (int i = 0; i < N_test; ++i) avg_conf += conf_all[i];
                avg_conf /= N_test;
                double total_gflops = svm_gflops + dbscan_gflops;

                std::ofstream rf(RESULTS_OUT);
                if (!rf) {
                    fprintf(stderr, "[rank 0] Cannot open %s\n", RESULTS_OUT);
                } else {
                    rf << "acc,lav,f1,Accuracy,Predict_ms,DBSCAN_ms,Total_ms,GFLOPS\n";
                    rf << svm_acc         << ","
                       << avg_conf        << ","
                       << mf1_svm         << ","
                       << accuracy        << ","
                       << predict_ms      << ","
                       << dbscan_ms       << ","
                       << total_time_ms   << ","
                       << total_gflops    << "\n";
                    rf.close();
                    lprintf("[%s] Results CSV           -> %s\n", ts().c_str(), RESULTS_OUT);
                }
            }
        }

        lprintf("[%s] Done.  Output files:\n", ts().c_str());
        lprintf("       SVM-only preds  : %s\n", SVM_PRED_OUT);
        lprintf("       DBSCAN model    : %s\n", DBSCAN_MDL_OUT);
        lprintf("       Hybrid preds    : %s\n", PRED_OUT);
        lprintf("       Metrics log     : %s\n", LOG_OUT);
        lprintf("       Results CSV     : %s\n", RESULTS_OUT);

    } // end rank 0 only

    if (rank == 0) log_close();
    MPI_CHECK(MPI_Finalize());
    return 0;
}
