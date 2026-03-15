/*
 * omp_infer.cpp  —  OpenMP Hybrid SVM (One-vs-One) + DBSCAN inference pipeline
 *
 * Purpose:
 *   Load the SVM model produced by omp_train (OvO format: K=N_PAIRS=6),
 *   run parallel inference on CICIDS2017 test data using OvO voting, then
 *   apply CPU DBSCAN re-classification for low-confidence and holdout samples.
 *
 * Two models used by this pipeline:
 *   model.bin        — SVM weights (from omp_train, OvO: K=6 pair classifiers)
 *   dbscan_model.bin — DBSCAN cluster centroids + labels (produced here)
 *
 * Algorithm summary:
 *
 *   ── PHASE 1  SVM INFERENCE ──────────────────────────────────────────
 *   1. Main thread loads model.bin, test_data.csv, test_labels.csv.
 *   2. #pragma omp parallel for schedule(static): distribute all N_test
 *      rows across threads.  Each thread runs OvO voting:
 *        for each pair p in {0..5}: score = W[p]·x + b[p]
 *          if score >= 0: votes[PAIR_I[p]]++  else  votes[PAIR_J[p]]++
 *        pred[i]  = argmax votes[]
 *        conf[i]  = votes[pred[i]] / (N_CLASSES-1)   range [0,1]
 *   3. Main thread evaluates SVM-only metrics; writes svm_predictions.csv.
 *
 *   ── PHASE 2  DBSCAN CLUSTERING ─────────────────────────────────────
 *   DBSCAN Pass 1 (neighbour counting) and Pass 2 (adjacency fill) are
 *   parallelised with #pragma omp parallel for schedule(static).
 *   BFS cluster expansion remains serial (shared queue, negligible time).
 *   4. Pool construction (same logic as mpi_infer rank-0):
 *        holdout (orig ∈ {0,1,6})        → DBSCAN pool (anchor slots 0..nh-1)
 *        conf < CONF_THRESHOLD AND
 *          pred != NORMAL_INT            → uncertain attack pool (capped at
 *                                          N_MAX_UNCERTAIN; slots nh..nh+nu-1)
 *        conf < CONF_THRESHOLD AND
 *          pred == NORMAL_INT            → direct Normal (skip DBSCAN)
 *   5. Run O(n²) CPU DBSCAN in 4-D vote-count space on the pool.
 *      Vote dims are integers in {0..3} summing to N_PAIRS=6.  Min nonzero
 *      inter-point dist² = 2.0 (differ by ±1 in two dims); EPS=1.5 captures
 *      nearest neighbours.  NORMAL_PROX_THRESH=1.5 restricts the Normal label
 *      to cluster centroids genuinely close to the Normal class centroid.
 *   6. Cluster labelling:
 *        has holdout anchors  → majority-vote original class ID
 *        no anchor, centroid argmax=Normal AND dist<NORMAL_PROX_THRESH
 *                             → NORMAL_ORIG_ID (4)
 *        no anchor, otherwise → PRED_UNKNOWN (-2)
 *        noise point          → PRED_NOVEL (-1)
 *   7. Save dbscan_model.bin (same binary format as mpi_infer).
 *
 *   ── PHASE 3  HYBRID COMBINATION ────────────────────────────────────
 *   8. #pragma omp parallel for: assign final_pred[i] in parallel:
 *        conf >= CONF_THRESHOLD AND not holdout → INT2ORIG[pred[i]]
 *        uncertain Normal (direct)              → NORMAL_ORIG_ID
 *        uncertain attack / holdout             → cluster_class[cluster_id]
 *   9. Serial evaluation tables (same format as mpi_infer).
 *
 * Build:
 *   g++ -O3 -std=c++17 -fopenmp -o omp_infer omp_infer.cpp -lm
 *
 * Run:
 *   OMP_NUM_THREADS=4 ./omp_infer
 *
 * Output files (same names as mpi_infer for direct comparison):
 *   svm_predictions.csv    — original_label,svm_predicted_label
 *   dbscan_model.bin       — DBSCAN cluster model: centroids + labels
 *   hybrid_predictions.csv — original_label,final_predicted_label
 *   hybrid_log.csv         — summary metrics
 *
 * dbscan_model.bin layout (byte-for-byte identical to mpi_infer output):
 *   int32   n_clusters
 *   float32 eps
 *   int32   min_samples
 *   float32 normal_prox_thresh
 *   For each cluster c in [0, n_clusters):
 *     float32[4]  centroid  (mean 4-D vote-count vector)
 *     int32       label     (original class ID; PRED_UNKNOWN=-2)
 */

#include <omp.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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
// Configuration — keep identical to mpi_infer.cpp
// ---------------------------------------------------------------------------
static constexpr int   N_FEATURES         = 52;
static constexpr int   N_CLASSES          = 4;    // trained class count
static constexpr int   N_PAIRS            = N_CLASSES * (N_CLASSES - 1) / 2; // 6
static constexpr int   PAIR_I[N_PAIRS]    = {0, 0, 0, 1, 1, 2};
static constexpr int   PAIR_J[N_PAIRS]    = {1, 2, 3, 2, 3, 3};
static constexpr int   N_ALL              = 7;    // original class count

// CONF_THRESHOLD: conf_all = votes[best_k]/(N_CLASSES-1) in [0,1].
// In OvO with N_CLASSES=4, each class competes in exactly 3 pairs, so
// the minimum possible winning conf = 2/3 ≈ 0.667 (wins exactly 2 of 3).
// Setting to 1.0 routes all 2-of-3 winners (conf≈0.667) to DBSCAN for
// rechecking; only 3-of-3 winners (conf=1.0) bypass DBSCAN as confident.
static constexpr float CONF_THRESHOLD     = 0.67;
static constexpr int   N_MAX_UNCERTAIN    = 110000;
// EPS for DBSCAN in 4-D OvO vote-count space.  Vote dims are integers with
// each class participating in N_CLASSES-1=3 pairs (so votes[k] ∈ {0..3})
// and total votes summing to N_PAIRS=6.  Two distinct valid vote vectors
// must differ in at least 2 dimensions (sum is fixed), so the minimum
// non-zero squared distance is 2.0 (±1 in two dims).
// EPS=1.5 → eps²=2.25 captures nearest neighbours that differ by 1 step.
static constexpr float EPS               = 1.1f;
static constexpr int   MIN_SAMPLES        = 5;
static constexpr float NORMAL_PROX_THRESH = 1.5f;

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
static const char* MODEL_IN       = "../model/model.bin";
static const char* TEST_DATA_CSV  = "../data/processed/test_data.csv";
static const char* TEST_LABEL_CSV = "../data/processed/test_labels.csv";

// Output paths — one per pipeline stage
static const char* SVM_PRED_OUT   = "../log/svm_predictions.csv";
static const char* DBSCAN_MDL_OUT = "../log/dbscan_model.bin";
static const char* PRED_OUT       = "../log/hybrid_predictions.csv";
static const char* LOG_OUT        = "../log/hybrid_log.csv";
static const char* RUN_LOG        = "../log/omp_infer_run.log";

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

static inline bool is_holdout(int orig) { return orig == 0 || orig == 1 || orig == 6; }

// Squared Euclidean distance between two 4-D vectors (vote-count space)
static inline float dist4_sq(const float* a, const float* b) {
    float d0 = a[0]-b[0], d1 = a[1]-b[1], d2 = a[2]-b[2], d3 = a[3]-b[3];
    return d0*d0 + d1*d1 + d2*d2 + d3*d3;
}

// ---------------------------------------------------------------------------
// CSV loaders
// ---------------------------------------------------------------------------
static std::vector<float> load_csv_float(const char* path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
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
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    std::vector<int> data;
    data.reserve(800000);
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) data.push_back(std::stoi(line));
    return data;
}

// ---------------------------------------------------------------------------
// SVM model loader
// ---------------------------------------------------------------------------
struct SVMModel {
    int K, F;                        // K=N_PAIRS, F=N_FEATURES
    std::vector<float> W, b;         // SVM weights: W[K×F], b[K]
};

static SVMModel load_svm_model(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "Cannot open SVM model %s\n", path);
        exit(1);
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
// CPU O(n²) DBSCAN on n_db points in 4-D vote-count space
//   Returns cluster_id[n_db] (-1 = noise).  n_clusters returned by value.
//
//   Parallelisation strategy:
//     Pass 1 (neighbour counting) and Pass 2 (adjacency fill) are
//     embarrassingly parallel: row i is fully independent of row j.
//     Both loops are parallelised with omp parallel for schedule(static).
//     BFS cluster expansion remains serial — the shared queue has no safe
//     parallel form and the BFS itself is negligible compared to the O(n²)
//     distance passes.
// ---------------------------------------------------------------------------
static std::vector<int> run_dbscan(const std::vector<float>& X_db, int n_db,
                                    float eps, int min_samples,
                                    int& n_clusters_out)
{
    const float eps2 = eps * eps;

    // Pass 1: count neighbours (self included → count >= min_samples = core)
    //   Each row i is independent: schedule(static) gives equal-load chunks.
    lprintf("[%s]   DBSCAN pass 1: counting neighbours (n=%d, eps=%.4f) ...\n",
           ts().c_str(), n_db, (double)eps);
    std::vector<int> nbr_count(n_db, 0);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_db; ++i) {
        const float* xi = X_db.data() + (size_t)i * 4;
        int cnt = 0;
        for (int j = 0; j < n_db; ++j)
            if (dist4_sq(xi, X_db.data() + (size_t)j * 4) <= eps2) ++cnt;
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

    //   Pass 2 correctness: thread t writes exclusively to the range
    //   col_idx[row_ptr[i]..row_ptr[i+1]-1].  Those ranges are non-overlapping
    //   by construction of the prefix-sum row_ptr, so no locks needed.
    std::vector<int> col_idx(total_edges);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_db; ++i) {
        const float* xi = X_db.data() + (size_t)i * 4;
        int pos = row_ptr[i];           // exclusive start of row i in col_idx
        for (int j = 0; j < n_db; ++j) {
            if (dist4_sq(xi, X_db.data() + (size_t)j * 4) <= eps2)
                col_idx[pos++] = j;
        }
    }

    // BFS cluster expansion
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
int main(int /*argc*/, char** /*argv*/) {

    int N_THREADS = omp_get_max_threads();
    omp_set_num_threads(N_THREADS);

    log_open(RUN_LOG);
    lprintf("OpenMP threads: %d\n", N_THREADS);
    double t_total_start = omp_get_wtime();

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 1 — SVM INFERENCE  (parallel across all threads)
    //
    //   OvO voting: for each of the N_PAIRS=6 binary classifiers, compute
    //   score = W[p]·x + b[p].  The class with the most votes is the
    //   prediction; confidence = max_votes / (N_CLASSES-1), range [0,1].
    // ══════════════════════════════════════════════════════════════════════

    lprintf("[%s] ════════════════════════════════════════════════════\n"
            "[%s] PHASE 1 — SVM INFERENCE (One-vs-One, loading model.bin)\n"
            "[%s] ════════════════════════════════════════════════════\n",
            ts().c_str(), ts().c_str(), ts().c_str());

    // Step 1a: Load SVM model (OvO: K=N_PAIRS=6)
    lprintf("[%s] Loading SVM model: %s\n", ts().c_str(), MODEL_IN);
    SVMModel mdl = load_svm_model(MODEL_IN);
    assert(mdl.K == N_PAIRS && mdl.F == N_FEATURES);
    lprintf("[%s] Model loaded: K=%d F=%d\n",
            ts().c_str(), mdl.K, mdl.F);
    const std::vector<float>& W_svm = mdl.W;
    const std::vector<float>& b_svm = mdl.b;

    // Step 1b: Load test data
    lprintf("[%s] Loading test data: %s\n", ts().c_str(), TEST_DATA_CSV);
    std::vector<float> X_test = load_csv_float(TEST_DATA_CSV);
    std::vector<int>   y_test = load_csv_int(TEST_LABEL_CSV);
    int N_test = (int)y_test.size();
    assert(X_test.size() == (size_t)N_test * N_FEATURES);
    lprintf("[%s] N_test=%d\n", ts().c_str(), N_test);

    // Allocate output arrays (written in parallel)
    // scores_all stores vote counts (0.0-3.0 per class) in N_CLASSES=4 dims.
    // DBSCAN operates in this 4-D vote space.
    std::vector<float> scores_all((size_t)N_test * N_CLASSES, 0.f);
    std::vector<float> conf_all  (N_test, 0.f);
    std::vector<int>   pred_all  (N_test, 0);

    // Step 1c: Parallel OvO SVM inference
    //   For each sample: run N_PAIRS=6 binary classifiers, tally votes per
    //   class, take argmax.  conf = max_votes / (N_CLASSES-1) in [0,1].
    double t0_phase1 = omp_get_wtime();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_test; ++i) {
        const float* x = X_test.data() + (size_t)i * N_FEATURES;
        // OvO voting: tally integer votes AND accumulate |score| per class
        // for tie-breaking when two classes share the maximum vote count.
        int   votes[N_CLASSES]  = {};
        float margin[N_CLASSES] = {};   // sum of |score| for won pairs
        for (int p = 0; p < N_PAIRS; ++p) {
            float s = b_svm[p];
            for (int f = 0; f < N_FEATURES; ++f)
                s += W_svm[p * N_FEATURES + f] * x[f];
            if (s >= 0.f) { votes[PAIR_I[p]]++;  margin[PAIR_I[p]] += s;  }
            else           { votes[PAIR_J[p]]++;  margin[PAIR_J[p]] -= s;  }
        }
        // Argmax: primary key = vote count; secondary = accumulated |score|
        int best_k = 0;
        for (int k = 1; k < N_CLASSES; ++k) {
            if (votes[k] > votes[best_k] ||
                (votes[k] == votes[best_k] && margin[k] > margin[best_k]))
                best_k = k;
        }
        // Store integer vote counts as 4-D DBSCAN features
        for (int k = 0; k < N_CLASSES; ++k)
            scores_all[(size_t)i * N_CLASSES + k] = (float)votes[k];
        conf_all[i] = (float)votes[best_k] / (N_CLASSES - 1);  // max_votes / 3, range [0,1]
        pred_all[i] = best_k;
    }

    double t1_phase1 = omp_get_wtime();
    lprintf("[%s] Phase 1 (SVM OvO inference) time: %.3fs\n",
            ts().c_str(), t1_phase1 - t0_phase1);
    lprintf("[%s] ─── SVM inference complete for all %d test samples.\n",
            ts().c_str(), N_test);

    // Step 1d: SVM-only evaluation (mirror of mpi_infer rank-0 Step 1f)
    {
        long cm_svm[N_CLASSES][N_CLASSES] = {};
        for (int i = 0; i < N_test; ++i) {
            int orig = y_test[i];
            if (orig < 0 || orig >= N_ALL) continue;
            int ti = ORIG2INT[orig]; if (ti < 0) continue;  // holdout: skip
            int pi = pred_all[i];
            if (pi >= 0 && pi < N_CLASSES) cm_svm[ti][pi]++;
        }
        lprintf("\n  SVM-only per-class metrics (trained classes, all non-holdout):\n");
        lprintf("  %-16s  %8s  %8s  %8s  %8s\n",
                "Class", "Prec", "Recall", "F1", "Support");
        float mf1_svm = 0.f;
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
                    CLASS_NAMES[INT2ORIG[k]], (double)p, (double)r, (double)f1,
                    tp + fn);
        }
        mf1_svm /= N_CLASSES;
        lprintf("  %-16s                          %8.4f  (SVM-only OvO)\n",
                "macro_F1", (double)mf1_svm);
    }

    // Step 1e: Save svm_predictions.csv — same format as mpi_infer
    //   Holdout samples get -3 (SVM cannot classify them)
    {
        std::ofstream sf(SVM_PRED_OUT);
        if (!sf) {
            fprintf(stderr, "Cannot open %s\n", SVM_PRED_OUT);
            exit(1);
        }
        for (int i = 0; i < N_test; ++i) {
            int svm_p = is_holdout(y_test[i]) ? -3 : INT2ORIG[pred_all[i]];
            sf << y_test[i] << "," << svm_p << "\n";
        }
        sf.close();
        lprintf("[%s] SVM-only predictions -> %s\n", ts().c_str(), SVM_PRED_OUT);
    }

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 2 — DBSCAN CLUSTERING  (serial — same as mpi_infer rank-0)
    //
    //   OpenMP speedup is in Phases 1 & 3.  DBSCAN is O(n²) over a pool
    //   that is typically <200 k points; parallelising the BFS expansion
    //   introduces thread-safety complexity with no guaranteed gain here.
    //
    //   DBSCAN operates in 4-D vote-count space.  Each dimension holds the
    //   vote count for one class (integers 0–3, summing to N_PAIRS=6).
    // ══════════════════════════════════════════════════════════════════════

    lprintf("\n[%s] ════════════════════════════════════════════════════\n"
            "[%s] PHASE 2 — DBSCAN CLUSTERING (4-D vote space)\n"
            "[%s] ════════════════════════════════════════════════════\n",
            ts().c_str(), ts().c_str(), ts().c_str());

    double t0_phase2 = omp_get_wtime();

    // Step 2a: Partition samples — identical logic to mpi_infer rank-0
    std::vector<int> ho_idx, unc_idx, unc_normal_idx;
    int n_full_uncertain = 0;
    for (int i = 0; i < N_test; ++i) {
        int orig = y_test[i];
        if (is_holdout(orig)) {
            ho_idx.push_back(i);
        } else if (conf_all[i] < CONF_THRESHOLD) {
            ++n_full_uncertain;
            if (pred_all[i] == NORMAL_INT) {
                unc_normal_idx.push_back(i);
            } else {
                unc_idx.push_back(i);
            }
        }
    }

    // Cap only the non-Normal uncertain pool
    if ((int)unc_idx.size() > N_MAX_UNCERTAIN)
        unc_idx.resize(N_MAX_UNCERTAIN);

    const int nh   = (int)ho_idx.size();
    const int nu   = (int)unc_idx.size();
    const int n_db = nh + nu;
    const int n_confident  = N_test - nh - n_full_uncertain;
    const int n_unc_normal = (int)unc_normal_idx.size();
    const int n_unc_attack = n_full_uncertain - n_unc_normal;

    lprintf("  Holdout samples  (anchors):          %d\n", nh);
    lprintf("  Confident samples (SVM final):        %d\n", n_confident);
    lprintf("  Uncertain: Normal-predicted (direct): %d  (SVM argmax=Normal, skip DBSCAN)\n",
            n_unc_normal);
    lprintf("  Uncertain: attack-predicted (DBSCAN): %d (capped at %d; "
            "%d overflow -> PRED_NOVEL)\n",
            n_unc_attack, N_MAX_UNCERTAIN, n_unc_attack - nu);
    lprintf("  DBSCAN pool total:                    %d\n", n_db);

    // Step 2b: Build X_db — each row is the 4-D vote-count vector
    //   Holdout anchors: pool [0, nh);  Uncertain attacks: pool [nh, nh+nu)
    std::vector<float> X_db((size_t)n_db * 4);
    for (int i = 0; i < nh; ++i)
        for (int k = 0; k < 4; ++k)
            X_db[(size_t)i * 4 + k] = scores_all[(size_t)ho_idx[i] * N_CLASSES + k];
    for (int i = 0; i < nu; ++i)
        for (int k = 0; k < 4; ++k)
            X_db[(size_t)(nh + i) * 4 + k] = scores_all[(size_t)unc_idx[i] * N_CLASSES + k];

    std::vector<int> ho_orig(nh);
    for (int i = 0; i < nh; ++i) ho_orig[i] = y_test[ho_idx[i]];

    // Step 2c: Normal-class centroid in 4-D vote space (reference point)
    std::array<double, 4> nsum = {0.0, 0.0, 0.0, 0.0};
    int n_norm = 0;
    for (int i = 0; i < N_test; ++i) {
        if (y_test[i] == NORMAL_ORIG_ID) {
            for (int k = 0; k < 4; ++k)
                nsum[k] += scores_all[(size_t)i * N_CLASSES + k];
            ++n_norm;
        }
    }
    std::vector<float> norm_ctr(4, 0.f);
    if (n_norm > 0)
        for (int k = 0; k < 4; ++k)
            norm_ctr[k] = (float)(nsum[k] / n_norm);
    lprintf("  Normal centroid (n=%d): [%.3f, %.3f, %.3f, %.3f]\n",
            n_norm,
            (double)norm_ctr[0], (double)norm_ctr[1],
            (double)norm_ctr[2], (double)norm_ctr[3]);

    // Step 2d: Run DBSCAN (serial BFS; pass 1 & 2 parallelised inside)
    int n_clusters = 0;
    std::vector<int> cluster_id = run_dbscan(X_db, n_db, EPS, MIN_SAMPLES,
                                              n_clusters);

    // Step 2e: Compute cluster centroids
    std::vector<std::array<double, 4>> centroid(n_clusters,
                                                 std::array<double,4>{0,0,0,0});
    std::vector<int> csz(n_clusters, 0);
    for (int i = 0; i < n_db; ++i) {
        int c = cluster_id[i]; if (c < 0) continue;
        for (int d = 0; d < 4; ++d)
            centroid[c][d] += X_db[(size_t)i * 4 + d];
        ++csz[c];
    }
    for (int c = 0; c < n_clusters; ++c)
        if (csz[c] > 0)
            for (int d = 0; d < 4; ++d) centroid[c][d] /= csz[c];

    // Step 2f: Label each cluster (identical logic to mpi_infer)
    std::vector<std::unordered_map<int, int>> votes(n_clusters);
    for (int i = 0; i < nh; ++i) {
        int c = cluster_id[i];
        if (c >= 0) votes[c][ho_orig[i]]++;
    }

    std::vector<int> cluster_class(n_clusters, PRED_UNKNOWN);
    int cnt_by_holdout = 0, cnt_normal = 0, cnt_unknown = 0;
    for (int c = 0; c < n_clusters; ++c) {
        if (!votes[c].empty()) {
            int best_class = -1, best_votes = 0;
            for (auto& [cl, v] : votes[c])
                if (v > best_votes) { best_votes = v; best_class = cl; }
            cluster_class[c] = best_class;
            ++cnt_by_holdout;
        } else {
            // Check centroid argmax AND proximity to Normal centroid
            int ctr_argmax = 0;
            for (int d = 1; d < 4; ++d)
                if (centroid[c][d] > centroid[c][ctr_argmax]) ctr_argmax = d;

            double d2 = 0.0;
            for (int d = 0; d < 4; ++d) {
                double diff = centroid[c][d] - norm_ctr[d];
                d2 += diff * diff;
            }
            if (ctr_argmax == NORMAL_INT &&
                d2 < (double)NORMAL_PROX_THRESH * NORMAL_PROX_THRESH) {
                cluster_class[c] = NORMAL_ORIG_ID;
                ++cnt_normal;
            } else {
                cluster_class[c] = PRED_UNKNOWN;
                ++cnt_unknown;
            }
        }
    }
    lprintf("  Cluster labelling: %d holdout-voted, %d reclassified Normal, "
            "%d unknown anomaly\n", cnt_by_holdout, cnt_normal, cnt_unknown);

    // Step 2g: Save dbscan_model.bin (same binary format as mpi_infer)
    {
        std::ofstream df(DBSCAN_MDL_OUT, std::ios::binary);
        if (!df) {
            fprintf(stderr, "Cannot open %s\n", DBSCAN_MDL_OUT);
            exit(1);
        }
        df.write(reinterpret_cast<const char*>(&n_clusters),   sizeof(int));
        float eps_save = EPS;
        df.write(reinterpret_cast<const char*>(&eps_save),     sizeof(float));
        df.write(reinterpret_cast<const char*>(&MIN_SAMPLES),  sizeof(int));
        float npt_save = NORMAL_PROX_THRESH;
        df.write(reinterpret_cast<const char*>(&npt_save),     sizeof(float));
        for (int c = 0; c < n_clusters; ++c) {
            float ctr[4] = {
                (float)centroid[c][0], (float)centroid[c][1],
                (float)centroid[c][2], (float)centroid[c][3]
            };
            df.write(reinterpret_cast<const char*>(ctr),               4 * sizeof(float));
            df.write(reinterpret_cast<const char*>(&cluster_class[c]), sizeof(int));
        }
        df.close();
        lprintf("[%s] DBSCAN model saved -> %s  (%d clusters)\n",
                ts().c_str(), DBSCAN_MDL_OUT, n_clusters);
    }

    double t1_phase2 = omp_get_wtime();
    lprintf("[%s] Phase 2 (DBSCAN clustering) time: %.3fs\n",
            ts().c_str(), t1_phase2 - t0_phase2);

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 3 — HYBRID COMBINATION  (parallel assignment + serial eval)
    //
    //   Build a reverse-lookup map from sample index to pool position for
    //   uncertain-attack and holdout samples, then parallelise the final
    //   prediction assignment with omp parallel for.
    // ══════════════════════════════════════════════════════════════════════

    lprintf("\n[%s] ════════════════════════════════════════════════════\n"
            "[%s] PHASE 3 — HYBRID COMBINATION  (SVM->DBSCAN fallback)\n"
            "[%s] ════════════════════════════════════════════════════\n",
            ts().c_str(), ts().c_str(), ts().c_str());
    lprintf("  Logic: conf >= %.2f → SVM argmax;\n"
            "         conf < %.2f AND pred=Normal → Normal direct (no DBSCAN);\n"
            "         conf < %.2f AND pred=attack → DBSCAN cluster;\n"
            "         holdout class → DBSCAN (novel detection).\n",
            (double)CONF_THRESHOLD,
            (double)CONF_THRESHOLD,
            (double)CONF_THRESHOLD);

    double t0_phase3 = omp_get_wtime();

    // Build index-to-pool-position maps for DBSCAN-assigned samples
    // (serial, O(nh + nu) — negligible)
    std::vector<int> sample_to_pool(N_test, -1);  // -1 = not in DBSCAN pool
    for (int hi = 0; hi < nh; ++hi)
        sample_to_pool[ho_idx[hi]] = hi;           // holdout: pool [0, nh)
    for (int ui = 0; ui < nu; ++ui)
        sample_to_pool[unc_idx[ui]] = nh + ui;     // uncertain attack: pool [nh, nh+nu)

    // Mark uncertain-Normal samples with a special sentinel (-2) so the
    // parallel loop can assign NORMAL_ORIG_ID without a separate hash lookup
    static constexpr int DIRECT_NORMAL_SENTINEL = -3;
    std::vector<int> sample_role(N_test, 0);  // 0 = confident, -3 = direct-Normal
    for (int idx : unc_normal_idx)
        sample_role[idx] = DIRECT_NORMAL_SENTINEL;

    // Step 3a: Parallel assignment of final predictions
    std::vector<int> final_pred(N_test, PRED_NOVEL);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N_test; ++i) {
        if (!is_holdout(y_test[i])) {
            if (conf_all[i] >= CONF_THRESHOLD) {
                // Confident non-holdout → SVM argmax mapped to original class
                final_pred[i] = INT2ORIG[pred_all[i]];
                continue;
            }
            if (sample_role[i] == DIRECT_NORMAL_SENTINEL) {
                // Uncertain but SVM says Normal → assign Normal directly
                final_pred[i] = NORMAL_ORIG_ID;
                continue;
            }
        }
        // Holdout OR uncertain attack-class → use DBSCAN cluster label
        int pool_pos = sample_to_pool[i];
        if (pool_pos >= 0) {
            int c = cluster_id[pool_pos];
            if (c < 0) {
                final_pred[i] = PRED_NOVEL;
            } else {
                int cls = cluster_class[c];
                // Unclaimed cluster: DBSCAN found no pattern → fall back to SVM argmax
                final_pred[i] = (cls == PRED_UNKNOWN && !is_holdout(y_test[i]))
                                ? INT2ORIG[pred_all[i]]
                                : cls;
}
        }
        // else: overflow uncertain attack (above cap) keeps PRED_NOVEL
    }

    double t1_phase3 = omp_get_wtime();
    lprintf("[%s] Phase 3 (hybrid combination) time: %.3fs\n",
            ts().c_str(), t1_phase3 - t0_phase3);

    // ── Evaluation (serial, identical layout to mpi_infer) ────────────────

    lprintf("\n[%s] ── Evaluation ──────────────────────────────────────\n",
            ts().c_str());

    // (A) Per-class confusion matrix for the 4 trained classes
    long cm[N_CLASSES][N_CLASSES] = {};
    for (int i = 0; i < N_test; ++i) {
        int orig = y_test[i];
        if (orig < 0 || orig >= N_ALL) continue;
        int ti = ORIG2INT[orig]; if (ti < 0) continue;   // holdout: skip
        int fp = final_pred[i];
        if (fp < 0 || fp >= N_ALL) continue;
        int pi = ORIG2INT[fp]; if (pi < 0) continue;
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

    // Overall accuracy from confusion matrix
    long diag_sum = 0, total_sum = 0;
    for (int k = 0; k < N_CLASSES; ++k) {
        diag_sum += cm[k][k];
        for (int j = 0; j < N_CLASSES; ++j) total_sum += cm[k][j];
    }
    float accuracy_hybrid = total_sum > 0 ? (float)diag_sum / (float)total_sum : 0.f;
    lprintf("  %-16s                          %8.4f  (hybrid)\n",
            "accuracy", (double)accuracy_hybrid);

    // GFLOP estimates
    //   SVM inference : N_test × N_PAIRS × N_FEATURES × 2 MACs
    //   DBSCAN        : 2 passes × n_db² pairs × dist4_sq cost
    //                   dist4_sq = 4 SUB + 4 MUL + 3 ADD = 11 FLOPs
    long   svm_flops     = (long)N_test * (long)N_PAIRS * N_FEATURES * 2;
    long   dbscan_flops  = 2LL * n_db * n_db * 11;
    double phase1_s      = t1_phase1 - t0_phase1;
    double phase2_s      = t1_phase2 - t0_phase2;
    double phase3_s      = t1_phase3 - t0_phase3;
    double predict_gflops = phase1_s > 0.0
                            ? (double)svm_flops   / (phase1_s * 1e9) : 0.0;
    double dbscan_gflops  = (phase2_s > 0.0 && n_db > 0)
                            ? (double)dbscan_flops / (phase2_s * 1e9) : 0.0;

    // (B) Holdout detection via DBSCAN
    lprintf("\n  Holdout class detection via DBSCAN:\n");
    lprintf("  %-16s  %8s  %8s  %8s\n", "Class", "Total", "Correct", "Noise");
    long ho_total_all = 0, ho_correct_all = 0;
    long bots_tot = 0, bots_cor = 0, bots_noi = 0;
    long bf_tot   = 0, bf_cor   = 0, bf_noi   = 0;
    long wa_tot   = 0, wa_cor   = 0, wa_noi   = 0;
    for (int hi = 0; hi < nh; ++hi) {
        int orig    = ho_orig[hi];
        int c       = cluster_id[hi];
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
    long ub = 0, uh = 0, un = 0, uu = 0;
    for (int ui = 0; ui < nu; ++ui) {
        int p = final_pred[unc_idx[ui]];
        if      (p == PRED_NOVEL)              ++un;
        else if (p == PRED_UNKNOWN)            ++uu;
        else if (p == NORMAL_ORIG_ID)          ++ub;
        else if (p == 0 || p == 1 || p == 6)  ++uh;
    }
    lprintf("\n  Uncertain pool breakdown (%d total uncertain):\n", n_full_uncertain);
    lprintf("  %-42s  %d\n",  "Direct Normal (SVM argmax=Normal, skip DBSCAN)", n_unc_normal);
    lprintf("  %-42s  %d\n",  "Attack-class uncertain sent to DBSCAN",           nu);
    lprintf("    %-40s  %ld\n", "-> Reclassified Normal (near-Normal cluster)",  ub);
    lprintf("    %-40s  %ld\n", "-> Matched holdout pattern (Bots/BF/WebAtk)",   uh);
    lprintf("    %-40s  %ld\n", "-> Novel attack (DBSCAN noise -1)",              un);
    lprintf("    %-40s  %ld\n", "-> Unknown anomaly (unclaimed cluster)",         uu);

    // ── Write hybrid_predictions.csv ──────────────────────────────────────
    {
        std::ofstream pf(PRED_OUT);
        if (!pf) {
            fprintf(stderr, "Cannot open %s\n", PRED_OUT);
            exit(1);
        }
        for (int i = 0; i < N_test; ++i)
            pf << y_test[i] << "," << final_pred[i] << "\n";
        pf.close();
        lprintf("\n[%s] Hybrid predictions    -> %s\n", ts().c_str(), PRED_OUT);
    }

    // ── Write hybrid_log.csv ──────────────────────────────────────────────
    {
        std::ofstream lf(LOG_OUT);
        if (!lf) {
            fprintf(stderr, "Cannot open %s\n", LOG_OUT);
            exit(1);
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
        lf << "dbscan_novel,"               << un               << "\n";
        lf << "dbscan_unknown,"             << uu               << "\n";
        {
            double t_total_end = omp_get_wtime();
            double total_s     = t_total_end - t_total_start;
            lf << "accuracy_hybrid,"   << accuracy_hybrid          << "\n";
            lf << "predict_ms,"        << (phase1_s * 1000.0)      << "\n";
            lf << "dbscan_ms,"         << (phase2_s * 1000.0)      << "\n";
            lf << "phase3_ms,"         << (phase3_s * 1000.0)      << "\n";
            lf << "total_ms,"          << (total_s  * 1000.0)      << "\n";
            lf << "phase1_time_s,"     << phase1_s                 << "\n";
            lf << "phase2_time_s,"     << phase2_s                 << "\n";
            lf << "phase3_time_s,"     << phase3_s                 << "\n";
            lf << "total_time_s,"      << total_s                  << "\n";
            lf << "predict_gflops,"    << predict_gflops            << "\n";
            lf << "dbscan_gflops,"     << dbscan_gflops             << "\n";
            lprintf("[%s] predict_ms=%.2f  dbscan_ms=%.2f  total_ms=%.2f"
                    "  predict_GFLOPS=%.2f  dbscan_GFLOPS=%.2f\n",
                    ts().c_str(),
                    phase1_s * 1000.0, phase2_s * 1000.0, total_s * 1000.0,
                    predict_gflops, dbscan_gflops);
        }
        lf.close();
        lprintf("[%s] Summary log           -> %s\n", ts().c_str(), LOG_OUT);
    }

    lprintf("[%s] Done.  Output files:\n", ts().c_str());
    lprintf("       SVM-only preds  : %s\n", SVM_PRED_OUT);
    lprintf("       DBSCAN model    : %s\n", DBSCAN_MDL_OUT);
    lprintf("       Hybrid preds    : %s\n", PRED_OUT);
    lprintf("       Metrics log     : %s\n", LOG_OUT);
    lprintf("       Run log         : %s\n", RUN_LOG);

    log_close();
    return 0;
}
