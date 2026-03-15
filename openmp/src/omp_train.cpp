/*
 * omp_train.cpp  —  OpenMP One-vs-One linear SVM training (Adam + cosine LR)
 *
 * Purpose:
 *   Train 6 binary OvO linear SVMs on CICIDS2017 using shared-memory SGD
 *   with Adam optimisation over OpenMP threads.  Drop-in replacement for
 *   mpi_train.cpp: produces model.bin binary format so that either omp_train
 *   or mpi_train output can feed omp_infer or mpi_infer.
 *
 * Algorithm summary:
 *   1. Main thread reads train_data.csv / train_labels.csv, remaps labels
 *      via ORIG2INT, drops holdout rows (orig ∈ {0,1,6}), shuffles (seed=42).
 *   2. omp_set_num_threads(T); all threads share W[6×52] and b[6] (init 0).
 *   3. Thread data partition (mirrors MPI_Scatter + remainder logic):
 *        base_n = N_train / T
 *        thread t owns rows [t*base_n, (t+1)*base_n)   for t < T-1
 *        last thread owns  [(T-1)*base_n, N_train)      (absorbs remainder)
 *   4. Per-epoch inside #pragma omp parallel:
 *        a. Thread 0 (omp single) resets shared gradient buffers, global_loss.
 *        b. Each thread samples BATCH_SIZE from its partition:
 *           seed = thread_id * 1000 + epoch  (matches MPI rank*1000+epoch).
 *        c. Each thread computes OvO hinge-loss gradient (dW_local, db_local)
 *           with L2 penalty LAMBDA * W[p][f].
 *        d. Thread writes gradient into its private slot in all_dW/all_db
 *           (no critical section — slots are non-overlapping).
 *        e. #pragma omp barrier: all slots written before reduction.
 *        f. #pragma omp single: reduce slots 0..T-1 in fixed order →
 *           deterministic mean gradient; Adam update on shared W, b.
 *        g. Implicit barrier after single: all threads see updated W, b.
 *        h. Every VAL_INTERVAL epochs: #pragma omp for schedule(static)
 *           evaluates all N_train rows in parallel using OvO voting, reduces
 *           per-thread confusion matrices via critical, computes macro-F1 in
 *           single; sets volatile stop_flag=1 on early-stopping condition.
 *   5. Main thread writes:
 *        model.bin         — int32 N_PAIRS=6, int32 N_FEATURES=52,
 *                            float W[6*52], float b[6]
 *        training_log.csv  — epoch,lr,train_loss,val_acc,val_macro_f1
 *
 * Hyperparameters (identical to MPI version):
 *   EPOCHS=200, BATCH_SIZE=512, LAMBDA=5e-4, LR_PEAK=5e-2, LR_MIN=1e-3
 *   ADAM_BETA1=0.9, ADAM_BETA2=0.999, ADAM_EPS=1e-8
 *   PATIENCE=30, MIN_DELTA=1e-4, VAL_INTERVAL=10
 *
 * Build:
 *   g++ -O3 -std=c++17 -fopenmp -o omp_train omp_train.cpp -lm
 *
 * Run:
 *   OMP_NUM_THREADS=4 ./omp_train
 *
 * Output files:
 *   model.bin          — SVM model (OvO: N_PAIRS=6 classifiers, K=6, F=52)
 *   training_log.csv   — epoch,lr,train_loss,val_acc,val_macro_f1
 */

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
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
// Configuration — keep identical to mpi_train.cpp
// ---------------------------------------------------------------------------
static constexpr int   N_FEATURES   = 52;
static constexpr int   N_CLASSES    = 4;
static constexpr int   N_PAIRS      = N_CLASSES * (N_CLASSES - 1) / 2; // 6
static constexpr int   PAIR_I[N_PAIRS] = {0, 0, 0, 1, 1, 2};
static constexpr int   PAIR_J[N_PAIRS] = {1, 2, 3, 2, 3, 3};
static constexpr int   N_ALL        = 7;        // original class count

static constexpr int   EPOCHS       = 200;
static constexpr int   BATCH_SIZE   = 512;
static constexpr float LAMBDA       = 5e-4f;
static constexpr float LR_PEAK      = 5e-2f;
static constexpr float LR_MIN       = 1e-3f;
static constexpr float ADAM_BETA1   = 0.9f;
static constexpr float ADAM_BETA2   = 0.999f;
static constexpr float ADAM_EPS     = 1e-8f;
static constexpr int   PATIENCE     = 30;
static constexpr float MIN_DELTA    = 1e-4f;
static constexpr int   VAL_INTERVAL = 10;       // validate every N epochs

// Class mappings (index = original ID 0-6)
static constexpr int ORIG2INT[N_ALL]     = { -1, -1,  0,  1,  2,  3, -1 };
static constexpr int INT2ORIG[N_CLASSES] = {  2,  3,  4,  5 };

// Class weights for hinge loss: DDoS=0, DoS=1, Normal=2, PortScan=3
// Line 121: was { 1.971f, 1.085f, 0.512f, 1.631f }
static constexpr float CLASS_WEIGHTS[N_CLASSES] = { 1.4f, 1.1f, 0.55f, 1.3f };

static const char* TRAIN_DATA_CSV   = "../data/processed/train_data.csv";
static const char* TRAIN_LABELS_CSV = "../data/processed/train_labels.csv";
static const char* MODEL_OUT        = "../model/model.bin";
static const char* LOG_OUT          = "../log/training_log.csv";
static const char* RUN_LOG          = "../log/omp_train_run.log";

// ---------------------------------------------------------------------------
// Timestamp string
// ---------------------------------------------------------------------------
static std::string ts() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return buf;
}

// ---------------------------------------------------------------------------
// CSV readers
// ---------------------------------------------------------------------------
static std::vector<float> load_csv_float(const char* path, int /*expected_cols*/) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    std::vector<float> data;
    data.reserve(600000 * N_FEATURES);
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
    data.reserve(600000);
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) data.push_back(std::stoi(line));
    return data;
}

// ---------------------------------------------------------------------------
// Compute macro-F1 over N_CLASSES given confusion matrix
// ---------------------------------------------------------------------------
static float macro_f1(const long cm[N_CLASSES][N_CLASSES]) {
    float mf1 = 0.f;
    for (int k = 0; k < N_CLASSES; ++k) {
        long tp = cm[k][k], fp = 0, fn = 0;
        for (int j = 0; j < N_CLASSES; ++j) {
            if (j != k) fp += cm[j][k];
            if (j != k) fn += cm[k][j];
        }
        float p  = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.f;
        float r  = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.f;
        float f1 = (p + r > 1e-9f) ? 2.f * p * r / (p + r) : 0.f;
        mf1 += f1;
    }
    return mf1 / N_CLASSES;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/) {

    // Determine thread count (honours OMP_NUM_THREADS env var)
    int N_THREADS = omp_get_max_threads();
    omp_set_num_threads(N_THREADS);

    log_open(RUN_LOG);
    lprintf("OpenMP threads: %d\n", N_THREADS);

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 1 — DATA LOADING (main thread)
    //
    //   Mirrors mpi_train rank-0 data load: remap labels, drop holdouts,
    //   shuffle with seed=42 so every thread's partition has a balanced
    //   class mix (same intent as the MPI shuffle).
    // ══════════════════════════════════════════════════════════════════════

    lprintf("[%s] ── SVM Training (One-vs-One) ── Loading data ...\n", ts().c_str());
    double t0_load = omp_get_wtime();
    auto raw_X = load_csv_float(TRAIN_DATA_CSV, N_FEATURES);
    auto raw_y = load_csv_int(TRAIN_LABELS_CSV);

    int N_raw = (int)raw_y.size();
    assert((int)raw_X.size() == N_raw * N_FEATURES);

    // Remap labels; drop holdout rows (ORIG2INT[orig] == -1)
    std::vector<float> X_all;
    std::vector<int>   y_all;
    X_all.reserve(N_raw * N_FEATURES);
    y_all.reserve(N_raw);
    for (int i = 0; i < N_raw; ++i) {
        int orig = raw_y[i];
        if (orig < 0 || orig >= N_ALL) continue;
        int mapped = ORIG2INT[orig];
        if (mapped < 0) continue;
        for (int f = 0; f < N_FEATURES; ++f)
            X_all.push_back(raw_X[(size_t)i * N_FEATURES + f]);
        y_all.push_back(mapped);
    }
    int N_train = (int)y_all.size();
    lprintf("[%s] N_train (after label remap, holdouts dropped) = %d\n",
            ts().c_str(), N_train);

    // Shuffle rows (seed=42) — identical to MPI version
    {
        std::mt19937 g(42);
        std::vector<int> perm(N_train);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), g);

        std::vector<float> Xs(X_all.size());
        std::vector<int>   ys(N_train);
        for (int i = 0; i < N_train; ++i) {
            ys[i] = y_all[perm[i]];
            for (int f = 0; f < N_FEATURES; ++f)
                Xs[(size_t)i * N_FEATURES + f] = X_all[(size_t)perm[i] * N_FEATURES + f];
        }
        X_all = std::move(Xs);
        y_all = std::move(ys);
    }
    lprintf("[%s] Data shuffled (seed=42) — each thread receives a balanced class mix.\n",
            ts().c_str());
    double t1_load = omp_get_wtime();
    lprintf("[%s] Data load+shuffle time: %.2fs\n", ts().c_str(), t1_load - t0_load);

    {
        int base_n    = N_train / N_THREADS;
        int remainder = N_train % N_THREADS;
        lprintf("[%s] Partition: base=%d/thread, remainder=%d on last thread  "
                "(thread 0 local_n=%d)\n",
                ts().c_str(), base_n, remainder, base_n);

        // Show class distribution in thread 0's partition (diagnostic)
        long cls_cnt[N_CLASSES] = {};
        for (int i = 0; i < base_n; ++i)
            if (y_all[i] >= 0 && y_all[i] < N_CLASSES)
                cls_cnt[y_all[i]]++;
        lprintf("[%s] Thread-0 partition class counts: "
                "DDoS=%ld  DoS=%ld  Normal=%ld  PortScan=%ld\n",
                ts().c_str(),
                cls_cnt[0], cls_cnt[1], cls_cnt[2], cls_cnt[3]);
    }

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 2 — SVM MODEL INITIALISATION (weights + Adam state)
    //
    //   All shared: W, b, Adam moments, global gradient buffers.
    //   Access pattern: all threads READ W/b during gradient computation
    //   (safe — no writes); only the omp-single block writes W/b.
    //   OvO: N_PAIRS=6 pair classifiers, each with N_FEATURES=52 weights.
    // ══════════════════════════════════════════════════════════════════════

    std::vector<float> W (N_PAIRS * N_FEATURES, 0.f);
    std::vector<float> b (N_PAIRS, 0.f);
    std::vector<float> mW(N_PAIRS * N_FEATURES, 0.f);  // Adam 1st moment W
    std::vector<float> vW(N_PAIRS * N_FEATURES, 0.f);  // Adam 2nd moment W
    std::vector<float> mb(N_PAIRS, 0.f);               // Adam 1st moment b
    std::vector<float> vb(N_PAIRS, 0.f);               // Adam 2nd moment b

    // Per-thread gradient slots — each thread writes to its own row so no
    // critical section is needed.  The single block reduces in fixed order
    // 0,1,...,T-1 → bit-identical weights every run given same OMP_NUM_THREADS.
    const int MAX_T = omp_get_max_threads();
    std::vector<float> all_dW  ((size_t)MAX_T * N_PAIRS * N_FEATURES, 0.f);
    std::vector<float> all_db  ((size_t)MAX_T * N_PAIRS, 0.f);
    std::vector<float> all_loss((size_t)MAX_T, 0.f);

    // Shared reduction targets (written only inside omp single blocks)
    std::vector<float> dW_global(N_PAIRS * N_FEATURES, 0.f);
    std::vector<float> db_global(N_PAIRS, 0.f);

    // Shared validation confusion matrix — reset in single, reduced via critical
    long cm_global[N_CLASSES][N_CLASSES];
    long correct_global = 0;
    std::memset(cm_global, 0, sizeof(cm_global));

    volatile int stop_flag    = 0;
    int          t_adam       = 0;   // Adam step counter
    float        best_f1      = -1.f;
    int          patience_cnt = 0;
    int          epochs_run   = 0;   // actual epochs completed (may be < EPOCHS if early-stopped)
    float        best_val_acc = 0.f; // val_acc at best macro-F1 epoch

    // Best-checkpoint: saved whenever validation F1 improves.
    // model.bin uses these weights, not the last-epoch weights.
    std::vector<float> W_best(N_PAIRS * N_FEATURES, 0.f);
    std::vector<float> b_best(N_PAIRS, 0.f);
    bool best_saved = false;
    double t_epoch_s = 0.0;  // epoch wall-clock start (set per epoch in omp single)

    std::ofstream log_file(LOG_OUT);
    if (!log_file) {
        fprintf(stderr, "Cannot open %s for writing\n", LOG_OUT);
        exit(1);
    }
    log_file << "epoch,lr,train_loss,val_acc,val_macro_f1,elapsed_s,epoch_ms\n";

    lprintf("[%s] Starting OvO SVM training: EPOCHS=%d BATCH=%d LAMBDA=%.1e "
            "LR_PEAK=%.3f LR_MIN=%.3f  N_PAIRS=%d\n",
            ts().c_str(), EPOCHS, BATCH_SIZE, (double)LAMBDA,
            (double)LR_PEAK, (double)LR_MIN, N_PAIRS);

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 3 — DISTRIBUTED SVM TRAINING LOOP (OpenMP)
    //
    //   Gradient reduction: each thread writes gradients to its own private
    //   slot in all_dW/all_db.  After a barrier, the single block reduces
    //   slots in thread-id order 0..T-1 → deterministic, reproducible.
    //
    //   Validation: all N_train rows evaluated in parallel via omp for
    //   every VAL_INTERVAL epochs using OvO voting (6 pair classifiers →
    //   vote tallied per class, argmax wins).
    // ══════════════════════════════════════════════════════════════════════

    double train_start = omp_get_wtime();

#pragma omp parallel shared(W, b, mW, vW, mb, vb,                               \
                             all_dW, all_db, all_loss,                            \
                             dW_global, db_global,                                \
                             cm_global, correct_global,                           \
                             stop_flag, t_adam, best_f1, patience_cnt,            \
                             epochs_run, best_val_acc, best_saved,                \
                             W_best, b_best, t_epoch_s,                           \
                             log_file, X_all, y_all, N_train)
    {
        const int tid = omp_get_thread_num();
        const int nt  = omp_get_num_threads();

        // ── Thread partition ─────────────────────────────────────────────
        //   Mirrors MPI scatter: thread t → [t*base_n, (t+1)*base_n)
        //   Last thread absorbs remainder (mirrors rank-0 remainder append)
        const int base_n  = N_train / nt;
        const int t_start = tid * base_n;
        const int t_end   = (tid == nt - 1) ? N_train : (tid + 1) * base_n;
        const int local_n = t_end - t_start;

        const float* X_local = X_all.data() + (size_t)t_start * N_FEATURES;
        const int*   y_local = y_all.data()  + t_start;

        // Per-thread gradient pointers — each thread owns its own row in
        // all_dW/all_db, so no synchronisation is needed for writes.
        float* dW_local = all_dW.data() + (size_t)tid * N_PAIRS * N_FEATURES;
        float* db_local = all_db.data() + (size_t)tid * N_PAIRS;

        // Thread-local confusion matrix for parallel validation
        long cm_local[N_CLASSES][N_CLASSES];
        long correct_local = 0;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {

            // Check early-stopping flag set inside the previous validation single.
            // The implicit barrier at that single's exit flushes stop_flag.
            if (stop_flag) break;

            // Record epoch start time (thread 0; implicit barrier after single)
#pragma omp single
            { t_epoch_s = omp_get_wtime(); }

            // ── (a) Cosine-annealing LR (same formula, computed by all) ──
            const float lr = LR_MIN + 0.5f * (LR_PEAK - LR_MIN) *
                             (1.f + std::cos((float)M_PI * epoch / EPOCHS));

            // ── (b) Reset own gradient slot; sample BATCH_SIZE; compute gradient
            //   Seed: tid*1000 + epoch  (mirrors MPI rank*1000+epoch)
            std::fill(dW_local, dW_local + N_PAIRS * N_FEATURES, 0.f);
            std::fill(db_local, db_local + N_PAIRS, 0.f);
            float loc_loss = 0.f;

            if (local_n > 0) {
                std::mt19937 rng((unsigned)(tid * 1000 + epoch));
                std::uniform_int_distribution<int> dist(0, local_n - 1);

                for (int bi = 0; bi < BATCH_SIZE; ++bi) {
                    int          idx    = dist(rng);
                    const float* x      = X_local + (size_t)idx * N_FEATURES;
                    int          y_true = y_local[idx];

                    // OvO gradient: iterate over all N_PAIRS binary classifiers
                    for (int p = 0; p < N_PAIRS; ++p) {
                        int ci = PAIR_I[p], cj = PAIR_J[p];
                        if (y_true != ci && y_true != cj) continue; // skip irrelevant pairs
                        float score = b[p];
                        for (int f = 0; f < N_FEATURES; ++f) score += W[p * N_FEATURES + f] * x[f];
                        int y_bin = (y_true == ci) ? 1 : -1;
                        float pair_w = (y_bin > 0) ? CLASS_WEIGHTS[ci] : CLASS_WEIGHTS[cj];
                        float margin  = (float)y_bin * score * pair_w;
                        if (margin < 1.f) {
                            float coeff = -(float)y_bin * pair_w;
                            for (int f = 0; f < N_FEATURES; ++f)
                                dW_local[p * N_FEATURES + f] += coeff * x[f];
                            db_local[p] += coeff;
                            loc_loss    += 1.f - margin;
                        }
                    }
                }

                // L2 regularisation gradient (always applied)
                for (int p = 0; p < N_PAIRS; ++p)
                    for (int f = 0; f < N_FEATURES; ++f)
                        dW_local[p * N_FEATURES + f] += LAMBDA * W[p * N_FEATURES + f];
            }

            // ── (c) Store per-thread loss; barrier ensures all slots visible ─
            all_loss[tid] = loc_loss;

#pragma omp barrier

            // ── (d) Deterministic reduction + Adam update (one thread) ────
            //   Threads summed in ascending id order → same floating-point
            //   result every run given the same OMP_NUM_THREADS value.
#pragma omp single
            {
                epochs_run = epoch + 1;  // track last completed epoch
                // Deterministic gradient sum
                std::fill(dW_global.begin(), dW_global.end(), 0.f);
                std::fill(db_global.begin(), db_global.end(), 0.f);
                float total_loss = 0.f;
                for (int t = 0; t < nt; ++t) {
                    const float* dWt = all_dW.data() + (size_t)t * N_PAIRS * N_FEATURES;
                    for (int i = 0; i < N_PAIRS * N_FEATURES; ++i)
                        dW_global[i] += dWt[i];
                    const float* dbt = all_db.data() + (size_t)t * N_PAIRS;
                    for (int i = 0; i < N_PAIRS; ++i)
                        db_global[i] += dbt[i];
                    total_loss += all_loss[t];
                }

                // Mean gradient
                const float inv_nt = 1.f / nt;
                for (auto& v : dW_global) v *= inv_nt;
                for (auto& v : db_global) v *= inv_nt;

                // Adam step
                ++t_adam;
                const float bc1 = 1.f - std::pow(ADAM_BETA1, (float)t_adam);
                const float bc2 = 1.f - std::pow(ADAM_BETA2, (float)t_adam);

                for (int i = 0; i < N_PAIRS * N_FEATURES; ++i) {
                    mW[i] = ADAM_BETA1 * mW[i] + (1.f - ADAM_BETA1) * dW_global[i];
                    vW[i] = ADAM_BETA2 * vW[i] + (1.f - ADAM_BETA2) * dW_global[i] * dW_global[i];
                    float mh = mW[i] / bc1;
                    float vh = vW[i] / bc2;
                    W[i] -= lr * mh / (std::sqrt(vh) + ADAM_EPS);
                }
                for (int i = 0; i < N_PAIRS; ++i) {
                    mb[i] = ADAM_BETA1 * mb[i] + (1.f - ADAM_BETA1) * db_global[i];
                    vb[i] = ADAM_BETA2 * vb[i] + (1.f - ADAM_BETA2) * db_global[i] * db_global[i];
                    float mh = mb[i] / bc1;
                    float vh = vb[i] / bc2;
                    b[i] -= lr * mh / (std::sqrt(vh) + ADAM_EPS);
                }

                // Log non-validation epochs; validation epochs logged in (e) below
                if ((epoch + 1) % VAL_INTERVAL != 0) {
                    const float train_loss = total_loss / (float)(nt * BATCH_SIZE);
                    const double epoch_ms = (omp_get_wtime() - t_epoch_s) * 1000.0;
                    log_file << (epoch + 1) << "," << lr << "," << train_loss
                             << ",0,0," << (omp_get_wtime() - train_start)
                             << "," << epoch_ms << "\n";
                }
            }
            // implicit barrier after single — all threads see updated W, b

            // ── (e) Parallel validation (every VAL_INTERVAL epochs) ───────
            //   All N_train rows distributed across threads via omp for.
            //   OvO voting: each of the 6 pair classifiers casts one vote.
            //   The class with the most votes wins (argmax over votes[]).
            if ((epoch + 1) % VAL_INTERVAL == 0) {
                std::memset(cm_local, 0, sizeof(cm_local));
                correct_local = 0;

#pragma omp for schedule(static)
                for (int i = 0; i < N_train; ++i) {
                    const float* x = X_all.data() + (size_t)i * N_FEATURES;
                    // OvO voting: each pair classifier casts one vote
                    int votes[N_CLASSES] = {};
                    for (int p = 0; p < N_PAIRS; ++p) {
                        float s = b[p];
                        for (int f = 0; f < N_FEATURES; ++f) s += W[p * N_FEATURES + f] * x[f];
                        if (s >= 0.f) votes[PAIR_I[p]]++;
                        else          votes[PAIR_J[p]]++;
                    }
                    int best_k = 0;
                    for (int k = 1; k < N_CLASSES; ++k)
                        if (votes[k] > votes[best_k]) best_k = k;
                    int ytrue = y_all[i];
                    if (ytrue >= 0 && ytrue < N_CLASSES) {
                        cm_local[ytrue][best_k]++;
                        if (best_k == ytrue) ++correct_local;
                    }
                }
                // implicit barrier at end of omp for

                // Reduce thread-local confusion matrices
#pragma omp critical
                {
                    for (int r = 0; r < N_CLASSES; ++r)
                        for (int c = 0; c < N_CLASSES; ++c)
                            cm_global[r][c] += cm_local[r][c];
                    correct_global += correct_local;
                }

#pragma omp barrier  // wait for all critical sections

#pragma omp single
                {
                    // Recompute train_loss (T float additions — negligible)
                    float total_loss = 0.f;
                    for (int t = 0; t < nt; ++t) total_loss += all_loss[t];
                    const float train_loss = total_loss / (float)(nt * BATCH_SIZE);

                    const float val_f1  = macro_f1(cm_global);
                    const float val_acc = (float)correct_global / N_train;

                    lprintf("[%s] epoch %3d/%d  lr=%.5f  loss=%.4f  "
                            "val_acc=%.4f  val_macro_f1=%.4f",
                            ts().c_str(), epoch + 1, EPOCHS,
                            (double)lr, (double)train_loss,
                            (double)val_acc, (double)val_f1);

                    if (val_f1 > best_f1 + MIN_DELTA) {
                        best_f1      = val_f1;
                        best_val_acc = val_acc;
                        patience_cnt = 0;
                        W_best = W;
                        b_best = b;
                        best_saved = true;
                        lprintf("  [new best]\n");
                    } else {
                        ++patience_cnt;
                        lprintf("  [patience %d/%d]\n", patience_cnt, PATIENCE);
                        if (patience_cnt >= PATIENCE) {
                            lprintf("[%s] Early stopping at epoch %d (best_f1=%.4f)\n",
                                    ts().c_str(), epoch + 1, (double)best_f1);
                            stop_flag = 1;
                        }
                    }

                    long rs[N_CLASSES] = {};
                    for (int k = 0; k < N_CLASSES; ++k)
                        for (int j = 0; j < N_CLASSES; ++j)
                            rs[k] += cm_global[k][j];
                    lprintf("         recall: DDoS=%.3f  DoS=%.3f  Normal=%.3f  PortScan=%.3f\n",
                            rs[0] > 0 ? (float)cm_global[0][0]/rs[0] : 0.f,
                            rs[1] > 0 ? (float)cm_global[1][1]/rs[1] : 0.f,
                            rs[2] > 0 ? (float)cm_global[2][2]/rs[2] : 0.f,
                            rs[3] > 0 ? (float)cm_global[3][3]/rs[3] : 0.f);

                    const double epoch_ms = (omp_get_wtime() - t_epoch_s) * 1000.0;
                    log_file << (epoch + 1) << "," << lr << "," << train_loss
                             << "," << val_acc << "," << val_f1
                             << "," << (omp_get_wtime() - train_start)
                             << "," << epoch_ms << "\n";

                    // Reset shared validation state for next validation epoch
                    std::memset(cm_global, 0, sizeof(cm_global));
                    correct_global = 0;
                }
                // implicit barrier — stop_flag flushed before next epoch check
            }
        }
    } // end #pragma omp parallel

    double train_end = omp_get_wtime();
    lprintf("[%s] Training time: %.3fs  (OvO, %d thread(s))\n",
            ts().c_str(), train_end - train_start, N_THREADS);

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 4 — SAVE SVM MODEL
    //
    //   Binary layout (OvO format):
    //     int32  N_PAIRS     (6)
    //     int32  N_FEATURES  (52)
    //     float  W[6*52]     (row-major: W[p*52 + f])
    //     float  b[6]
    // ══════════════════════════════════════════════════════════════════════

    log_file.close();

    // ── Training summary: compute GFLOPS and append to training_log.csv ──────
    //
    //   FLOP model (dominant terms):
    //   • Gradient: N_THREADS × BATCH_SIZE × (N_CLASSES-1) pairs per sample
    //               × 4×N_FEATURES FLOPs (score dot + gradient, each 2×N_FEATURES MACs)
    //   • L2 reg  : N_PAIRS × N_FEATURES × 2 per epoch (single-thread step)
    //   • Validation: every VAL_INTERVAL epochs, N_train rows × N_PAIRS × 2×N_FEATURES
    {
        double train_s = train_end - train_start;
        long   grad_f  = (long)N_THREADS * BATCH_SIZE * (N_CLASSES - 1)
                         * 4L * N_FEATURES * epochs_run;
        long   l2_f    = (long)N_PAIRS * N_FEATURES * 2L * epochs_run;
        int    n_val   = (epochs_run + VAL_INTERVAL - 1) / VAL_INTERVAL;
        long   val_f   = (long)N_train * N_PAIRS * 2L * N_FEATURES * n_val;
        double gflops  = (grad_f + l2_f + val_f) / (train_s * 1e9);

        // Append key,value summary rows after the per-epoch data
        std::ofstream sf(LOG_OUT, std::ios::app);
        sf << "train_ms,"     << (train_s * 1000.0) << "\n";
        sf << "accuracy,"     << best_val_acc        << "\n";
        sf << "macro_f1,"     << best_f1             << "\n";
        sf << "train_gflops," << gflops              << "\n";
        sf.close();

        lprintf("[%s] train_ms=%.1f  accuracy=%.4f  macro_f1=%.4f  GFLOPS=%.2f\n",
                ts().c_str(), train_s * 1000.0, (double)best_val_acc, (double)best_f1, gflops);
    }

    std::ofstream mf(MODEL_OUT, std::ios::binary);
    if (!mf) {
        fprintf(stderr, "Cannot open %s for writing\n", MODEL_OUT);
        exit(1);
    }
    int dims[2] = { N_PAIRS, N_FEATURES };
    mf.write(reinterpret_cast<const char*>(dims),     sizeof(dims));
    const std::vector<float>& W_save = best_saved ? W_best : W;
    const std::vector<float>& b_save = best_saved ? b_best : b;
    mf.write(reinterpret_cast<const char*>(W_save.data()), N_PAIRS * N_FEATURES * sizeof(float));
    mf.write(reinterpret_cast<const char*>(b_save.data()), N_PAIRS * sizeof(float));
    mf.close();

    lprintf("[%s] SVM model saved -> %s  (K=%d F=%d)  [%s weights]\n",
            ts().c_str(), MODEL_OUT, N_PAIRS, N_FEATURES,
            best_saved ? "best-checkpoint" : "final-epoch");
    lprintf("[%s] Training log    -> %s\n", ts().c_str(), LOG_OUT);
    lprintf("[%s] Run log         -> %s\n", ts().c_str(), RUN_LOG);
    lprintf("[%s] Done. Run omp_infer to apply the hybrid SVM+DBSCAN pipeline.\n",
            ts().c_str());

    log_close();
    return 0;
}
