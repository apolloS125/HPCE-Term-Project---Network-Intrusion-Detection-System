/*
 * mpi_train.cpp  —  Distributed One-vs-One linear SVM training (MPI + Adam)
 *
 * Purpose:
 *   Train 6 binary OvO linear SVMs on CICIDS2017 using distributed SGD with
 *   Adam optimisation over MPI ranks.  Produces a binary SVM model file
 *   (model.bin) and a per-epoch training log (training_log.csv).
 *
 *   This file is SVM-ONLY.  DBSCAN clustering is performed at inference time
 *   by mpi_infer.cpp, which loads model.bin produced here.
 *
 * Algorithm summary:
 *   1. Rank 0 reads train_data.csv / train_labels.csv, remaps labels via
 *      ORIG2INT, drops rows whose original label is a holdout class.
 *   2. MPI_Bcast: N_train, N_FEATURES, N_CLASSES.
 *   3. MPI_Scatter: each rank receives floor(N_train/size) rows.
 *      Rank 0 additionally retains the last (N_train % size) rows
 *      (appended to its local partition after the scatter).
 *   4. Each rank maintains identical W[6×52] and b[6], initialised to 0.
 *   5. Per epoch:
 *        a. Sample BATCH_SIZE indices (with replacement) from local partition.
 *           Seed: rank*1000 + epoch.
 *        b. Hinge-loss gradient for each OvO classifier + L2 penalty.
 *        c. MPI_Allreduce(SUM) → divide by comm_size → mean gradient.
 *        d. Adam update.
 *        e. Cosine-annealing learning rate.
 *   6. Every 10 epochs, rank 0 evaluates on its full local partition:
 *      macro-F1 computed; early stopping if no MIN_DELTA improvement
 *      for PATIENCE epochs. Stop flag broadcast to all ranks.
 *   7. Rank 0 writes model.bin, training_log.csv, and train_results.csv.
 *
 * Reproducibility note:
 *   Rank 0 output is identical for runs with 1, 2, or 4 ranks *except* for
 *   tiny floating-point differences arising from non-associative summation
 *   order in MPI_Allreduce when comm_size differs.
 *
 * Build:
 *   mpicxx -O3 -std=c++17 -o mpi_train mpi_train.cpp -lm
 *
 * Run:
 *   mpirun -np 4 ./mpi_train
 *
 * Expected output files:
 *   model.bin          — SVM model: int32 N_PAIRS=6, int32 N_FEATURES=52,
 *                        float W[6*52], float b[6]
 *   training_log.csv   — per-epoch: epoch,lr,train_loss,val_acc,val_macro_f1,elapsed_s
 *   train_results.csv  — one-row summary: n_ranks,n_train,epochs_run,
 *                        val_acc,val_f1,best_f1,loss_avg,train_time_ms,train_gflops
 */

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
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
static constexpr float CLASS_WEIGHTS[N_CLASSES] = { 1.971f, 1.085f, 0.512f, 1.631f };

static const char* TRAIN_DATA_CSV   = "data/processed/train_data.csv";
static const char* TRAIN_LABELS_CSV = "data/processed/train_labels.csv";
static const char* MODEL_OUT        = "model.bin";
static const char* LOG_OUT          = "training_log.csv";
static const char* TRAIN_RESULTS    = "train_results.csv";
static const char* RUN_LOG          = "mpi_train_run.log";

// ---------------------------------------------------------------------------
// Path helpers — allow running from either mpi/ or mpi/src directories
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
// CSV readers (rank 0 only)
// ---------------------------------------------------------------------------
static std::vector<float> load_csv_float(const char* path, int expected_cols) {
    std::string resolved = resolve_input_path(path);
    std::ifstream f(resolved);
    if (!f) {
        std::string cwd = current_working_dir();
        fprintf(stderr, "[rank 0] Cannot open %s (cwd=%s)\n",
                resolved.c_str(), cwd.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<float> data;
    data.reserve(600000 * expected_cols);
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
    data.reserve(600000);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty())
            data.push_back(std::stoi(line));
    }
    return data;
}

// ---------------------------------------------------------------------------
// Compute macro-F1 over 4 classes given a confusion matrix
// ---------------------------------------------------------------------------
static float macro_f1(const long cm[N_CLASSES][N_CLASSES]) {
    float mf1 = 0.f;
    for (int k = 0; k < N_CLASSES; ++k) {
        long tp = cm[k][k], fp = 0, fn = 0;
        for (int j = 0; j < N_CLASSES; ++j) {
            if (j != k) fp += cm[j][k];
            if (j != k) fn += cm[k][j];
        }
        float p = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.f;
        float r = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.f;
        float f1 = (p + r > 1e-9f) ? 2.f * p * r / (p + r) : 0.f;
        mf1 += f1;
    }
    return mf1 / N_CLASSES;
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

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 1 — DATA LOADING (rank 0) + BROADCAST + SCATTER
    // ══════════════════════════════════════════════════════════════════════

    std::vector<float> X_all;
    std::vector<int>   y_all;
    int N_train = 0;

    if (rank == 0) {
        lprintf("[%s] ── SVM Training ── Loading data ...\n", ts().c_str());
        double t0_load = MPI_Wtime();
        auto raw_X = load_csv_float(TRAIN_DATA_CSV, N_FEATURES);
        auto raw_y = load_csv_int(TRAIN_LABELS_CSV);

        int N_raw = (int)raw_y.size();
        assert((int)raw_X.size() == N_raw * N_FEATURES);

        // Remap labels; drop holdout rows (ORIG2INT[orig] == -1)
        X_all.reserve(N_raw * N_FEATURES);
        y_all.reserve(N_raw);
        for (int i = 0; i < N_raw; ++i) {
            int orig = raw_y[i];
            if (orig < 0 || orig >= N_ALL) continue;
            int mapped = ORIG2INT[orig];
            if (mapped < 0) continue;               // holdout — skip
            for (int f = 0; f < N_FEATURES; ++f)
                X_all.push_back(raw_X[(size_t)i * N_FEATURES + f]);
            y_all.push_back(mapped);
        }
        N_train = (int)y_all.size();
        lprintf("[%s] N_train (after label remap, holdouts dropped) = %d\n",
               ts().c_str(), N_train);

        // Shuffle rows before scatter so every rank gets a proportional class mix.
        // Without this, a class-sorted CSV makes rank 0's validation partition
        // contain only the first class(es), making val_macro_f1 unreliable and
        // early stopping blind to Normal/PortScan.  Seed fixed for reproducibility.
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
        lprintf("[%s] Data shuffled (seed=42) — each rank receives a balanced class mix.\n",
               ts().c_str());
        double t1_load = MPI_Wtime();
        lprintf("[%s] Data load+shuffle time: %.2fs\n", ts().c_str(), t1_load - t0_load);
    }

    // Broadcast sizes to all ranks
    int bcast_dims[2] = { N_train, N_FEATURES };
    MPI_CHECK(MPI_Bcast(bcast_dims, 2, MPI_INT, 0, MPI_COMM_WORLD));
    N_train = bcast_dims[0];
    assert(bcast_dims[1] == N_FEATURES);

    // Scatter: floor(N_train/size) rows per rank; rank 0 keeps remainder
    int base_n    = N_train / world_size;
    int remainder = N_train % world_size;   // kept by rank 0

    std::vector<float> X_scatter(base_n * N_FEATURES);
    std::vector<int>   y_scatter(base_n);

    MPI_CHECK(MPI_Scatter(
        rank == 0 ? X_all.data() : nullptr, base_n * N_FEATURES, MPI_FLOAT,
        X_scatter.data(),                    base_n * N_FEATURES, MPI_FLOAT,
        0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatter(
        rank == 0 ? y_all.data() : nullptr, base_n, MPI_INT,
        y_scatter.data(),                    base_n, MPI_INT,
        0, MPI_COMM_WORLD));

    // Rank 0 appends remainder rows to its local partition
    int local_n = base_n;
    std::vector<float> X_local = X_scatter;
    std::vector<int>   y_local = y_scatter;

    if (rank == 0 && remainder > 0) {
        int rem_off = world_size * base_n;
        X_local.insert(X_local.end(),
                        X_all.begin() + (size_t)rem_off * N_FEATURES,
                        X_all.end());
        y_local.insert(y_local.end(),
                        y_all.begin() + rem_off,
                        y_all.end());
        local_n += remainder;
    }

    if (rank == 0) {
        lprintf("[%s] Partition: base=%d/rank, remainder=%d on rank 0  "
               "(rank 0 local_n=%d)\n",
               ts().c_str(), base_n, remainder, local_n);

        // Show class distribution in rank 0's local partition (diagnostic).
        long cls_cnt[N_CLASSES] = {};
        for (int i = 0; i < local_n; ++i)
            if (y_local[i] >= 0 && y_local[i] < N_CLASSES)
                cls_cnt[y_local[i]]++;
        lprintf("[%s] Rank-0 partition class counts: "
               "DDoS=%ld  DoS=%ld  Normal=%ld  PortScan=%ld\n",
               ts().c_str(),
               cls_cnt[0], cls_cnt[1], cls_cnt[2], cls_cnt[3]);
    }

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 2 — SVM MODEL INITIALISATION (weights + Adam state)
    // ══════════════════════════════════════════════════════════════════════

    std::vector<float> W(N_PAIRS * N_FEATURES, 0.f);
    std::vector<float> b(N_PAIRS, 0.f);

    std::vector<float> mW(N_PAIRS * N_FEATURES, 0.f);  // Adam 1st moment W
    std::vector<float> vW(N_PAIRS * N_FEATURES, 0.f);  // Adam 2nd moment W
    std::vector<float> mb(N_PAIRS, 0.f);               // Adam 1st moment b
    std::vector<float> vb(N_PAIRS, 0.f);               // Adam 2nd moment b

    std::vector<float> dW_local (N_PAIRS * N_FEATURES);
    std::vector<float> db_local (N_PAIRS);
    std::vector<float> dW_global(N_PAIRS * N_FEATURES);
    std::vector<float> db_global(N_PAIRS);

    float best_f1      = -1.f;
    int   patience_cnt = 0;
    int   stop_flag    = 0;

    // Best-checkpoint: saved whenever validation F1 improves.
    // model.bin uses these weights, not the last-epoch weights.
    std::vector<float> W_best(N_PAIRS * N_FEATURES, 0.f);
    std::vector<float> b_best(N_PAIRS, 0.f);
    float best_val_acc = 0.f;
    bool  best_saved   = false;

    std::ofstream log_file;
    if (rank == 0) {
        log_file.open(LOG_OUT);
        if (!log_file) {
            fprintf(stderr, "[rank 0] Cannot open %s for writing\n", LOG_OUT);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        log_file << "epoch,lr,train_loss,val_acc,val_macro_f1,elapsed_s,epoch_ms\n";
    }

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 3 — DISTRIBUTED SVM TRAINING LOOP (One-vs-One, Adam+SGD)
    //
    //   Each rank independently samples from its local partition, computes
    //   hinge-loss gradients with L2 regularisation, then all ranks
    //   synchronise via MPI_Allreduce → mean gradient → Adam step.
    //   Learning rate follows cosine annealing per epoch.
    // ══════════════════════════════════════════════════════════════════════

    if (rank == 0)
        lprintf("[%s] Starting SVM training: EPOCHS=%d BATCH=%d LAMBDA=%.1e "
               "LR_PEAK=%.3f LR_MIN=%.3f\n",
               ts().c_str(), EPOCHS, BATCH_SIZE, (double)LAMBDA,
               (double)LR_PEAK, (double)LR_MIN);

    int t_adam = 0;  // Adam time-step (incremented each epoch)

    // Accumulators for train_results.csv summary
    float sum_global_loss = 0.f;
    int   loss_epochs     = 0;
    float last_val_acc    = 0.f;
    float last_val_f1     = 0.f;

    double t_train_start = MPI_Wtime();

    for (int epoch = 0; epoch < EPOCHS && !stop_flag; ++epoch) {
        double t_epoch_start = (rank == 0) ? MPI_Wtime() : 0.0;

        // 3a. Cosine annealing learning rate
        float lr = LR_MIN + 0.5f * (LR_PEAK - LR_MIN) *
                   (1.f + std::cos((float)M_PI * epoch / EPOCHS));

        // 3b. Sample BATCH_SIZE indices (with replacement). Seed = rank*1000+epoch.
        std::mt19937 rng(rank * 1000 + epoch);
        std::uniform_int_distribution<int> dist(0, local_n - 1);

        std::fill(dW_local.begin(), dW_local.end(), 0.f);
        std::fill(db_local.begin(), db_local.end(), 0.f);
        float batch_loss = 0.f;

        for (int bi = 0; bi < BATCH_SIZE; ++bi) {
            int idx   = dist(rng);
            const float* x = X_local.data() + (size_t)idx * N_FEATURES;
            int y_true = y_local[idx];

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
                    batch_loss  += 1.f - margin;
                }
            }
        }

        // L2 regularisation gradient (always applied, independent of hinge)
        for (int p = 0; p < N_PAIRS; ++p)
            for (int f = 0; f < N_FEATURES; ++f)
                dW_local[p * N_FEATURES + f] += LAMBDA * W[p * N_FEATURES + f];

        // 3c. Aggregate gradients across all ranks → mean gradient
        MPI_CHECK(MPI_Allreduce(dW_local.data(), dW_global.data(),
                                N_PAIRS * N_FEATURES, MPI_FLOAT,
                                MPI_SUM, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Allreduce(db_local.data(), db_global.data(),
                                N_PAIRS, MPI_FLOAT,
                                MPI_SUM, MPI_COMM_WORLD));

        for (auto& v : dW_global) v /= world_size;
        for (auto& v : db_global) v /= world_size;

        // 3d. Adam update (identical on all ranks — same averaged gradient)
        ++t_adam;
        float bc1 = 1.f - std::pow(ADAM_BETA1, (float)t_adam);
        float bc2 = 1.f - std::pow(ADAM_BETA2, (float)t_adam);

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

        // 3e. Average train loss for logging
        float global_loss = 0.f;
        MPI_CHECK(MPI_Allreduce(&batch_loss, &global_loss, 1, MPI_FLOAT,
                                MPI_SUM, MPI_COMM_WORLD));
        global_loss /= (float)(world_size * BATCH_SIZE);

        // 3f. Validation every VAL_INTERVAL epochs (rank 0 only)
        float val_f1  = 0.f;
        float val_acc = 0.f;
        if (rank == 0 && (epoch + 1) % VAL_INTERVAL == 0) {
            long cm[N_CLASSES][N_CLASSES] = {};
            long correct = 0;
            for (int i = 0; i < local_n; ++i) {
                const float* x = X_local.data() + (size_t)i * N_FEATURES;
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
                int ytrue = y_local[i];
                if (ytrue >= 0 && ytrue < N_CLASSES && best_k >= 0) {
                    cm[ytrue][best_k]++;
                    if (best_k == ytrue) ++correct;
                }
            }
            val_f1  = macro_f1(cm);
            val_acc = (float)correct / local_n;

            last_val_acc = val_acc;
            last_val_f1  = val_f1;

            lprintf("[%s] epoch %3d/%d  lr=%.5f  loss=%.4f  "
                   "val_acc=%.4f  val_macro_f1=%.4f",
                   ts().c_str(), epoch + 1, EPOCHS,
                   (double)lr, (double)global_loss,
                   (double)val_acc, (double)val_f1);

            if (val_f1 > best_f1 + MIN_DELTA) {
                best_f1      = val_f1;
                best_val_acc = val_acc;
                W_best       = W;
                b_best       = b;
                best_saved   = true;
                patience_cnt = 0;
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

            // Per-class recall breakdown — shows which classes are being learned
            long rs[N_CLASSES] = {};
            for (int k = 0; k < N_CLASSES; ++k)
                for (int j = 0; j < N_CLASSES; ++j)
                    rs[k] += cm[k][j];
            lprintf("         recall: DDoS=%.3f  DoS=%.3f  Normal=%.3f  PortScan=%.3f\n",
                   rs[0] > 0 ? (float)cm[0][0]/rs[0] : 0.f,
                   rs[1] > 0 ? (float)cm[1][1]/rs[1] : 0.f,
                   rs[2] > 0 ? (float)cm[2][2]/rs[2] : 0.f,
                   rs[3] > 0 ? (float)cm[3][3]/rs[3] : 0.f);
        }

        if (rank == 0) {
            double epoch_ms = (MPI_Wtime() - t_epoch_start) * 1000.0;
            log_file << (epoch + 1) << "," << lr << "," << global_loss
                     << "," << val_acc << "," << val_f1
                     << "," << (MPI_Wtime() - t_train_start)
                     << "," << epoch_ms << "\n";
            sum_global_loss += global_loss;
            ++loss_epochs;
        }

        // Broadcast early-stopping flag from rank 0 to all ranks
        MPI_CHECK(MPI_Bcast(&stop_flag, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }

    // ══════════════════════════════════════════════════════════════════════
    // PHASE 4 — SAVE SVM MODEL  (rank 0 only)
    //
    //   model.bin layout:
    //     int32  N_PAIRS     (6)
    //     int32  N_FEATURES  (52)
    //     float  W[6*52]     (row-major: W[p*52 + f])
    //     float  b[6]
    // ══════════════════════════════════════════════════════════════════════

    if (rank == 0) {
        double t_train_end = MPI_Wtime();
        lprintf("[%s] Training time: %.1fs\n", ts().c_str(), t_train_end - t_train_start);
        log_file.close();

        std::ofstream mf(MODEL_OUT, std::ios::binary);
        if (!mf) {
            fprintf(stderr, "[rank 0] Cannot open %s for writing\n", MODEL_OUT);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int dims[2] = { N_PAIRS, N_FEATURES };
        mf.write(reinterpret_cast<const char*>(dims),        sizeof(dims));
        const std::vector<float>& W_save = best_saved ? W_best : W;
        const std::vector<float>& b_save = best_saved ? b_best : b;
        mf.write(reinterpret_cast<const char*>(W_save.data()), N_PAIRS * N_FEATURES * sizeof(float));
        mf.write(reinterpret_cast<const char*>(b_save.data()), N_PAIRS * sizeof(float));
        mf.close();

        lprintf("[%s] SVM model saved -> %s  (K=%d F=%d)  [%s weights]\n",
               ts().c_str(), MODEL_OUT, N_PAIRS, N_FEATURES,
               best_saved ? "best-checkpoint" : "final-epoch");
        lprintf("[%s] Training log    -> %s\n", ts().c_str(), LOG_OUT);

        // ── Training summary: compute GFLOPS ──────────────────────────────
        //
        //   FLOP model (dominant terms):
        //   • Gradient: world_size × BATCH_SIZE × (N_CLASSES-1) pairs/sample
        //               × 4×N_FEATURES FLOPs (score dot + gradient: 2×N_FEATURES MACs each)
        //   • L2 reg  : N_PAIRS × N_FEATURES × 2 per epoch
        //   • Val loop: every VAL_INTERVAL epochs, N_train × N_PAIRS × 2×N_FEATURES
        {
            double train_s  = t_train_end - t_train_start;
            long   grad_f   = (long)world_size * BATCH_SIZE * (N_CLASSES - 1)
                              * 4L * N_FEATURES * t_adam;
            long   l2_f     = (long)N_PAIRS * N_FEATURES * 2L * t_adam;
            int    n_val    = (t_adam + VAL_INTERVAL - 1) / VAL_INTERVAL;
            long   val_f    = (long)N_train * N_PAIRS * 2L * N_FEATURES * n_val;
            double train_gflops = (train_s > 0.0)
                                  ? (grad_f + l2_f + val_f) / (train_s * 1e9)
                                  : 0.0;
            float  loss_avg     = (loss_epochs > 0)
                                  ? sum_global_loss / loss_epochs : 0.f;
            float  acc_out  = best_saved ? best_val_acc : last_val_acc;

            // Append key,value summary rows to training_log.csv (matches omp_train format)
            {
                std::ofstream sf(LOG_OUT, std::ios::app);
                sf << "train_ms,"     << (train_s * 1000.0) << "\n";
                sf << "accuracy,"     << acc_out             << "\n";
                sf << "macro_f1,"     << best_f1             << "\n";
                sf << "train_gflops," << train_gflops        << "\n";
            }

            // train_results.csv — one-row summary
            std::ofstream rf(TRAIN_RESULTS);
            if (rf) {
                rf << "n_ranks,n_train,epochs_run,val_acc,val_f1,"
                      "best_f1,loss_avg,train_time_ms,train_gflops\n";
                rf << world_size        << ","
                   << N_train           << ","
                   << t_adam            << ","
                   << acc_out           << ","
                   << best_f1           << ","
                   << best_f1           << ","
                   << loss_avg          << ","
                   << (train_s * 1000.0) << ","
                   << train_gflops      << "\n";
                rf.close();
                lprintf("[%s] Train results    -> %s\n", ts().c_str(), TRAIN_RESULTS);
            } else {
                fprintf(stderr, "[rank 0] Warning: cannot open %s\n", TRAIN_RESULTS);
            }

            lprintf("[%s] train_ms=%.1f  accuracy=%.4f  macro_f1=%.4f  GFLOPS=%.2f\n",
                   ts().c_str(), train_s * 1000.0,
                   (double)acc_out, (double)best_f1, train_gflops);
        }

        lprintf("[%s] Done. Run mpi_infer to apply the hybrid SVM+DBSCAN pipeline.\n",
               ts().c_str());
    }

    if (rank == 0) log_close();
    MPI_CHECK(MPI_Finalize());
    return 0;
}
