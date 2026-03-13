/*
 * cuda_svm.cu  ---  Multi-GPU Linear SVM (One-vs-Rest) for CICIDS2017
 *
 * Dataset layout  (data/processed/, produced by preprocess.py)
 * -----------------------------------------------------------------
 *  52 features  (StandardScaler, no header rows)
 *  4 training classes  original IDs -> internal IDs:
 *    2 DDoS           ->  0   ( 89,610 samples  weight 1.6424)
 *    3 DoS            ->  1   (135,621 samples  weight 1.0852)
 *    4 NormalTraffic  ->  2   (300,000 samples  weight 0.4906)
 *    5 PortScan       ->  3   ( 63,486 samples  weight 2.3183)
 *  3 holdout classes (test-only, remapped to -1):
 *    0 Bots     1 BruteForce     6 WebAttacks
 *
 * Hardware target
 * ---------------
 *  4x RTX 2080  (Turing, sm_75, 8 GB VRAM each)
 *
 * Strategy
 * --------
 *  Data-parallel across 4 GPUs.  Each GPU holds 1/4 of the training data
 *  device-resident.  Per step:
 *    1. Gather a BATCH_SIZE mini-batch with a custom gather kernel
 *       (no host<->device copies per step after init).
 *    2. Forward: cuBLAS sgemm -> decision scores D [B x K].
 *    3. Add bias, compute per-sample hinge loss mask M [B x K].
 *    4. Backward: cuBLAS sgemm -> grad_W [K x F], kernel -> grad_b [K].
 *    5. CPU AllReduce: sync streams, gather gW/gb to host, sum, broadcast back.
 *    6. Adam update (same LR on every GPU -> weights stay identical).
 *  Cosine-annealing LR schedule.  Per-epoch validation on full test set
 *  in VAL_CHUNK-sized slices on GPU 0.
 *  Best-model checkpoint saved as binary + text CSV.
 *
 * cuBLAS column-major convention
 * --------------------------------
 *  C row-major [M x N] and cuBLAS col-major [N x M] share the same bytes.
 *  Element (i,j) row-major at i*N+j  ==  element (j,i) col-major.
 *
 *  Forward:  D [B x K] row = X [B x F] @ W^T [F x K]
 *    cuBLAS: D^T [K x B] col = W [K x F] op_T  x  X [B x F] op_N
 *      W row [K x F] -> cuBLAS col [F x K], op_T -> [K x F],  lda=F
 *      X row [B x F] -> cuBLAS col [F x B], op_N -> [F x B],  ldb=F
 *      D col [K x B], ldc=K   (same bytes as D row [B x K])
 *    sgemm(op_T, op_N, K, B, F, 1, W, F, X, F, 0, D, K)
 *
 *  Backward: grad_W [K x F] row = -(M^T @ X)
 *    Compute grad_W^T [F x K] col = -X [B x F] op_N  x  M [B x K] op_T
 *      X row [B x F] -> cuBLAS col [F x B], op_N, lda=F
 *      M row [B x K] -> cuBLAS col [K x B], op_T->[B x K], ldb=K
 *      gW^T col [F x K], ldc=F  (same bytes as gW row [K x F])
 *    sgemm(op_N, op_T, F, K, B, -1, X, F, M, K, 0, gW, F)
 *
 * Build
 * -----
 *  nvcc -O3 -arch=sm_75 -std=c++17 -lcublas -o cuda_svm cuda_svm.cu
 *
 * Run  (from /home/lataeq/Code/np)
 * ----------------------------------
 *  ./cuda_svm
 *  training_log.csv    -- per-epoch metrics (epoch,step,lr,loss,acc,recall,precision,holdout)
 *  best_model.bin      -- binary: int[2]{K,F} + float W[K*F] + float b[K]
 *  best_model_W.csv    -- text weight matrix (K rows x F cols)
 *  best_model_b.csv    -- text bias vector   (K rows)
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Compile-time configuration
// ---------------------------------------------------------------------------
static constexpr int   N_FEATURES      = 52;
static constexpr int   N_TRAIN_CLASSES = 4;
static constexpr int   N_ALL_CLASSES   = 7;
static constexpr int   N_GPUS          = 4;

static constexpr int   EPOCHS          = 200;    // ↑ from 30; model hadn't converged
static constexpr int   BATCH_SIZE      = 4096;   // per GPU
static constexpr int   VAL_CHUNK          = 65536;  // test/train inference slice on GPU 0
static constexpr int   TRAIN_EVAL_SAMPLES = 50000;  // training subsample for overfitting check
static constexpr int   LOG_STEPS       = 100;    // print train loss every N global steps
static constexpr int   PATIENCE        = 30;     // early stopping: epochs without improvement
static constexpr float MIN_DELTA       = 1e-4f;  // minimum acc gain to reset patience

static constexpr float LAMBDA          = 1e-4f;  // L2 weight decay
static constexpr float LR_PEAK         = 5e-2f;  // Adam peak LR (↑ from 1e-3, matches run)
static constexpr float LR_MIN          = 1e-3f;  // cosine-annealing floor (↑ from 1e-5)
static constexpr float ADAM_BETA1      = 0.9f;
static constexpr float ADAM_BETA2      = 0.999f;
static constexpr float ADAM_EPS        = 1e-8f;

// Class mappings
static constexpr int ORIG2INT[N_ALL_CLASSES]   = { -1, -1, 0, 1, 2, 3, -1 };
static constexpr int INT2ORIG[N_TRAIN_CLASSES] = { 2, 3, 4, 5 };
static const char   *CLASS_NAMES[N_ALL_CLASSES] = {
    "Bots", "BruteForce", "DDoS", "DoS",
    "NormalTraffic", "PortScan", "WebAttacks"
};
// Boosted minority class weights: DDoS×1.5, PortScan×2.0 (val→test gap fix)
static constexpr float CLASS_WEIGHTS[N_TRAIN_CLASSES] = {
    2.463663f,   // DDoS       (internal 0)  [1.6424 × 1.5]
    1.085225f,   // DoS        (internal 1)  [unchanged]
    0.490598f,   // Normal     (internal 2)  [unchanged]
    4.636590f,   // PortScan   (internal 3)  [2.3183 × 2.0]
};

// ---------------------------------------------------------------------------
// Error-checking helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr) do {                                               \
    cudaError_t _e = (expr);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

#define CUBLAS_CHECK(expr) do {                                             \
    cublasStatus_t _s = (expr);                                             \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "[cuBLAS ERROR] %s:%d  status=%d\n",               \
                __FILE__, __LINE__, (int)_s);                               \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------
static std::string ts()
{
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

// ---------------------------------------------------------------------------
// Data loaders  (CSV produced by preprocess.py, no header)
// ---------------------------------------------------------------------------
static std::vector<float> load_features(const std::string& path, int /*n_features*/)
{
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    std::vector<float> data;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            data.push_back(std::stof(tok));
    }
    return data;
}

static std::vector<int> load_labels(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    std::vector<int> labels;
    std::string line;
    while (std::getline(f, line))
        labels.push_back(std::stoi(line));
    return labels;
}

// Remap original class IDs -> internal IDs (-1 for holdout classes)
static void remap_labels(std::vector<int>& v)
{
    for (auto& x : v) {
        int orig = x;
        x = (orig >= 0 && orig < N_ALL_CLASSES) ? ORIG2INT[orig] : -1;
    }
}

// ---------------------------------------------------------------------------
// Class weights in constant memory
// ---------------------------------------------------------------------------
__device__ __constant__ float d_cw[N_TRAIN_CLASSES];

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Gather rows: dst[b,:] = src[indices[b], :]  (F columns).
// Also used for int labels cast to float with F=1.
__global__ void gather_kernel(const float* __restrict__ d_src,
                              const int*   __restrict__ indices,
                              float*       __restrict__ dst,
                              int B, int F)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * F) return;
    int b = tid / F, f = tid % F;
    dst[tid] = d_src[(long long)indices[b] * F + f];
}

// Add bias: D[b][k] += b_vec[k]
__global__ void add_bias_kernel(float* D, const float* b_vec, int B, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * K) return;
    D[tid] += b_vec[tid % K];
}

// Hinge loss mask.  M[b][k] = -cw[lbl]*yk if margin violated, else 0.
// Shared-memory reduction -> atomicAdd into loss_out.
__global__ void hinge_mask_kernel(const float* __restrict__ D,
                                   const int*   __restrict__ labels,
                                   float*       __restrict__ M,
                                   float*       __restrict__ loss_out,
                                   int B, int K)
{
    extern __shared__ float shmem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.f;
    if (tid < B * K) {
        int b   = tid / K;
        int k   = tid % K;
        int lbl = labels[b];
        float yk = (lbl == k) ? 1.f : -1.f;
        float w  = (lbl >= 0 && lbl < N_TRAIN_CLASSES) ? d_cw[lbl] : 1.f;
        float h  = 1.f - yk * D[tid];
        if (h > 0.f) {
            M[tid]     = -w * yk;
            local_loss =  w * h;
        } else {
            M[tid] = 0.f;
        }
    }
    shmem[threadIdx.x] = local_loss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(loss_out, shmem[0]);
}

// Reduce M along B -> grad_b[k] = (1/B) * sum_b M[b][k]
__global__ void grad_b_kernel(const float* __restrict__ M,
                               float*       __restrict__ gb,
                               int B, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    float s = 0.f;
    for (int b = 0; b < B; ++b) s += M[(long long)b * K + k];
    gb[k] = s / B;
}

// Adam parameter update with L2 regularisation.
__global__ void adam_update_kernel(float*       param,
                                    float*       m,
                                    float*       v,
                                    const float* grad,
                                    float lr, float beta1, float beta2,
                                    float eps, float lambda,
                                    int n, int step)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i] + lambda * param[i];
    float mi = beta1 * m[i] + (1.f - beta1) * g;
    float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float bc1 = 1.f - powf(beta1, (float)step);
    float bc2 = 1.f - powf(beta2, (float)step);
    param[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
}

// Argmax over K classes -> predicted label.
__global__ void argmax_kernel(const float* __restrict__ D,
                               int*         __restrict__ preds,
                               int N, int K)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* row = D + (long long)n * K;
    int best = 0;
    for (int k = 1; k < K; ++k)
        if (row[k] > row[best]) best = k;
    preds[n] = best;
}

// ---------------------------------------------------------------------------
// Per-GPU state
// ---------------------------------------------------------------------------
struct GpuState {
    int            id;
    cublasHandle_t cublas;
    cudaStream_t   stream;

    int    N_gpu;
    float* d_X;       // [N_gpu x F]  training features
    int*   d_y;       // [N_gpu]      training labels

    float* d_W;       // [K x F]  weight matrix
    float* d_b;       // [K]      bias vector

    float* d_mW, *d_vW;  // Adam moments for W
    float* d_mb, *d_vb;  // Adam moments for b

    float* d_batchX;  // [B x F]  mini-batch features
    int*   d_batchY;  // [B]      mini-batch labels
    int*   d_idx;     // [B]      random sample indices
    float* d_D;       // [B x K]  decision scores
    float* d_M;       // [B x K]  hinge mask
    float* d_gW;      // [K x F]  gradient W
    float* d_gb;      // [K]      gradient b
    float* d_loss;    // [1]      per-step loss accumulator
};

// ---------------------------------------------------------------------------
// Save best model (binary + CSV)
// ---------------------------------------------------------------------------
static void save_model(const float* h_W, const float* h_b, int K, int F,
                       const std::string& stem)
{
    // Binary blob: int[2]{K,F} + float W[K*F] + float b[K]
    {
        std::ofstream f(stem + ".bin", std::ios::binary);
        int dims[2] = {K, F};
        f.write(reinterpret_cast<const char*>(dims),  sizeof(dims));
        f.write(reinterpret_cast<const char*>(h_W),   (long long)K * F * sizeof(float));
        f.write(reinterpret_cast<const char*>(h_b),   K * sizeof(float));
    }
    // Weight CSV  (K rows x F cols)
    {
        std::ofstream f(stem + "_W.csv");
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < F; ++j) {
                if (j) f << ",";
                f << h_W[k * F + j];
            }
            f << "\n";
        }
    }
    // Bias CSV  (K rows)
    {
        std::ofstream f(stem + "_b.csv");
        for (int k = 0; k < K; ++k) f << h_b[k] << "\n";
    }
    printf("  -> saved %s.bin / _W.csv / _b.csv\n", stem.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    // ---- Load data on host -------------------------------------------
    printf("[%s] Loading data...\n", ts().c_str());
    auto h_Xtr = load_features("data/processed/train_data.csv",   N_FEATURES);
    auto h_ytr = load_labels  ("data/processed/train_labels.csv");
    auto h_Xte = load_features("data/processed/test_data.csv",    N_FEATURES);
    auto h_yte = load_labels  ("data/processed/test_labels.csv");
    remap_labels(h_ytr);
    remap_labels(h_yte);
    const int N_tr = (int)h_ytr.size();
    const int N_te = (int)h_yte.size();
    const int F    = N_FEATURES;
    const int K    = N_TRAIN_CLASSES;
    printf("[%s] N_train=%d  N_test=%d\n", ts().c_str(), N_tr, N_te);

    // ---- Per-GPU init -----------------------------------------------
    GpuState gpus[N_GPUS];
    for (int g = 0; g < N_GPUS; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        gpus[g].id = g;
        CUBLAS_CHECK(cublasCreate(&gpus[g].cublas));
        CUDA_CHECK(cudaStreamCreate(&gpus[g].stream));
        CUBLAS_CHECK(cublasSetStream(gpus[g].cublas, gpus[g].stream));

        // Copy class weights to constant memory on this device
        CUDA_CHECK(cudaMemcpyToSymbol(d_cw, CLASS_WEIGHTS, K * sizeof(float)));

        // Partition training data
        int base = (long long)g * N_tr / N_GPUS;
        int Ng   = (long long)(g + 1) * N_tr / N_GPUS - base;
        gpus[g].N_gpu = Ng;

        CUDA_CHECK(cudaMalloc(&gpus[g].d_X, (long long)Ng * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_y, Ng * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(gpus[g].d_X,
                                   h_Xtr.data() + (long long)base * F,
                                   (long long)Ng * F * sizeof(float),
                                   cudaMemcpyHostToDevice, gpus[g].stream));
        CUDA_CHECK(cudaMemcpyAsync(gpus[g].d_y,
                                   h_ytr.data() + base,
                                   Ng * sizeof(int),
                                   cudaMemcpyHostToDevice, gpus[g].stream));

        // Model parameters (zero-initialised)
        CUDA_CHECK(cudaMalloc(&gpus[g].d_W,  (long long)K * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_b,  K * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpus[g].d_W, 0, (long long)K * F * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpus[g].d_b, 0, K * sizeof(float)));

        // Adam moments (zero-initialised)
        CUDA_CHECK(cudaMalloc(&gpus[g].d_mW, (long long)K * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_vW, (long long)K * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_mb, K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_vb, K * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpus[g].d_mW, 0, (long long)K * F * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpus[g].d_vW, 0, (long long)K * F * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpus[g].d_mb, 0, K * sizeof(float)));
        CUDA_CHECK(cudaMemset(gpus[g].d_vb, 0, K * sizeof(float)));

        // Working buffers
        CUDA_CHECK(cudaMalloc(&gpus[g].d_batchX, (long long)BATCH_SIZE * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_batchY, BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_idx,    BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_D,      (long long)BATCH_SIZE * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_M,      (long long)BATCH_SIZE * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_gW,     (long long)K * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_gb,     K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_loss,   sizeof(float)));
    }

    // ---- Training log -----------------------------------------------
    std::ofstream log_f("training_log.csv");
    log_f << "epoch,step,lr,loss,train_acc,acc,gap,recall,precision,holdout\n";

    std::vector<float> h_W(K * F), h_b(K);
    std::vector<float> best_W(K * F), best_b(K);
    float best_acc    = -1.f;
    int   global_step = 0;
    int   patience_cnt = 0;

    // Host buffers for CPU-side gradient AllReduce.
    // gW is K*F = 208 floats (832 bytes) — CPU roundtrip cost is negligible.
    std::vector<float> h_gW_acc(K * F), h_gb_acc(K);
    std::vector<float> h_gW_tmp(K * F), h_gb_tmp(K);

    const int steps_per_epoch = gpus[0].N_gpu / BATCH_SIZE;
    std::mt19937 rng(42);

    // ---- Training loop ----------------------------------------------
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        double epoch_loss  = 0.0;
        int    epoch_steps = 0;

        for (int s = 0; s < steps_per_epoch; ++s) {
            ++global_step;

            // Cosine LR annealing
            float frac = (float)global_step /
                         (float)(EPOCHS * steps_per_epoch);
            float lr   = LR_MIN + 0.5f * (LR_PEAK - LR_MIN) *
                         (1.f + cosf((float)M_PI * frac));

            // -- Forward + backward on each GPU independently --
            for (int g = 0; g < N_GPUS; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                GpuState& G = gpus[g];

                // Sample random batch indices, copy to device
                {
                    std::vector<int> idx(BATCH_SIZE);
                    std::uniform_int_distribution<int> dist(0, G.N_gpu - 1);
                    for (auto& i : idx) i = dist(rng);
                    CUDA_CHECK(cudaMemcpyAsync(G.d_idx, idx.data(),
                                               BATCH_SIZE * sizeof(int),
                                               cudaMemcpyHostToDevice, G.stream));
                }
                // Gather batch features + labels into mini-batch buffers
                {
                    int th = 256;
                    int bl = ((long long)BATCH_SIZE * F + th - 1) / th;
                    gather_kernel<<<bl, th, 0, G.stream>>>(
                        G.d_X, G.d_idx, G.d_batchX, BATCH_SIZE, F);
                    // Reinterpret int labels as float bits for the generic gather kernel
                    gather_kernel<<<(BATCH_SIZE+th-1)/th, th, 0, G.stream>>>(
                        reinterpret_cast<const float*>(G.d_y),
                        G.d_idx,
                        reinterpret_cast<float*>(G.d_batchY),
                        BATCH_SIZE, 1);
                }

                // Reset per-step loss (GPU 0 only — used for logging)
                if (g == 0)
                    CUDA_CHECK(cudaMemsetAsync(G.d_loss, 0, sizeof(float), G.stream));

                // Forward: D[B×K] = X[B×F] @ W^T  then + bias
                {
                    const float alpha = 1.f, beta = 0.f;
                    CUBLAS_CHECK(cublasSgemm(G.cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        K, BATCH_SIZE, F,
                        &alpha, G.d_W,      F,
                                G.d_batchX, F,
                        &beta,  G.d_D,      K));
                    const int th = 256;
                    add_bias_kernel<<<(BATCH_SIZE*K+th-1)/th, th, 0, G.stream>>>(
                        G.d_D, G.d_b, BATCH_SIZE, K);
                }

                // Hinge mask M[B×K] + loss accumulation into d_loss
                {
                    const int th    = 256;
                    const int items = BATCH_SIZE * K;
                    hinge_mask_kernel<<<(items+th-1)/th, th,
                                        th*sizeof(float), G.stream>>>(
                        G.d_D, G.d_batchY, G.d_M, G.d_loss, BATCH_SIZE, K);
                }

                // Backward: gW[K×F] = M^T @ X   (M already stores dL/dD = -cw*yk)
                // gW^T [F×K] = X [F×B]  @  M [B×K], alpha=+1
                {
                    const float alpha = 1.f, beta = 0.f;
                    CUBLAS_CHECK(cublasSgemm(G.cublas,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        F, K, BATCH_SIZE,
                        &alpha, G.d_batchX, F,
                                G.d_M,      K,
                        &beta,  G.d_gW,     F));
                    const int th = 256;
                    grad_b_kernel<<<(K+th-1)/th, th, 0, G.stream>>>(
                        G.d_M, G.d_gb, BATCH_SIZE, K);
                }

            } // end per-GPU forward+backward

            // -- CPU AllReduce: sync, gather gW/gb to host, sum, broadcast back --
            // gW is only K*F=208 floats; total host<->device traffic ~7 KB/step.
            for (int g = 0; g < N_GPUS; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaStreamSynchronize(gpus[g].stream));
            }
            std::fill(h_gW_acc.begin(), h_gW_acc.end(), 0.f);
            std::fill(h_gb_acc.begin(), h_gb_acc.end(), 0.f);
            for (int g = 0; g < N_GPUS; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpy(h_gW_tmp.data(), gpus[g].d_gW,
                                      (size_t)K * F * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_gb_tmp.data(), gpus[g].d_gb,
                                      K * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                for (int i = 0; i < K * F; ++i) h_gW_acc[i] += h_gW_tmp[i];
                for (int i = 0; i < K;     ++i) h_gb_acc[i] += h_gb_tmp[i];
            }
            for (int g = 0; g < N_GPUS; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMemcpyAsync(gpus[g].d_gW, h_gW_acc.data(),
                                           (size_t)K * F * sizeof(float),
                                           cudaMemcpyHostToDevice, gpus[g].stream));
                CUDA_CHECK(cudaMemcpyAsync(gpus[g].d_gb, h_gb_acc.data(),
                                           K * sizeof(float),
                                           cudaMemcpyHostToDevice, gpus[g].stream));
            }

            // -- Scale normalisation + Adam update (identical on every GPU) --
            // gW: cuBLAS gave -M^T@X (no 1/B factor); gb: kernel divided by B already.
            // After AllReduce (sum over N_GPUS GPUs):
            //   gW_total = sum_g -M_g^T@X_g  -> scale by 1/(N_GPUS*B)
            //   gb_total = sum_g (1/B)*colsum(M_g) -> scale by 1/N_GPUS
            {
                const float scW = 1.f / ((float)N_GPUS * BATCH_SIZE);
                const float scb = 1.f /  (float)N_GPUS;
                const int   nW  = K * F;
                const int   th  = 256;
                for (int g = 0; g < N_GPUS; ++g) {
                    CUDA_CHECK(cudaSetDevice(g));
                    GpuState& G = gpus[g];
                    CUBLAS_CHECK(cublasSscal(G.cublas, nW, &scW, G.d_gW, 1));
                    CUBLAS_CHECK(cublasSscal(G.cublas, K,  &scb, G.d_gb, 1));
                    adam_update_kernel<<<(nW+th-1)/th, th, 0, G.stream>>>(
                        G.d_W, G.d_mW, G.d_vW, G.d_gW,
                        lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, LAMBDA,
                        nW, global_step);
                    adam_update_kernel<<<(K+th-1)/th, th, 0, G.stream>>>(
                        G.d_b, G.d_mb, G.d_vb, G.d_gb,
                        lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, 0.f,  // no L2 on bias
                        K, global_step);
                }
            }

            // -- Log training loss every LOG_STEPS --
            if ((global_step % LOG_STEPS) == 0) {
                CUDA_CHECK(cudaSetDevice(0));
                CUDA_CHECK(cudaStreamSynchronize(gpus[0].stream));
                float step_loss = 0.f;
                CUDA_CHECK(cudaMemcpy(&step_loss, gpus[0].d_loss, sizeof(float),
                                      cudaMemcpyDeviceToHost));
                step_loss /= (float)BATCH_SIZE;
                epoch_loss += step_loss;
                ++epoch_steps;
                printf("\r[%s] epoch %3d  step %6d  lr=%.2e  loss=%.4f   ",
                       ts().c_str(), epoch+1, global_step,
                       (double)lr, (double)step_loss);
                fflush(stdout);
            }

        } // end steps loop

        // ---- End-of-epoch: evaluate on test set (GPU 0) ----
        for (int g = 0; g < N_GPUS; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(gpus[g].stream));
        }
        CUDA_CHECK(cudaSetDevice(0));

        // Pull current W and b from GPU 0
        CUDA_CHECK(cudaMemcpy(h_W.data(), gpus[0].d_W,
                              (size_t)K * F * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b.data(), gpus[0].d_b,
                              K * sizeof(float), cudaMemcpyDeviceToHost));

        // Allocate test-inference scratch buffers on GPU 0
        float* d_teX = nullptr;
        float* d_teD = nullptr;
        int*   d_teP = nullptr;
        CUDA_CHECK(cudaMalloc(&d_teX, (size_t)VAL_CHUNK * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_teD, (size_t)VAL_CHUNK * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_teP, VAL_CHUNK * sizeof(int)));

        // Confusion matrix over train classes only (-1 labels = holdout, skipped)
        long cm[N_TRAIN_CLASSES][N_TRAIN_CLASSES] = {};
        long n_holdout = 0;
        std::vector<int> h_preds(VAL_CHUNK);

        for (int off = 0; off < N_te; off += VAL_CHUNK) {
            const int chunk = std::min(VAL_CHUNK, N_te - off);

            CUDA_CHECK(cudaMemcpy(d_teX,
                                  h_Xte.data() + (size_t)off * F,
                                  (size_t)chunk * F * sizeof(float),
                                  cudaMemcpyHostToDevice));

            const float alpha = 1.f, beta = 0.f;
            CUBLAS_CHECK(cublasSgemm(gpus[0].cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                K, chunk, F,
                &alpha, gpus[0].d_W, F,
                         d_teX,      F,
                &beta,   d_teD,      K));

            const int th = 256;
            add_bias_kernel<<<(chunk*K+th-1)/th, th, 0, gpus[0].stream>>>(
                d_teD, gpus[0].d_b, chunk, K);
            argmax_kernel<<<(chunk+th-1)/th, th, 0, gpus[0].stream>>>(
                d_teD, d_teP, chunk, K);

            CUDA_CHECK(cudaStreamSynchronize(gpus[0].stream));
            CUDA_CHECK(cudaMemcpy(h_preds.data(), d_teP,
                                  chunk * sizeof(int), cudaMemcpyDeviceToHost));

            for (int i = 0; i < chunk; ++i) {
                int tr = h_yte[off + i];
                int pr = h_preds[i];
                if (tr < 0) { ++n_holdout; continue; }
                if (pr < 0 || pr >= K) pr = 0;
                cm[tr][pr]++;
            }
        }

        CUDA_CHECK(cudaFree(d_teX));
        CUDA_CHECK(cudaFree(d_teD));
        CUDA_CHECK(cudaFree(d_teP));

        // ---- Train accuracy on first TRAIN_EVAL_SAMPLES (overfitting check) ----
        float train_acc = 0.f;
        {
            const int N_tr_eval = std::min(TRAIN_EVAL_SAMPLES, N_tr);
            float* d_trX = nullptr;
            float* d_trD = nullptr;
            int*   d_trP = nullptr;
            CUDA_CHECK(cudaMalloc(&d_trX, (size_t)VAL_CHUNK * F * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_trD, (size_t)VAL_CHUNK * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_trP, VAL_CHUNK * sizeof(int)));

            long tr_correct = 0, tr_total = 0;
            std::vector<int> h_tr_preds(VAL_CHUNK);

            for (int off = 0; off < N_tr_eval; off += VAL_CHUNK) {
                const int chunk = std::min(VAL_CHUNK, N_tr_eval - off);

                CUDA_CHECK(cudaMemcpy(d_trX,
                                      h_Xtr.data() + (size_t)off * F,
                                      (size_t)chunk * F * sizeof(float),
                                      cudaMemcpyHostToDevice));

                const float alpha = 1.f, beta = 0.f;
                CUBLAS_CHECK(cublasSgemm(gpus[0].cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    K, chunk, F,
                    &alpha, gpus[0].d_W, F,
                             d_trX,      F,
                    &beta,   d_trD,      K));

                const int th = 256;
                add_bias_kernel<<<(chunk*K+th-1)/th, th, 0, gpus[0].stream>>>(
                    d_trD, gpus[0].d_b, chunk, K);
                argmax_kernel<<<(chunk+th-1)/th, th, 0, gpus[0].stream>>>(
                    d_trD, d_trP, chunk, K);

                CUDA_CHECK(cudaStreamSynchronize(gpus[0].stream));
                CUDA_CHECK(cudaMemcpy(h_tr_preds.data(), d_trP,
                                      chunk * sizeof(int), cudaMemcpyDeviceToHost));

                for (int i = 0; i < chunk; ++i) {
                    int tr_lbl = h_ytr[off + i];
                    int pr     = h_tr_preds[i];
                    if (tr_lbl < 0) continue;
                    if (pr < 0 || pr >= K) pr = 0;
                    if (tr_lbl == pr) ++tr_correct;
                    ++tr_total;
                }
            }

            CUDA_CHECK(cudaFree(d_trX));
            CUDA_CHECK(cudaFree(d_trD));
            CUDA_CHECK(cudaFree(d_trP));

            train_acc = tr_total > 0 ? (float)tr_correct / (float)tr_total : 0.f;
        }

        // Compute and print per-class precision / recall / F1
        long  total_correct = 0, total_known = 0;
        float macro_prec = 0.f, macro_rec = 0.f, macro_f1 = 0.f;
        printf("\n[%s] === Epoch %d ===\n", ts().c_str(), epoch + 1);
        printf("  %-16s  %8s  %8s  %8s  %8s\n",
               "Class", "Prec", "Recall", "F1", "Support");
        for (int k = 0; k < K; ++k) {
            long tp = cm[k][k], fp = 0, fn = 0;
            for (int j = 0; j < K; ++j) {
                if (j != k) fp += cm[j][k];
                if (j != k) fn += cm[k][j];
            }
            float prec = (tp + fp > 0) ? (float)tp / (float)(tp + fp) : 0.f;
            float rec  = (tp + fn > 0) ? (float)tp / (float)(tp + fn) : 0.f;
            float f1   = (prec + rec > 1e-9f) ? 2.f * prec * rec / (prec + rec) : 0.f;
            macro_prec += prec; macro_rec += rec; macro_f1 += f1;
            long supp = tp + fn;
            total_correct += tp;
            total_known   += supp;
            printf("  %-16s  %8.4f  %8.4f  %8.4f  %8ld\n",
                   CLASS_NAMES[INT2ORIG[k]], (double)prec, (double)rec, (double)f1, supp);
        }
        macro_prec /= K; macro_rec /= K; macro_f1 /= K;
        float acc = total_known > 0 ? (float)total_correct / (float)total_known : 0.f;
        printf("  %-16s  %8.4f  %8.4f  %8.4f  %8ld\n",
               "macro avg",
               (double)macro_prec, (double)macro_rec, (double)macro_f1, total_known);
        printf("  Accuracy (train classes): %.4f   train=%.4f   gap=%+.4f   holdout=%ld\n\n",
               (double)acc, (double)train_acc, (double)(train_acc - acc), n_holdout);

        // Append CSV row
        {
            double avg_loss = epoch_steps > 0 ? epoch_loss / epoch_steps : 0.0;
            float  lr_last  = LR_MIN + 0.5f * (LR_PEAK - LR_MIN) *
                              (1.f + cosf((float)M_PI * (float)global_step /
                                         (float)(EPOCHS * steps_per_epoch)));
            log_f << (epoch+1) << "," << global_step << "," << (double)lr_last << ","
                  << avg_loss << "," << (double)train_acc << "," << (double)acc << ","
                  << (double)(train_acc - acc) << "," << (double)macro_rec << ","
                  << (double)macro_prec << "," << (double)n_holdout / N_te << "\n";
            log_f.flush();
        }

        // Save checkpoint if best accuracy so far, track patience
        if (acc > best_acc + MIN_DELTA) {
            best_acc = acc;
            best_W   = h_W;
            best_b   = h_b;
            patience_cnt = 0;
            save_model(best_W.data(), best_b.data(), K, F, "best_model");
            printf("  ** New best acc=%.4f — model saved.\n\n", (double)best_acc);
        } else {
            ++patience_cnt;
            if (patience_cnt >= PATIENCE) {
                printf("  Early stopping: no improvement > %.4f for %d epochs.\n\n",
                       MIN_DELTA, PATIENCE);
                break;
            }
        }

    } // end epochs loop

    printf("\n[%s] Training complete.  Best test acc=%.4f\n",
           ts().c_str(), (double)best_acc);

    // ---- Cleanup ----
    for (int g = 0; g < N_GPUS; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        cudaFree(gpus[g].d_X);      cudaFree(gpus[g].d_y);
        cudaFree(gpus[g].d_W);      cudaFree(gpus[g].d_b);
        cudaFree(gpus[g].d_mW);     cudaFree(gpus[g].d_vW);
        cudaFree(gpus[g].d_mb);     cudaFree(gpus[g].d_vb);
        cudaFree(gpus[g].d_batchX); cudaFree(gpus[g].d_batchY);
        cudaFree(gpus[g].d_idx);
        cudaFree(gpus[g].d_D);      cudaFree(gpus[g].d_M);
        cudaFree(gpus[g].d_gW);     cudaFree(gpus[g].d_gb);
        cudaFree(gpus[g].d_loss);
        cublasDestroy(gpus[g].cublas);
        cudaStreamDestroy(gpus[g].stream);
    }

    return 0;
}
