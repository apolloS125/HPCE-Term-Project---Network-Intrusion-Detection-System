/*
 * cuda_rff_svm.cu  ---  Multi-GPU RBF Kernel SVM via Random Fourier Features
 *
 * Approximates an RBF (Gaussian) kernel SVM by projecting the 52 raw input
 * features through N_RFF=2048 Random Fourier Features, then training a linear
 * One-vs-Rest SVM on the high-dimensional transformed space.
 *
 * RFF mapping:  z(x) = sqrt(2/D) * cos(Ω x + b)
 *   Ω ∈ R^{D×F}  each entry iid N(0, 2·RFF_GAMMA)
 *   b ∈ R^D       each entry iid U(0, 2π)
 *   kernel approximated: k(x,z) ≈ exp(-RFF_GAMMA · ||x-z||²)
 *
 * vs cuda_svm.cu:
 *   • adds rff_transform_kernel + Omega/phi per GPU
 *   • pre-transforms all training data at init (raw 52-dim → 2048-dim)
 *   • eval transforms chunks on-the-fly (raw → RFF → matmul)
 *   • outputs: training_rff_log.csv, best_rff_model.bin/.csv
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -std=c++17 -lcublas -o cuda_rff_svm cuda_rff_svm.cu
 *
 * Hardware target: 4× RTX 2080 (sm_75, 8 GB each)
 *   Memory per GPU: ~1.2 GB raw training partition + ~35 MB working buffers
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
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

// RFF parameters
static constexpr int   N_RFF           = 4096;    // feature dim after projection
static constexpr float RFF_GAMMA       = 0.05f;   // RBF bandwidth γ: k(x,z)=exp(-γ||x-z||²)

static constexpr int   EPOCHS          = 200;
static constexpr int   BATCH_SIZE      = 4096;    // per GPU
static constexpr int   VAL_CHUNK       = 32768;   // eval chunk size on GPU 0
static constexpr int   TRAIN_EVAL_SAMPLES = 50000;
static constexpr int   LOG_STEPS       = 100;
static constexpr int   PATIENCE        = 30;
static constexpr float MIN_DELTA       = 1e-4f;

static constexpr float LAMBDA          = 1e-4f;
static constexpr float LR_PEAK         = 5e-2f;
static constexpr float LR_MIN          = 1e-3f;
static constexpr float ADAM_BETA1      = 0.9f;
static constexpr float ADAM_BETA2      = 0.999f;
static constexpr float ADAM_EPS        = 1e-8f;

// Class mappings (same as cuda_svm)
static constexpr int ORIG2INT[N_ALL_CLASSES]   = { -1, -1, 0, 1, 2, 3, -1 };
static constexpr int INT2ORIG[N_TRAIN_CLASSES] = { 2, 3, 4, 5 };
static const char   *CLASS_NAMES[N_ALL_CLASSES] = {
    "Bots", "BruteForce", "DDoS", "DoS",
    "NormalTraffic", "PortScan", "WebAttacks"
};
static constexpr float CLASS_WEIGHTS[N_TRAIN_CLASSES] = {
    2.463663f,   // DDoS
    1.800000f,   // DoS  ↑ from 1.085 — recover precision
    0.490598f,   // Normal
    4.636590f,   // PortScan
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
// Data loaders  (CSV, no header)
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

// RFF transform: Z[n, d] = scale * cos(Omega[d,:] · X[n,:] + phi[d])
// N:    number of samples
// IN_F: raw input feature dimension (52)
// D:    RFF output dimension (N_RFF = 2048)
// scale = sqrt(2 / D)
__global__ void rff_transform_kernel(
    const float* __restrict__ X,       // N × IN_F  (row-major)
    const float* __restrict__ Omega,   // D × IN_F  (row-major)
    const float* __restrict__ phi,     // D
    float*       __restrict__ Z,       // N × D     (row-major)
    int N, int IN_F, int D, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * D) return;
    int n = idx / D;
    int d = idx % D;
    float dot = 0.f;
    const float* x  = X     + (long long)n * IN_F;
    const float* om = Omega + (long long)d * IN_F;
    for (int f = 0; f < IN_F; ++f) dot += om[f] * x[f];
    Z[idx] = scale * cosf(dot + phi[d]);
}

// Gather rows: dst[b,:] = src[indices[b], :]
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

// Hinge loss mask + loss accumulation
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

// grad_b[k] = (1/B) * sum_b M[b][k]
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

// Adam update with L2 regularisation
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

// Argmax over K classes
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
    float* d_X;       // [N_gpu × N_RFF]       RFF-transformed training features
    int*   d_y;       // [N_gpu]                training labels

    float* d_Omega;   // [N_RFF × N_FEATURES]  RFF projection matrix
    float* d_phi;     // [N_RFF]               RFF phases

    float* d_W;       // [K × N_RFF]   weight matrix
    float* d_b;       // [K]           bias

    float* d_mW, *d_vW;  // Adam moments for W
    float* d_mb, *d_vb;  // Adam moments for b

    float* d_batchX;  // [BATCH_SIZE × N_RFF]
    int*   d_batchY;  // [BATCH_SIZE]
    int*   d_idx;     // [BATCH_SIZE]
    float* d_D;       // [BATCH_SIZE × K]
    float* d_M;       // [BATCH_SIZE × K]
    float* d_gW;      // [K × N_RFF]
    float* d_gb;      // [K]
    float* d_loss;    // [1]
};

// ---------------------------------------------------------------------------
// Save best model (binary + CSV)
// ---------------------------------------------------------------------------
static void save_model(const float* h_W, const float* h_b, int K, int F,
                       const std::string& stem)
{
    {
        std::ofstream f(stem + ".bin", std::ios::binary);
        int dims[2] = {K, F};
        f.write(reinterpret_cast<const char*>(dims), sizeof(dims));
        f.write(reinterpret_cast<const char*>(h_W),  (long long)K * F * sizeof(float));
        f.write(reinterpret_cast<const char*>(h_b),  K * sizeof(float));
    }
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
    {
        std::ofstream f(stem + "_b.csv");
        for (int k = 0; k < K; ++k) f << h_b[k] << "\n";
    }
    printf("  -> saved %s.bin / _W.csv / _b.csv\n", stem.c_str());
}

// ---------------------------------------------------------------------------
// Save RFF projection matrices (Omega, phi) so inference tools can reload them.
//   best_rff_Omega.bin : int[2]{D,IN_F}  + float Omega[D × IN_F]
//   best_rff_phi.bin   : int D            + float phi[D]
// Format matches what cuda_dbscan.cu / cuda_hybrid.cu expect to read.
// ---------------------------------------------------------------------------
static void save_rff_params(const std::vector<float>& h_Omega,
                             const std::vector<float>& h_phi,
                             int D, int IN_F)
{
    {
        std::ofstream f("best_rff_Omega.bin", std::ios::binary);
        int dims[2] = {D, IN_F};
        f.write(reinterpret_cast<const char*>(dims),          sizeof(dims));
        f.write(reinterpret_cast<const char*>(h_Omega.data()), (long long)D * IN_F * sizeof(float));
    }
    {
        std::ofstream f("best_rff_phi.bin", std::ios::binary);
        f.write(reinterpret_cast<const char*>(&D),           sizeof(int));
        f.write(reinterpret_cast<const char*>(h_phi.data()), D * sizeof(float));
    }
    printf("  -> saved best_rff_Omega.bin  best_rff_phi.bin  (D=%d IN_F=%d)\n", D, IN_F);
}

// ---------------------------------------------------------------------------
// Eval helper: transform a raw chunk (host) -> RFF -> scores -> predictions
// Returns {correct, total} for accuracy, fills cm confusion matrix slice.
// ---------------------------------------------------------------------------
static void eval_chunk(
    const float* h_Xchunk,   // host: chunk × IN_F
    const int*   h_ychunk,   // host: chunk labels
    int chunk, int IN_F, int F, int K,
    float* d_Xraw, float* d_Xrff, float* d_D, int* d_P,
    GpuState& gpu0,
    long cm[N_TRAIN_CLASSES][N_TRAIN_CLASSES],
    long& n_skip,
    std::vector<int>& h_preds,
    bool is_test)             // true=update cm, false=only count correct
{
    CUDA_CHECK(cudaMemcpy(d_Xraw,
                          h_Xchunk,
                          (size_t)chunk * IN_F * sizeof(float),
                          cudaMemcpyHostToDevice));

    // RFF transform
    {
        long long total = (long long)chunk * F;
        int th = 256;
        int bl = (int)((total + th - 1) / th);
        rff_transform_kernel<<<bl, th, 0, gpu0.stream>>>(
            d_Xraw, gpu0.d_Omega, gpu0.d_phi,
            d_Xrff,
            chunk, IN_F, F, sqrtf(2.f / F));
    }

    // Forward: scores = W^T @ X_rff + b
    const float alpha = 1.f, beta = 0.f;
    CUBLAS_CHECK(cublasSgemm(gpu0.cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        K, chunk, F,
        &alpha, gpu0.d_W, F,
                 d_Xrff,  F,
        &beta,   d_D,     K));
    {
        const int th = 256;
        add_bias_kernel<<<(chunk*K+th-1)/th, th, 0, gpu0.stream>>>(
            d_D, gpu0.d_b, chunk, K);
        argmax_kernel<<<(chunk+th-1)/th, th, 0, gpu0.stream>>>(
            d_D, d_P, chunk, K);
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu0.stream));
    CUDA_CHECK(cudaMemcpy(h_preds.data(), d_P,
                          chunk * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < chunk; ++i) {
        int tr = h_ychunk[i];
        int pr = h_preds[i];
        if (tr < 0) { ++n_skip; continue; }
        if (pr < 0 || pr >= K) pr = 0;
        if (is_test)
            cm[tr][pr]++;
        else
            if (tr == pr) ++n_skip; // reuse n_skip as correct counter for train
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    const int IN_F = N_FEATURES;  // raw input dimension
    const int F    = N_RFF;       // model feature dimension (post-RFF)
    const int K    = N_TRAIN_CLASSES;

    // ---- Load data -------------------------------------------------------
    printf("[%s] Loading data...\n", ts().c_str());
    auto h_Xtr = load_features("data/processed/train_data.csv",  IN_F);
    auto h_ytr = load_labels  ("data/processed/train_labels.csv");
    auto h_Xte = load_features("data/processed/test_data.csv",   IN_F);
    auto h_yte = load_labels  ("data/processed/test_labels.csv");
    remap_labels(h_ytr);
    remap_labels(h_yte);
    const int N_tr = (int)h_ytr.size();
    const int N_te = (int)h_yte.size();
    printf("[%s] N_train=%d  N_test=%d  IN_F=%d  N_RFF=%d  gamma=%.4f\n",
           ts().c_str(), N_tr, N_te, IN_F, F, (double)RFF_GAMMA);

    // ---- Generate RFF parameters on host ---------------------------------
    printf("[%s] Sampling RFF parameters...\n", ts().c_str());
    std::mt19937 rng_init(12345);
    // Spectral density for k(x,z) = exp(-gamma*||x-z||^2): omega ~ N(0, 2*gamma)
    std::normal_distribution<float>       norm_d(0.f, sqrtf(2.f * RFF_GAMMA));
    std::uniform_real_distribution<float> unif_d(0.f, 2.f * (float)M_PI);
    std::vector<float> h_Omega((long long)F * IN_F), h_phi(F);
    for (auto& v : h_Omega) v = norm_d(rng_init);
    for (auto& v : h_phi)   v = unif_d(rng_init);

    // ---- Per-GPU init + pre-transform training data ----------------------
    printf("[%s] Initialising GPUs and computing RFF training features...\n", ts().c_str());
    GpuState gpus[N_GPUS];
    for (int g = 0; g < N_GPUS; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        gpus[g].id = g;
        CUBLAS_CHECK(cublasCreate(&gpus[g].cublas));
        CUDA_CHECK(cudaStreamCreate(&gpus[g].stream));
        CUBLAS_CHECK(cublasSetStream(gpus[g].cublas, gpus[g].stream));
        CUDA_CHECK(cudaMemcpyToSymbol(d_cw, CLASS_WEIGHTS, K * sizeof(float)));

        // Copy Omega and phi to this GPU
        CUDA_CHECK(cudaMalloc(&gpus[g].d_Omega, (long long)F * IN_F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_phi,   F * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(gpus[g].d_Omega, h_Omega.data(),
                                   (long long)F * IN_F * sizeof(float),
                                   cudaMemcpyHostToDevice, gpus[g].stream));
        CUDA_CHECK(cudaMemcpyAsync(gpus[g].d_phi, h_phi.data(),
                                   F * sizeof(float),
                                   cudaMemcpyHostToDevice, gpus[g].stream));

        // Partition training data for this GPU
        int base = (long long)g * N_tr / N_GPUS;
        int Ng   = (long long)(g + 1) * N_tr / N_GPUS - base;
        gpus[g].N_gpu = Ng;

        // Upload raw features, transform to RFF, then free raw buffer
        float* d_Xraw = nullptr;
        CUDA_CHECK(cudaMalloc(&d_Xraw, (long long)Ng * IN_F * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(d_Xraw,
                                   h_Xtr.data() + (long long)base * IN_F,
                                   (long long)Ng * IN_F * sizeof(float),
                                   cudaMemcpyHostToDevice, gpus[g].stream));

        CUDA_CHECK(cudaMalloc(&gpus[g].d_X, (long long)Ng * F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpus[g].d_y, Ng * sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(gpus[g].d_y,
                                   h_ytr.data() + base,
                                   Ng * sizeof(int),
                                   cudaMemcpyHostToDevice, gpus[g].stream));

        // Sync before kernel so Omega/phi are ready
        CUDA_CHECK(cudaStreamSynchronize(gpus[g].stream));

        {
            long long total = (long long)Ng * F;
            int th = 256;
            int bl = (int)((total + th - 1) / th);
            rff_transform_kernel<<<bl, th, 0, gpus[g].stream>>>(
                d_Xraw, gpus[g].d_Omega, gpus[g].d_phi,
                gpus[g].d_X,
                Ng, IN_F, F, sqrtf(2.f / F));
        }
        CUDA_CHECK(cudaStreamSynchronize(gpus[g].stream));
        CUDA_CHECK(cudaFree(d_Xraw));

        printf("[%s]   GPU %d: %d samples transformed.\n", ts().c_str(), g, Ng);

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
    printf("[%s] GPU init complete. Beginning training.\n\n", ts().c_str());

    // ---- Training log ----------------------------------------------------
    std::ofstream log_f("training_rff_log.csv");
    log_f << "epoch,step,lr,loss,train_acc,acc,gap,recall,precision,holdout\n";

    std::vector<float> h_W(K * F), h_b(K);
    std::vector<float> best_W(K * F), best_b(K);
    float best_acc    = -1.f;
    int   global_step = 0;
    int   patience_cnt = 0;

    // CPU AllReduce buffers (K*F = 4*2048 = 8192 floats = 32 KB — fast)
    std::vector<float> h_gW_acc(K * F), h_gb_acc(K);
    std::vector<float> h_gW_tmp(K * F), h_gb_tmp(K);

    const int steps_per_epoch = gpus[0].N_gpu / BATCH_SIZE;
    std::mt19937 rng(42);

    // ---- Training loop ---------------------------------------------------
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

                // Random batch indices
                {
                    std::vector<int> idx(BATCH_SIZE);
                    std::uniform_int_distribution<int> dist(0, G.N_gpu - 1);
                    for (auto& i : idx) i = dist(rng);
                    CUDA_CHECK(cudaMemcpyAsync(G.d_idx, idx.data(),
                                               BATCH_SIZE * sizeof(int),
                                               cudaMemcpyHostToDevice, G.stream));
                }

                // Gather RFF features + labels into mini-batch buffers
                {
                    int th = 256;
                    int bl = ((long long)BATCH_SIZE * F + th - 1) / th;
                    gather_kernel<<<bl, th, 0, G.stream>>>(
                        G.d_X, G.d_idx, G.d_batchX, BATCH_SIZE, F);
                    gather_kernel<<<(BATCH_SIZE+th-1)/th, th, 0, G.stream>>>(
                        reinterpret_cast<const float*>(G.d_y),
                        G.d_idx,
                        reinterpret_cast<float*>(G.d_batchY),
                        BATCH_SIZE, 1);
                }

                if (g == 0)
                    CUDA_CHECK(cudaMemsetAsync(G.d_loss, 0, sizeof(float), G.stream));

                // Forward: D[B×K] = batchX[B×F] @ W^T + b
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

                // Hinge mask M[B×K] + loss accumulation
                {
                    const int th    = 256;
                    const int items = BATCH_SIZE * K;
                    hinge_mask_kernel<<<(items+th-1)/th, th,
                                        th*sizeof(float), G.stream>>>(
                        G.d_D, G.d_batchY, G.d_M, G.d_loss, BATCH_SIZE, K);
                }

                // Backward: gW[K×F] = M^T @ batchX
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

            // -- CPU AllReduce: sync, gather gW/gb, sum, broadcast --
            // K*F = 8192 floats = 32 KB per GPU; total ~128 KB bandwidth per step
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

            // -- Scale + Adam update (identical on every GPU) --
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
                        lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, 0.f,
                        K, global_step);
                }
            }

            // Periodic loss logging
            if (global_step % LOG_STEPS == 0) {
                float h_loss = 0.f;
                CUDA_CHECK(cudaSetDevice(0));
                CUDA_CHECK(cudaStreamSynchronize(gpus[0].stream));
                CUDA_CHECK(cudaMemcpy(&h_loss, gpus[0].d_loss, sizeof(float),
                                      cudaMemcpyDeviceToHost));
                double step_loss = h_loss / BATCH_SIZE;
                epoch_loss  += step_loss;
                epoch_steps++;
                printf("[%s] epoch %d  step %5d  lr=%.2e  loss=%.4f\n",
                       ts().c_str(), epoch + 1, global_step, (double)lr, step_loss);
            }

        } // end steps loop

        // ---- End-of-epoch: sync + pull weights from GPU 0 ---------------
        for (int g = 0; g < N_GPUS; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaStreamSynchronize(gpus[g].stream));
        }
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(h_W.data(), gpus[0].d_W,
                              (size_t)K * F * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b.data(), gpus[0].d_b,
                              K * sizeof(float), cudaMemcpyDeviceToHost));

        // ---- Test evaluation on GPU 0 (raw → RFF → scores) ---------------
        float* d_teXraw = nullptr;
        float* d_teXrff = nullptr;
        float* d_teD    = nullptr;
        int*   d_teP    = nullptr;
        CUDA_CHECK(cudaMalloc(&d_teXraw, (size_t)VAL_CHUNK * IN_F * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_teXrff, (size_t)VAL_CHUNK * F    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_teD,    (size_t)VAL_CHUNK * K    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_teP,    VAL_CHUNK * sizeof(int)));

        long cm[N_TRAIN_CLASSES][N_TRAIN_CLASSES] = {};
        long n_holdout = 0;
        std::vector<int> h_preds(VAL_CHUNK);

        for (int off = 0; off < N_te; off += VAL_CHUNK) {
            const int chunk = std::min(VAL_CHUNK, N_te - off);

            CUDA_CHECK(cudaMemcpy(d_teXraw,
                                  h_Xte.data() + (size_t)off * IN_F,
                                  (size_t)chunk * IN_F * sizeof(float),
                                  cudaMemcpyHostToDevice));

            // RFF transform
            {
                long long total = (long long)chunk * F;
                int th = 256, bl = (int)((total + th - 1) / th);
                rff_transform_kernel<<<bl, th, 0, gpus[0].stream>>>(
                    d_teXraw, gpus[0].d_Omega, gpus[0].d_phi,
                    d_teXrff, chunk, IN_F, F, sqrtf(2.f / F));
            }

            // Forward
            const float alpha = 1.f, beta = 0.f;
            CUBLAS_CHECK(cublasSgemm(gpus[0].cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                K, chunk, F,
                &alpha, gpus[0].d_W, F,
                         d_teXrff,   F,
                &beta,   d_teD,      K));
            {
                const int th = 256;
                add_bias_kernel<<<(chunk*K+th-1)/th, th, 0, gpus[0].stream>>>(
                    d_teD, gpus[0].d_b, chunk, K);
                argmax_kernel<<<(chunk+th-1)/th, th, 0, gpus[0].stream>>>(
                    d_teD, d_teP, chunk, K);
            }
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

        CUDA_CHECK(cudaFree(d_teXraw));
        CUDA_CHECK(cudaFree(d_teXrff));
        CUDA_CHECK(cudaFree(d_teD));
        CUDA_CHECK(cudaFree(d_teP));

        // ---- Train accuracy on first TRAIN_EVAL_SAMPLES ------------------
        float train_acc = 0.f;
        {
            const int N_tr_eval = std::min(TRAIN_EVAL_SAMPLES, N_tr);
            float* d_trXraw = nullptr;
            float* d_trXrff = nullptr;
            float* d_trD    = nullptr;
            int*   d_trP    = nullptr;
            CUDA_CHECK(cudaMalloc(&d_trXraw, (size_t)VAL_CHUNK * IN_F * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_trXrff, (size_t)VAL_CHUNK * F    * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_trD,    (size_t)VAL_CHUNK * K    * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_trP,    VAL_CHUNK * sizeof(int)));

            long tr_correct = 0, tr_total = 0;
            std::vector<int> h_tr_preds(VAL_CHUNK);

            for (int off = 0; off < N_tr_eval; off += VAL_CHUNK) {
                const int chunk = std::min(VAL_CHUNK, N_tr_eval - off);

                CUDA_CHECK(cudaMemcpy(d_trXraw,
                                      h_Xtr.data() + (size_t)off * IN_F,
                                      (size_t)chunk * IN_F * sizeof(float),
                                      cudaMemcpyHostToDevice));

                {
                    long long total = (long long)chunk * F;
                    int th = 256, bl = (int)((total + th - 1) / th);
                    rff_transform_kernel<<<bl, th, 0, gpus[0].stream>>>(
                        d_trXraw, gpus[0].d_Omega, gpus[0].d_phi,
                        d_trXrff, chunk, IN_F, F, sqrtf(2.f / F));
                }

                const float alpha = 1.f, beta = 0.f;
                CUBLAS_CHECK(cublasSgemm(gpus[0].cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    K, chunk, F,
                    &alpha, gpus[0].d_W, F,
                             d_trXrff,   F,
                    &beta,   d_trD,      K));
                {
                    const int th = 256;
                    add_bias_kernel<<<(chunk*K+th-1)/th, th, 0, gpus[0].stream>>>(
                        d_trD, gpus[0].d_b, chunk, K);
                    argmax_kernel<<<(chunk+th-1)/th, th, 0, gpus[0].stream>>>(
                        d_trD, d_trP, chunk, K);
                }
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

            CUDA_CHECK(cudaFree(d_trXraw));
            CUDA_CHECK(cudaFree(d_trXrff));
            CUDA_CHECK(cudaFree(d_trD));
            CUDA_CHECK(cudaFree(d_trP));

            train_acc = tr_total > 0 ? (float)tr_correct / (float)tr_total : 0.f;
        }

        // ---- Compute and print per-class metrics -------------------------
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

        // Checkpoint + early stopping
        if (acc > best_acc + MIN_DELTA) {
            best_acc = acc;
            best_W   = h_W;
            best_b   = h_b;
            patience_cnt = 0;
            save_model(best_W.data(), best_b.data(), K, F, "best_rff_model");
            save_rff_params(h_Omega, h_phi, F, IN_F);
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

    printf("[%s] Training complete.  Best test acc=%.4f\n\n",
           ts().c_str(), (double)best_acc);
    return 0;
}
