/*
 * ============================================================================
 * cuda_train.cu  (RBF-SVM via Random Fourier Features, One-vs-One, no NCCL)
 * ============================================================================
 * Purpose       : Multi-GPU One-vs-One RBF SVM training on CICIDS2017.
 *                 The RBF kernel k(x,y)=exp(-γ||x-y||²) is approximated by
 *                 explicit Random Fourier Features (Rahimi & Recht 2007):
 *                   z(x) = sqrt(2/D) · cos(Ω x + b_rff)
 *                 where Ω ~ N(0, sqrt(2γ)), b_rff ~ Uniform(0,2π), D=N_RFF.
 *
 * One-vs-One    : For N_CLASSES=4 we train C(4,2)=6 binary classifiers,
 *                 one per class pair:
 *                   c=0:(DDoS,DoS)  c=1:(DDoS,Normal)  c=2:(DDoS,PortScan)
 *                   c=3:(DoS,Normal) c=4:(DoS,PortScan)  c=5:(Normal,PortScan)
 *                 For each classifier, only samples whose label matches one of
 *                 the pair's two classes contribute gradients.
 *                 Validation inference uses majority vote across all 6 pairs.
 *
 * Algorithm     : Data-parallel mini-batch SGD.  Per step:
 *                   1. sample_indices_kernel  — random batch indices (cuRAND)
 *                   2. hinge_grad_kernel       — OvO hinge gradient on Z
 *                   3. Host AllReduce          — two-barrier reduce + avg
 *                   4. adam_update_kernel      — Adam weight update
 *                   5. cudaMemsetAsync         — zero gradient buffers
 *
 * GPU           : 4× RTX 2080 (sm_75, 8 GB VRAM, 46 SMs)
 * Frameworks    : CUDA 12.x + cuRAND + NVTX3
 *
 * Build         : nvcc -O3 -arch=sm_75 -std=c++17 cuda_train.cu \
 *                      -lcurand -o cuda_train
 *
 * Output Files  : model.bin        — nc, nf, n_rff, ncls,
 *                                    W[ncls×n_rff], b[ncls],
 *                                    omega[n_rff×nf], rff_b[n_rff]
 *                 training_log.csv — epoch,lr,train_loss,val_accuracy,
 *                                    val_macro_f1,epoch_train_ms,
 *                                    epoch_predict_ms,epoch_gflops
 * ============================================================================
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdio>
#include <string>
#include <numeric>
#include <cassert>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>

/* ── C++17-compatible reusable barrier ─────────────────────────────────── */
struct Barrier {
    explicit Barrier(int n) : n_(n), count_(n), gen_(0) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(mtx_);
        int gen = gen_;
        if (--count_ == 0) { ++gen_; count_ = n_; cv_.notify_all(); }
        else cv_.wait(lk, [this, gen]{ return gen_ != gen; });
    }
private:
    std::mutex              mtx_;
    std::condition_variable cv_;
    int n_, count_, gen_;
};

/* ── Error-checking macro ──────────────────────────────────────────────── */
#define CUDA_CHECK(call) do {                                                 \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                            \
                __FILE__, __LINE__, cudaGetErrorString(_e));                  \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while(0)

/* ── Constants ─────────────────────────────────────────────────────────── */
static constexpr int   NUM_GPUS        = 4;
static constexpr int   N_FEATURES      = 52;
static constexpr int   N_CLASSES       = 4;
static constexpr int   N_CLASSIFIERS   = 6;   /* C(4,2) binary classifiers  */
static constexpr int   N_RFF           = 1024; /* Random Fourier Feature dim */
static constexpr float RBF_GAMMA       = 0.01f; /* γ: k(x,z)=exp(-γ·||x-z||²); 1/(2·F)≈0.01 for F=52 */
static constexpr int   EPOCHS          = 200;
static constexpr int   BATCH_SIZE      = 2048;
static constexpr float LAMBDA          = 5e-4f;
static constexpr float LR_PEAK         = 5e-2f;
static constexpr float LR_MIN          = 1e-3f;
static constexpr float ADAM_BETA1      = 0.9f;
static constexpr float ADAM_BETA2      = 0.999f;
static constexpr float ADAM_EPS        = 1e-8f;
static constexpr int   PATIENCE        = 30;
static constexpr float MIN_DELTA       = 1e-4f;
static constexpr int   BLOCK_SIZE      = 128;
static constexpr int   WARP_SIZE       = 32;
static constexpr int   VAL_INTERVAL    = 10;

/* ── Validation DBSCAN constants ─────────────────────────────────────── */
static constexpr float VAL_CONF_THRESH = 0.5f;   /* vote fraction threshold */
static constexpr int   N_MAX_UNCERTAIN = 35000;   /* cap on DBSCAN pool size */
static constexpr int   VAL_MIN_SAMPLES = 8;
static constexpr float VAL_DBSCAN_EPS  = 0.148f;
static constexpr int   DBSCAN_GPU_THRESH = 20000;
static constexpr int   DBSCAN_TILE     = 32;

/* Per-class weights (index = internal class id) */
__constant__ float c_class_weights[N_CLASSES] = { 2.464f, 1.800f, 0.491f, 4.637f };

/* Class pair table: c_pairs[c] = {positive_class, negative_class}
 * CUDA initialises __constant__ variables at module load on every device. */
__constant__ int c_pairs[N_CLASSIFIERS][2] = {
    {0, 1},   /* c=0: DDoS  vs DoS      */
    {0, 2},   /* c=1: DDoS  vs Normal   */
    {0, 3},   /* c=2: DDoS  vs PortScan */
    {1, 2},   /* c=3: DoS   vs Normal   */
    {1, 3},   /* c=4: DoS   vs PortScan */
    {2, 3}    /* c=5: Normal vs PortScan*/
};

static constexpr int ORIG2INT[7] = { -1, -1, 0, 1, 2, 3, -1 };

/* ── Shared state between threads ──────────────────────────────────────── */
static float* g_h_X      = nullptr;
static int*   g_h_y      = nullptr;
static float* g_h_omega  = nullptr;
static float* g_h_rff_b  = nullptr;

/* ── Test-set data (CPU pinned, loaded once, used by GPU 0 validation) ── */
static float* g_h_X_test = nullptr;
static int*   g_h_y_test = nullptr;
static int    g_N_test   = 0;

/* Host-side AllReduce staging */
static float* g_h_dW_staging[NUM_GPUS]; /* each [N_CLASSIFIERS × N_RFF] */
static float* g_h_db_staging[NUM_GPUS]; /* each [N_CLASSIFIERS]          */
static float  g_h_dW_sum[N_CLASSIFIERS * N_RFF];
static float  g_h_db_sum[N_CLASSIFIERS];

static std::atomic<int> g_stop_flag{0};
static int g_steps_per_epoch = 1;

/* ==========================================================================
 * KERNEL 1: sample_indices_kernel
 * ==========================================================================
 */
__global__ void sample_indices_kernel(int* d_idx, int local_n,
                                       unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= BATCH_SIZE) return;
    curandState_t state;
    curand_init(seed, (unsigned long long)tid, 0ULL, &state);
    float r = curand_uniform(&state);
    d_idx[tid] = (int)(r * (float)local_n) % local_n;
}

/* ==========================================================================
 * KERNEL 2: rff_transform_kernel
 *
 * Purpose : z[i][j] = sqrt(2/N_RFF) * cos( omega[j] · x[i] + rff_b[j] )
 *           One thread per (sample i, RFF dim j).
 * ==========================================================================
 */
__global__ void rff_transform_kernel(
    const float* __restrict__ d_X,
    const float* __restrict__ d_omega,
    const float* __restrict__ d_rff_b,
    float* d_Z,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n * N_RFF) return;

    int i = tid / N_RFF;
    int j = tid % N_RFF;

    const float* xi      = d_X     + (long long)i * N_FEATURES;
    const float* omega_j = d_omega + (long long)j * N_FEATURES;

    float dot = __ldg(&d_rff_b[j]);
    #pragma unroll 4
    for (int f = 0; f < N_FEATURES; f++)
        dot += __ldg(&omega_j[f]) * __ldg(&xi[f]);

    d_Z[(long long)i * N_RFF + j] = sqrtf(2.0f / N_RFF) * cosf(dot);
}

/* ==========================================================================
 * KERNEL 3: hinge_grad_kernel  (One-vs-One, N_CLASSIFIERS=6 binary SVMs)
 *
 * Purpose : For each mini-batch sample, contributes hinge gradient to the
 *           3 classifiers whose pair includes the sample's class (each class
 *           appears in exactly N_CLASSES-1 = 3 pairs).  The other 3
 *           classifiers receive zero gradient from this sample.
 *
 * Shmem   : s_W [N_CLASSIFIERS × N_RFF] ≈ 12 KB
 *           s_dW[N_CLASSIFIERS × N_RFF] ≈ 12 KB
 *           s_db[N_CLASSIFIERS]          =  24 B
 *           Total ≈ 24 KB — within 48 KB limit.
 *
 * Occupancy (sm_75, BLOCK_SIZE=128):
 *   Shmem/block ≈ 24 KB → floor(49152/24576) = 2 blocks/SM
 *   Active warps = 2 × 4 = 8 / 64 — shmem-bound, N_RFF drives this.
 * ==========================================================================
 */
__global__ void hinge_grad_kernel(
    const float* __restrict__ d_Z_local,
    const int*   __restrict__ d_y_local,
    const int*   __restrict__ d_idx,
    int   batch_size,
    const float* __restrict__ d_W,
    const float* __restrict__ d_b,
    float* d_dW,
    float* d_db,
    float* d_loss)
{
    __shared__ float s_W [N_CLASSIFIERS * N_RFF];
    /* s_dW removed — gradients atomicAdd directly to global d_dW */
    __shared__ float s_db[N_CLASSIFIERS];
    __shared__ float s_loss_blk;

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * BLOCK_SIZE + tid;
    int lane = tid & 31;

    for (int j = tid; j < N_CLASSIFIERS * N_RFF; j += BLOCK_SIZE)
        s_W[j] = d_W[j];
    if (tid < N_CLASSIFIERS) s_db[tid] = 0.0f;
    if (tid == 0) s_loss_blk = 0.0f;
    __syncthreads();

    /* hinge_scale[c] / db_delta[c]: non-zero only for the 3 pairs that
     * include this sample's class; zero for all others.               */
    float hinge_scale[N_CLASSIFIERS] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    float db_delta[N_CLASSIFIERS]    = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    if (gid < batch_size) {
        int   idx   = d_idx[gid];
        int   label = d_y_local[idx];
        const float* z = d_Z_local + (long long)idx * N_RFF;

        #pragma unroll
        for (int c = 0; c < N_CLASSIFIERS; c++) {
            int pos = c_pairs[c][0];
            int neg = c_pairs[c][1];
            if (label != pos && label != neg) continue;  /* irrelevant pair */

            float y_c = (label == pos) ? 1.0f : -1.0f;
            float w   = c_class_weights[label];

            float score = d_b[c];
            const float* wc = s_W + c * N_RFF;
            #pragma unroll 8
            for (int f = 0; f < N_RFF; f++)
                score += wc[f] * __ldg(z + f);

            float margin = y_c * score * w;
            if (margin < 1.0f) {
                float sc = -y_c * w;
                hinge_scale[c] = sc;
                db_delta[c]    = sc;
            }
        }
    }

    /* Intra-warp reduction over N_CLASSIFIERS scalars */
    #pragma unroll
    for (int c = 0; c < N_CLASSIFIERS; c++) {
        #pragma unroll
        for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
            hinge_scale[c] += __shfl_down_sync(0xFFFFFFFFu, hinge_scale[c], off);
            db_delta[c]    += __shfl_down_sync(0xFFFFFFFFu, db_delta[c],    off);
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int c = 0; c < N_CLASSIFIERS; c++)
            atomicAdd(&s_db[c], db_delta[c]);
    }

    /* Per-thread feature-level gradient → s_dW + per-sample loss */
    float thread_loss = 0.0f;
    if (gid < batch_size) {
        int   idx   = d_idx[gid];
        int   label = d_y_local[idx];
        const float* z = d_Z_local + (long long)idx * N_RFF;

        float my_scale[N_CLASSIFIERS] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

        #pragma unroll
        for (int c = 0; c < N_CLASSIFIERS; c++) {
            int pos = c_pairs[c][0];
            int neg = c_pairs[c][1];
            if (label != pos && label != neg) continue;

            float y_c = (label == pos) ? 1.0f : -1.0f;
            float w   = c_class_weights[label];

            float score = d_b[c];
            const float* wc = s_W + c * N_RFF;
            #pragma unroll 8
            for (int f = 0; f < N_RFF; f++)
                score += wc[f] * __ldg(z + f);

            float margin = y_c * score * w;
            if (margin < 1.0f) {
                my_scale[c]  = -y_c * w;
                thread_loss += (1.0f - margin);
            }
        }

        #pragma unroll
        for (int c = 0; c < N_CLASSIFIERS; c++) {
            if (my_scale[c] != 0.0f) {
                float* gdwc = d_dW + c * N_RFF;
                #pragma unroll 8
                for (int f = 0; f < N_RFF; f++)
                    atomicAdd(&gdwc[f], my_scale[c] * __ldg(z + f));
            }
        }
    }

    #pragma unroll
    for (int off = WARP_SIZE >> 1; off > 0; off >>= 1)
        thread_loss += __shfl_down_sync(0xFFFFFFFFu, thread_loss, off);
    if (lane == 0) atomicAdd(&s_loss_blk, thread_loss);

    __syncthreads();

    if (tid < N_CLASSIFIERS)
        atomicAdd(&d_db[tid], s_db[tid]);
    if (tid == 0)
        atomicAdd(d_loss, s_loss_blk);
}

/* ==========================================================================
 * KERNEL 4: adam_update_kernel
 * ==========================================================================
 */
__global__ void adam_update_kernel(
    float* d_W,  const float* d_dW,
    float* d_mW, float* d_vW,
    int n, float lr, float beta1, float beta2, float eps, int t, float lambda)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = d_dW[i] + lambda * d_W[i];
    float m  = beta1 * d_mW[i] + (1.0f - beta1) * g;
    float v  = beta2 * d_vW[i] + (1.0f - beta2) * g * g;
    d_mW[i]  = m;
    d_vW[i]  = v;
    float mh = m / (1.0f - __powf(beta1, (float)t));
    float vh = v / (1.0f - __powf(beta2, (float)t));
    d_W[i]  -= lr * mh / (sqrtf(vh) + eps);
}

__global__ void adam_update_bias_kernel(
    float* d_b,  const float* d_db,
    float* d_mb, float* d_vb,
    int n, float lr, float beta1, float beta2, float eps, int t)
{
    int i = threadIdx.x;
    if (i >= n) return;
    float g  = d_db[i];
    float m  = beta1 * d_mb[i] + (1.0f - beta1) * g;
    float v  = beta2 * d_vb[i] + (1.0f - beta2) * g * g;
    d_mb[i]  = m;
    d_vb[i]  = v;
    float mh = m / (1.0f - __powf(beta1, (float)t));
    float vh = v / (1.0f - __powf(beta2, (float)t));
    d_b[i]  -= lr * mh / (sqrtf(vh) + eps);
}

/* ==========================================================================
 * KERNEL 5: infer_kernel  (validation, GPU 0 only, One-vs-One majority vote)
 *
 * Purpose : Computes 6 binary scores, casts one vote per classifier, and
 *           returns the class with the most votes (ties broken by first max).
 *
 * Shmem   : s_W[N_CLASSIFIERS × N_RFF] ≈ 12 KB.
 * ==========================================================================
 */
__global__ void infer_kernel(
    const float* __restrict__ d_Z_local,
    const float* __restrict__ d_W,
    const float* __restrict__ d_b,
    int* d_pred,
    int  local_n)
{
    __shared__ float s_W[N_CLASSIFIERS * N_RFF];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    for (int j = tid; j < N_CLASSIFIERS * N_RFF; j += blockDim.x)
        s_W[j] = d_W[j];
    __syncthreads();

    if (i >= local_n) return;

    const float* z = d_Z_local + (long long)i * N_RFF;

    /* Compute 6 binary classifier scores */
    float sc[N_CLASSIFIERS];
    #pragma unroll
    for (int c = 0; c < N_CLASSIFIERS; c++) {
        float val = d_b[c];
        const float* wc = s_W + c * N_RFF;
        #pragma unroll 8
        for (int f = 0; f < N_RFF; f++)
            val += wc[f] * __ldg(z + f);
        sc[c] = val;
    }

    /* Majority vote: each classifier votes for its winning class */
    int votes[N_CLASSES] = { 0, 0, 0, 0 };
    #pragma unroll
    for (int c = 0; c < N_CLASSIFIERS; c++)
        votes[sc[c] > 0.0f ? c_pairs[c][0] : c_pairs[c][1]]++;

    /* Argmax of votes */
    int best_k = 0;
    #pragma unroll
    for (int k = 1; k < N_CLASSES; k++)
        if (votes[k] > votes[best_k]) best_k = k;

    d_pred[i] = best_k;
}

/* ==========================================================================
 * KERNEL 6: val_infer_kernel  (validation, GPU 0, returns scores + conf + pred)
 *
 * Extends infer_kernel by also writing the 6 raw binary scores and the
 * vote-fraction confidence needed for the DBSCAN uncertain-sample pool.
 * ==========================================================================
 */
__global__ void val_infer_kernel(
    const float* __restrict__ d_Z_local,
    const float* __restrict__ d_W,
    const float* __restrict__ d_b,
    float* d_scores,   /* [n × N_CLASSIFIERS] — raw binary scores   */
    float* d_conf,     /* [n]                 — vote fraction        */
    int*   d_pred,
    int    n)
{
    __shared__ float s_W[N_CLASSIFIERS * N_RFF];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    for (int j = tid; j < N_CLASSIFIERS * N_RFF; j += blockDim.x)
        s_W[j] = d_W[j];
    __syncthreads();

    if (i >= n) return;

    const float* z = d_Z_local + (long long)i * N_RFF;

    float sc[N_CLASSIFIERS];
    #pragma unroll
    for (int c = 0; c < N_CLASSIFIERS; c++) {
        float val = d_b[c];
        const float* wc = s_W + c * N_RFF;
        #pragma unroll 8
        for (int f = 0; f < N_RFF; f++)
            val += wc[f] * __ldg(z + f);
        sc[c] = val;
    }

    int votes[N_CLASSES] = { 0, 0, 0, 0 };
    #pragma unroll
    for (int c = 0; c < N_CLASSIFIERS; c++)
        votes[sc[c] > 0.0f ? c_pairs[c][0] : c_pairs[c][1]]++;

    int best_k = 0;
    #pragma unroll
    for (int k = 1; k < N_CLASSES; k++)
        if (votes[k] > votes[best_k]) best_k = k;

    float* out_sc = d_scores + (long long)i * N_CLASSIFIERS;
    #pragma unroll
    for (int c = 0; c < N_CLASSIFIERS; c++) out_sc[c] = sc[c];
    d_conf[i]  = (float)votes[best_k] / (float)N_CLASSIFIERS;
    d_pred[i]  = best_k;
}

/* ==========================================================================
 * DBSCAN kernels (mirror of cuda_infer.cu; embedding dim = N_CLASSIFIERS)
 * ==========================================================================
 */
__global__ void dbscan_neighbor_kernel(
    const float* __restrict__ d_X_db,
    int   n_db,
    float eps2,
    int*  d_ncount)
{
    __shared__ float s_A[DBSCAN_TILE][N_CLASSIFIERS];
    __shared__ float s_B[DBSCAN_TILE][N_CLASSIFIERS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * DBSCAN_TILE + ty;
    int col = blockIdx.x * DBSCAN_TILE + tx;

    int linear = ty * DBSCAN_TILE + tx;
    if (linear < DBSCAN_TILE * N_CLASSIFIERS) {
        int p_in_tile = linear / N_CLASSIFIERS;
        int dim       = linear % N_CLASSIFIERS;
        int pr        = blockIdx.y * DBSCAN_TILE + p_in_tile;
        int pc        = blockIdx.x * DBSCAN_TILE + p_in_tile;
        s_A[p_in_tile][dim] = (pr < n_db) ? d_X_db[(long long)pr * N_CLASSIFIERS + dim] : 0.0f;
        s_B[p_in_tile][dim] = (pc < n_db) ? d_X_db[(long long)pc * N_CLASSIFIERS + dim] : 0.0f;
    }
    __syncthreads();

    if (row < n_db && col < n_db) {
        float d2 = 0.0f;
        #pragma unroll
        for (int d = 0; d < N_CLASSIFIERS; d++) {
            float diff = s_A[ty][d] - s_B[tx][d];
            d2 += diff * diff;
        }
        if (d2 < eps2) atomicAdd(&d_ncount[row], 1);
    }
}

__global__ void mark_core_kernel(const int* d_ncount, int* d_is_core,
                                  int n_db, int min_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_db)
        d_is_core[i] = ((d_ncount[i] - 1) >= min_samples) ? 1 : 0;
}

__global__ void dbscan_range_query_kernel(
    const float* __restrict__ d_X_db,
    int   n_db, int seed_idx, float eps2, int* d_reach)
{
    __shared__ float s_seed[N_CLASSIFIERS];
    if (threadIdx.x < N_CLASSIFIERS)
        s_seed[threadIdx.x] = d_X_db[(long long)seed_idx * N_CLASSIFIERS + threadIdx.x];
    __syncthreads();

    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n_db) return;
    float d2 = 0.0f;
    #pragma unroll
    for (int d = 0; d < N_CLASSIFIERS; d++) {
        float diff = d_X_db[(long long)i * N_CLASSIFIERS + d] - s_seed[d];
        d2 += diff * diff;
    }
    if (d2 < eps2) d_reach[i] = 1;
}

/* ── CPU DBSCAN BFS (n_db ≤ DBSCAN_GPU_THRESH) ─────────────────────────── */
static void cpu_dbscan(const float* X_db, int n_db, float eps,
                        int min_samples, std::vector<int>& cluster_id)
{
    float eps2 = eps * eps;
    cluster_id.assign(n_db, -1);
    std::vector<bool> visited(n_db, false);
    std::vector<bool> is_core(n_db, false);

    auto neighbours = [&](int i, std::vector<int>& nb) {
        nb.clear();
        for (int j = 0; j < n_db; j++) {
            if (j == i) continue;
            float d2 = 0.0f;
            for (int d = 0; d < N_CLASSIFIERS; d++) {
                float diff = X_db[(long long)i*N_CLASSIFIERS+d]
                           - X_db[(long long)j*N_CLASSIFIERS+d];
                d2 += diff * diff;
            }
            if (d2 < eps2) nb.push_back(j);
        }
    };

    std::vector<int> nb_tmp;
    for (int i = 0; i < n_db; i++) {
        neighbours(i, nb_tmp);
        if ((int)nb_tmp.size() >= min_samples) is_core[i] = true;
    }

    int cid = 0;
    std::queue<int> bfs;
    for (int i = 0; i < n_db; i++) {
        if (visited[i] || !is_core[i]) continue;
        visited[i] = true;
        cluster_id[i] = cid;
        bfs.push(i);
        while (!bfs.empty()) {
            int cur = bfs.front(); bfs.pop();
            std::vector<int> nb;
            neighbours(cur, nb);
            for (int q : nb) {
                if (!visited[q]) {
                    visited[q]    = true;
                    cluster_id[q] = cid;
                    if (is_core[q]) bfs.push(q);
                }
            }
        }
        cid++;
    }
}

/* ── GPU streaming DBSCAN BFS (n_db > DBSCAN_GPU_THRESH) ───────────────── */
static void gpu_dbscan(float* d_X_db, int n_db, float eps, int min_samples,
                        cudaStream_t stream, std::vector<int>& cluster_id)
{
    float eps2 = eps * eps;
    int *d_ncount, *d_is_core, *d_reach;
    CUDA_CHECK(cudaMalloc(&d_ncount,  n_db * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_is_core, n_db * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_reach,   n_db * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ncount, 0, n_db * sizeof(int)));

    dim3 tile_dim(DBSCAN_TILE, DBSCAN_TILE);
    int  grid_n = (n_db + DBSCAN_TILE - 1) / DBSCAN_TILE;
    dbscan_neighbor_kernel<<<dim3(grid_n, grid_n), tile_dim, 0, stream>>>(
        d_X_db, n_db, eps2, d_ncount);
    mark_core_kernel<<<(n_db+255)/256, 256, 0, stream>>>(
        d_ncount, d_is_core, n_db, min_samples);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int> h_is_core(n_db);
    CUDA_CHECK(cudaMemcpy(h_is_core.data(), d_is_core,
                          n_db*sizeof(int), cudaMemcpyDeviceToHost));

    cluster_id.assign(n_db, -1);
    std::vector<bool> visited(n_db, false);
    std::vector<int>  h_reach(n_db);
    int cid = 0;
    std::queue<int> bfs;

    for (int i = 0; i < n_db; i++) {
        if (visited[i] || !h_is_core[i]) continue;
        visited[i]    = true;
        cluster_id[i] = cid;
        bfs.push(i);
        while (!bfs.empty()) {
            int seed = bfs.front(); bfs.pop();
            CUDA_CHECK(cudaMemset(d_reach, 0, n_db * sizeof(int)));
            dbscan_range_query_kernel<<<(n_db+255)/256, 256, 0, stream>>>(
                d_X_db, n_db, seed, eps2, d_reach);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(h_reach.data(), d_reach,
                                  n_db*sizeof(int), cudaMemcpyDeviceToHost));
            for (int j = 0; j < n_db; j++) {
                if (h_reach[j] && !visited[j]) {
                    visited[j]    = true;
                    cluster_id[j] = cid;
                    if (h_is_core[j]) bfs.push(j);
                }
            }
        }
        cid++;
    }
    CUDA_CHECK(cudaFree(d_ncount));
    CUDA_CHECK(cudaFree(d_is_core));
    CUDA_CHECK(cudaFree(d_reach));
}

/* ── Save model helper ──────────────────────────────────────────────────── */
static void save_model_to_file(const float* h_W, const float* h_b,
                                const char* path)
{
    FILE* fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); return; }
    int nc = N_CLASSES, nf = N_FEATURES, nr = N_RFF, ncls = N_CLASSIFIERS;
    fwrite(&nc,   sizeof(int),   1,                     fp);
    fwrite(&nf,   sizeof(int),   1,                     fp);
    fwrite(&nr,   sizeof(int),   1,                     fp);
    fwrite(&ncls, sizeof(int),   1,                     fp);
    fwrite(h_W,   sizeof(float), N_CLASSIFIERS * N_RFF, fp);
    fwrite(h_b,   sizeof(float), N_CLASSIFIERS,         fp);
    fwrite(g_h_omega, sizeof(float), N_RFF * N_FEATURES, fp);
    fwrite(g_h_rff_b, sizeof(float), N_RFF,              fp);
    fclose(fp);
    fprintf(stdout, "  Saved %s\n", path);
}

/* ── CSV loading helpers ────────────────────────────────────────────────── */
static bool load_csv_features(const char* path, std::vector<double>& out,
                               int& nrows, int ncols)
{
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    std::string line;
    nrows = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        int col = 0;
        while (std::getline(ss, tok, ',')) { out.push_back(std::stod(tok)); col++; }
        if (col != ncols) {
            fprintf(stderr, "Row %d: expected %d cols, got %d\n", nrows, ncols, col);
            return false;
        }
        nrows++;
    }
    return true;
}

static bool load_csv_labels(const char* path, std::vector<int>& out)
{
    std::ifstream f(path);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        out.push_back(std::stoi(line));
    }
    return true;
}

/* OvO validation metrics: computes macro-F1 and accuracy together */
static float compute_val_metrics(const int* pred, const int* true_label, int n,
                                  float& out_accuracy)
{
    int tp[N_CLASSES]={}, fp[N_CLASSES]={}, fn_[N_CLASSES]={};
    int correct = 0;
    for (int i = 0; i < n; i++) {
        int p = pred[i], t = true_label[i];
        if (p == t) { tp[p]++; correct++; }
        else { fp[p]++; fn_[t]++; }
    }
    out_accuracy = n > 0 ? (float)correct / n : 0.0f;
    float macro_f1 = 0.0f;
    for (int k = 0; k < N_CLASSES; k++) {
        float prec = (tp[k]+fp[k]) > 0 ? (float)tp[k]/(tp[k]+fp[k]) : 0.0f;
        float rec  = (tp[k]+fn_[k]) > 0 ? (float)tp[k]/(tp[k]+fn_[k]) : 0.0f;
        float f1   = (prec+rec) > 0 ? 2*prec*rec/(prec+rec) : 0.0f;
        macro_f1  += f1;
    }
    return macro_f1 / N_CLASSES;
}

/* ── Per-GPU worker thread context ──────────────────────────────────────── */
struct GpuContext {
    int    gpu_id;
    int    local_n;
    int    local_start;
    float* d_X_local;
    int*   d_y_local;
    float* d_Z_local;    /* [local_n × N_RFF]            */
    float* d_omega;      /* [N_RFF × N_FEATURES]         */
    float* d_rff_b;      /* [N_RFF]                      */
    float* d_W;          /* [N_CLASSIFIERS × N_RFF]      */
    float* d_b;          /* [N_CLASSIFIERS]              */
    float* d_mW, *d_vW;
    float* d_mb, *d_vb;
    float* d_dW;
    float* d_db;
    int*   d_idx;
    float* d_loss;
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;
};

static Barrier* g_barrier             = nullptr;
static Barrier* g_barrier_allreduce_A = nullptr;
static Barrier* g_barrier_allreduce_B = nullptr;

static std::mutex  g_val_mutex;
static float       g_best_f1   = 0.0f;
static int         g_no_improve = 0;

static std::ofstream g_log_file;

/* Per-epoch timing (written by GPU 0, read by logger) */
static double g_epoch_train_ms   = 0.0;
static double g_epoch_predict_ms = 0.0;
static double g_epoch_dbscan_ms  = 0.0;
static std::chrono::high_resolution_clock::time_point g_train_start_tp;

static void gpu_worker(GpuContext* ctx)
{
    int g = ctx->gpu_id;
    CUDA_CHECK(cudaSetDevice(g));

    for (int j = 0; j < NUM_GPUS; j++) {
        if (j == g) continue;
        int can = 0;
        cudaDeviceCanAccessPeer(&can, g, j);
        if (can) cudaDeviceEnablePeerAccess(j, 0);
    }

    int    local_n = ctx->local_n;
    size_t X_sz    = (size_t)local_n * N_FEATURES       * sizeof(float);
    size_t y_sz    = (size_t)local_n                    * sizeof(int);
    size_t Z_sz    = (size_t)local_n * N_RFF             * sizeof(float);
    size_t W_sz    = (size_t)N_CLASSIFIERS * N_RFF       * sizeof(float);
    size_t b_sz    = (size_t)N_CLASSIFIERS               * sizeof(float);
    size_t Om_sz   = (size_t)N_RFF * N_FEATURES          * sizeof(float);
    size_t Rb_sz   = (size_t)N_RFF                       * sizeof(float);

    CUDA_CHECK(cudaMalloc(&ctx->d_X_local,  X_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_y_local,  y_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_Z_local,  Z_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_omega,    Om_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_rff_b,    Rb_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_W,        W_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_b,        b_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_mW,       W_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_vW,       W_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_mb,       b_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_vb,       b_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_dW,       W_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_db,       b_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_idx,      BATCH_SIZE  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_loss,     sizeof(float)));

    CUDA_CHECK(cudaStreamCreate(&ctx->stream_compute));
    CUDA_CHECK(cudaStreamCreate(&ctx->stream_transfer));

    CUDA_CHECK(cudaMemsetAsync(ctx->d_W,  0, W_sz, ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_b,  0, b_sz, ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_mW, 0, W_sz, ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_vW, 0, W_sz, ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_mb, 0, b_sz, ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_vb, 0, b_sz, ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_dW, 0, W_sz, ctx->stream_compute));
    CUDA_CHECK(cudaMemsetAsync(ctx->d_db, 0, b_sz, ctx->stream_compute));

    const float* h_X_slice = g_h_X + (size_t)ctx->local_start * N_FEATURES;
    const int*   h_y_slice = g_h_y + ctx->local_start;
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_X_local, h_X_slice, X_sz,
                               cudaMemcpyHostToDevice, ctx->stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_y_local, h_y_slice, y_sz,
                               cudaMemcpyHostToDevice, ctx->stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_omega,   g_h_omega, Om_sz,
                               cudaMemcpyHostToDevice, ctx->stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_rff_b,   g_h_rff_b, Rb_sz,
                               cudaMemcpyHostToDevice, ctx->stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream_transfer));

    /* Precompute RFF features once */
    {
        long long total_rff = (long long)local_n * N_RFF;
        rff_transform_kernel<<<(int)((total_rff + 255) / 256), 256, 0,
                                ctx->stream_compute>>>(
            ctx->d_X_local, ctx->d_omega, ctx->d_rff_b,
            ctx->d_Z_local, local_n);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));
        CUDA_CHECK(cudaFree(ctx->d_X_local));
        ctx->d_X_local = nullptr;
    }

    const int steps_per_epoch = g_steps_per_epoch;
    int global_step = 0;

    if (g == 0)
        g_train_start_tp = std::chrono::high_resolution_clock::now();
    g_barrier->arrive_and_wait();   /* all GPUs start timing together */

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        if (g_stop_flag.load()) break;

        if (g == 0) {
            char marker[64];
            snprintf(marker, sizeof(marker), "Epoch %d", epoch);
            nvtxRangePushA(marker);
        }

        float lr = LR_MIN + 0.5f * (LR_PEAK - LR_MIN) *
                   (1.0f + cosf((float)M_PI * epoch / EPOCHS));

        float epoch_loss = 0.0f;
        CUDA_CHECK(cudaMemsetAsync(ctx->d_loss, 0, sizeof(float),
                                   ctx->stream_compute));

        auto epoch_t0 = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < steps_per_epoch; step++) {

            global_step++;

            unsigned long long seed = (unsigned long long)g * 100000ULL
                                    + (unsigned long long)global_step;
            sample_indices_kernel<<<(BATCH_SIZE+255)/256, 256, 0,
                                    ctx->stream_compute>>>(
                ctx->d_idx, local_n, seed);

            int g_blocks = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
            hinge_grad_kernel<<<g_blocks, BLOCK_SIZE, 0, ctx->stream_compute>>>(
                ctx->d_Z_local, ctx->d_y_local, ctx->d_idx,
                BATCH_SIZE, ctx->d_W, ctx->d_b,
                ctx->d_dW, ctx->d_db, ctx->d_loss);

            CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));

            /* Host AllReduce — Phase A: copy gradients to staging */
            cudaMemcpy(g_h_dW_staging[g], ctx->d_dW, W_sz, cudaMemcpyDeviceToHost);
            cudaMemcpy(g_h_db_staging[g], ctx->d_db, b_sz, cudaMemcpyDeviceToHost);

            g_barrier_allreduce_A->arrive_and_wait();

            /* Phase B: GPU 0 sums and averages */
            if (g == 0) {
                const int WN = N_CLASSIFIERS * N_RFF;
                for (int j = 0; j < WN; j++) {
                    float s = 0.0f;
                    for (int gg = 0; gg < NUM_GPUS; gg++) s += g_h_dW_staging[gg][j];
                    g_h_dW_sum[j] = s / NUM_GPUS;
                }
                for (int j = 0; j < N_CLASSIFIERS; j++) {
                    float s = 0.0f;
                    for (int gg = 0; gg < NUM_GPUS; gg++) s += g_h_db_staging[gg][j];
                    g_h_db_sum[j] = s / NUM_GPUS;
                }
            }

            g_barrier_allreduce_B->arrive_and_wait();

            cudaMemcpy(ctx->d_dW, g_h_dW_sum, W_sz, cudaMemcpyHostToDevice);
            cudaMemcpy(ctx->d_db, g_h_db_sum, b_sz, cudaMemcpyHostToDevice);

            adam_update_kernel<<<(N_CLASSIFIERS*N_RFF+255)/256, 256,
                                  0, ctx->stream_compute>>>(
                ctx->d_W, ctx->d_dW, ctx->d_mW, ctx->d_vW,
                N_CLASSIFIERS * N_RFF, lr,
                ADAM_BETA1, ADAM_BETA2, ADAM_EPS, global_step, LAMBDA);

            adam_update_bias_kernel<<<1, N_CLASSIFIERS, 0, ctx->stream_compute>>>(
                ctx->d_b, ctx->d_db, ctx->d_mb, ctx->d_vb,
                N_CLASSIFIERS, lr, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, global_step);

            CUDA_CHECK(cudaMemsetAsync(ctx->d_dW, 0, W_sz, ctx->stream_compute));
            CUDA_CHECK(cudaMemsetAsync(ctx->d_db, 0, b_sz, ctx->stream_compute));
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));

            g_barrier->arrive_and_wait();
        }

        if (g == 0) {
            auto epoch_t1 = std::chrono::high_resolution_clock::now();
            g_epoch_train_ms = std::chrono::duration<double, std::milli>(
                epoch_t1 - epoch_t0).count();
            float raw_loss = 0.0f;
            CUDA_CHECK(cudaMemcpy(&raw_loss, ctx->d_loss, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            epoch_loss = raw_loss / (float)(steps_per_epoch * BATCH_SIZE);
        }

        float val_f1       = -1.0f;
        float val_accuracy = -1.0f;
        if (g == 0 && (epoch % VAL_INTERVAL == 0)) {
            auto pred_t0 = std::chrono::high_resolution_clock::now();

            static constexpr int VAL_CHUNK = 32768;
            int  N_test_local = g_N_test;
            std::vector<int>   h_pred_test  (N_test_local);
            std::vector<float> h_scores_test((size_t)N_test_local * N_CLASSIFIERS);
            std::vector<float> h_conf_test  (N_test_local);

            /* Temp device buffers for one chunk */
            float* d_Xchunk      = nullptr;
            float* d_Zchunk      = nullptr;
            int*   d_predchunk   = nullptr;
            float* d_scoreschunk = nullptr;
            float* d_confchunk   = nullptr;
            CUDA_CHECK(cudaMalloc(&d_Xchunk,
                (size_t)VAL_CHUNK * N_FEATURES   * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_Zchunk,
                (size_t)VAL_CHUNK * N_RFF         * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_predchunk,
                (size_t)VAL_CHUNK                 * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_scoreschunk,
                (size_t)VAL_CHUNK * N_CLASSIFIERS * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_confchunk,
                (size_t)VAL_CHUNK                 * sizeof(float)));

            for (int ofs = 0; ofs < N_test_local; ofs += VAL_CHUNK) {
                int cn = std::min(VAL_CHUNK, N_test_local - ofs);
                CUDA_CHECK(cudaMemcpy(d_Xchunk,
                    g_h_X_test + (size_t)ofs * N_FEATURES,
                    (size_t)cn * N_FEATURES * sizeof(float),
                    cudaMemcpyHostToDevice));
                long long total_rff = (long long)cn * N_RFF;
                rff_transform_kernel<<<(int)((total_rff + 255) / 256), 256,
                                       0, ctx->stream_compute>>>(
                    d_Xchunk, ctx->d_omega, ctx->d_rff_b, d_Zchunk, cn);
                int inf_blk = (cn + 255) / 256;
                val_infer_kernel<<<inf_blk, 256, 0, ctx->stream_compute>>>(
                    d_Zchunk, ctx->d_W, ctx->d_b,
                    d_scoreschunk, d_confchunk, d_predchunk, cn);
                CUDA_CHECK(cudaMemcpy(
                    h_pred_test.data() + ofs, d_predchunk,
                    (size_t)cn * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(
                    h_scores_test.data() + (size_t)ofs * N_CLASSIFIERS,
                    d_scoreschunk,
                    (size_t)cn * N_CLASSIFIERS * sizeof(float),
                    cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(
                    h_conf_test.data() + ofs, d_confchunk,
                    (size_t)cn * sizeof(float), cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream_compute));

            CUDA_CHECK(cudaFree(d_Xchunk));
            CUDA_CHECK(cudaFree(d_Zchunk));
            CUDA_CHECK(cudaFree(d_predchunk));
            CUDA_CHECK(cudaFree(d_scoreschunk));
            CUDA_CHECK(cudaFree(d_confchunk));

            auto pred_t1 = std::chrono::high_resolution_clock::now();
            g_epoch_predict_ms = std::chrono::duration<double, std::milli>(
                pred_t1 - pred_t0).count();

            val_f1 = compute_val_metrics(h_pred_test.data(), g_h_y_test,
                                          N_test_local, val_accuracy);

            /* ── DBSCAN on uncertain pool ─────────────────────────────── */
            auto db_t0 = std::chrono::high_resolution_clock::now();

            /* Collect low-confidence score vectors (cap at N_MAX_UNCERTAIN) */
            int n_uncertain = 0;
            std::vector<float> db_scores;
            db_scores.reserve((size_t)std::min(N_test_local, N_MAX_UNCERTAIN)
                              * N_CLASSIFIERS);
            for (int i = 0; i < N_test_local && n_uncertain < N_MAX_UNCERTAIN; i++) {
                if (h_conf_test[i] < VAL_CONF_THRESH) {
                    const float* sc = h_scores_test.data() + (size_t)i * N_CLASSIFIERS;
                    for (int c = 0; c < N_CLASSIFIERS; c++)
                        db_scores.push_back(sc[c]);
                    n_uncertain++;
                }
            }

            int n_clusters = 0;
            if (n_uncertain > 0) {
                std::vector<int> cluster_id;
                if (n_uncertain > DBSCAN_GPU_THRESH) {
                    float* d_db = nullptr;
                    CUDA_CHECK(cudaMalloc(&d_db,
                        (size_t)n_uncertain * N_CLASSIFIERS * sizeof(float)));
                    CUDA_CHECK(cudaMemcpy(d_db, db_scores.data(),
                        (size_t)n_uncertain * N_CLASSIFIERS * sizeof(float),
                        cudaMemcpyHostToDevice));
                    gpu_dbscan(d_db, n_uncertain, VAL_DBSCAN_EPS,
                               VAL_MIN_SAMPLES, ctx->stream_compute, cluster_id);
                    CUDA_CHECK(cudaFree(d_db));
                } else {
                    cpu_dbscan(db_scores.data(), n_uncertain,
                               VAL_DBSCAN_EPS, VAL_MIN_SAMPLES, cluster_id);
                }
                if (!cluster_id.empty())
                    n_clusters = *std::max_element(cluster_id.begin(),
                                                    cluster_id.end()) + 1;
            }

            auto db_t1 = std::chrono::high_resolution_clock::now();
            g_epoch_dbscan_ms = std::chrono::duration<double, std::milli>(
                db_t1 - db_t0).count();

            /* ── GFLOPS ──────────────────────────────────────────────── */
            double epoch_flops  = 2.0 * steps_per_epoch * BATCH_SIZE
                                      * N_CLASSIFIERS * N_RFF * NUM_GPUS;
            double epoch_gflops = epoch_flops / (g_epoch_train_ms * 1e-3) / 1e9;

            /* ── Total elapsed time from training start ──────────────── */
            double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now()
                - g_train_start_tp).count();

            {
                std::lock_guard<std::mutex> lock(g_val_mutex);

                /* Save best model checkpoint */
                if (val_f1 > g_best_f1 + MIN_DELTA) {
                    g_best_f1    = val_f1;
                    g_no_improve = 0;

                    std::vector<float> h_W(N_CLASSIFIERS * N_RFF);
                    std::vector<float> h_b(N_CLASSIFIERS);
                    CUDA_CHECK(cudaMemcpy(h_W.data(), ctx->d_W,
                        N_CLASSIFIERS * N_RFF * sizeof(float),
                        cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_b.data(), ctx->d_b,
                        N_CLASSIFIERS * sizeof(float),
                        cudaMemcpyDeviceToHost));
                    save_model_to_file(h_W.data(), h_b.data(), "model_best.bin");
                } else {
                    g_no_improve++;
                }
                if (g_no_improve >= PATIENCE)
                    g_stop_flag.store(1);

                fprintf(stdout,
                    "Epoch %3d | lr=%.5f | loss=%.4f | acc=%.4f"
                    " | F1=%.4f | train=%.1fms pred=%.1fms"
                    " dbscan=%.1fms total=%.1fs %.2fGFLOPS"
                    " uncert=%d cls=%d%s\n",
                    epoch, lr, epoch_loss, val_accuracy, val_f1,
                    g_epoch_train_ms, g_epoch_predict_ms,
                    g_epoch_dbscan_ms, total_ms / 1000.0,
                    epoch_gflops, n_uncertain, n_clusters,
                    g_no_improve >= PATIENCE ? " [EARLY STOP]" : "");
            }

            if (g_log_file.is_open())
                g_log_file
                    << epoch          << ","
                    << lr             << ","
                    << epoch_loss     << ","
                    << val_accuracy   << ","
                    << val_f1         << ","
                    << g_epoch_train_ms   << ","
                    << g_epoch_predict_ms << ","
                    << g_epoch_dbscan_ms  << ","
                    << total_ms           << ","
                    << epoch_gflops   << ","
                    << n_uncertain    << ","
                    << n_clusters     << "\n";
        }

        if (g == 0) nvtxRangePop();

        g_barrier->arrive_and_wait();
    }

    /* ── Save final model (GPU 0 only) ── */
    if (g == 0) {
        std::vector<float> h_W(N_CLASSIFIERS * N_RFF);
        std::vector<float> h_b(N_CLASSIFIERS);
        CUDA_CHECK(cudaMemcpy(h_W.data(), ctx->d_W,
                              N_CLASSIFIERS * N_RFF * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b.data(), ctx->d_b,
                              N_CLASSIFIERS * sizeof(float),
                              cudaMemcpyDeviceToHost));
        save_model_to_file(h_W.data(), h_b.data(), "model.bin");
        fprintf(stdout, "Saved model.bin (final) and model_best.bin (best val F1=%.4f)\n",
                g_best_f1);
    }

    /* ── Cleanup ── */
    if (ctx->d_X_local) CUDA_CHECK(cudaFree(ctx->d_X_local));
    CUDA_CHECK(cudaFree(ctx->d_y_local));
    CUDA_CHECK(cudaFree(ctx->d_Z_local));
    CUDA_CHECK(cudaFree(ctx->d_omega));
    CUDA_CHECK(cudaFree(ctx->d_rff_b));
    CUDA_CHECK(cudaFree(ctx->d_W));
    CUDA_CHECK(cudaFree(ctx->d_b));
    CUDA_CHECK(cudaFree(ctx->d_mW));
    CUDA_CHECK(cudaFree(ctx->d_vW));
    CUDA_CHECK(cudaFree(ctx->d_mb));
    CUDA_CHECK(cudaFree(ctx->d_vb));
    CUDA_CHECK(cudaFree(ctx->d_dW));
    CUDA_CHECK(cudaFree(ctx->d_db));
    CUDA_CHECK(cudaFree(ctx->d_idx));
    CUDA_CHECK(cudaFree(ctx->d_loss));
    CUDA_CHECK(cudaStreamDestroy(ctx->stream_compute));
    CUDA_CHECK(cudaStreamDestroy(ctx->stream_transfer));
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main()
{
    nvtxRangePushA("cuda_train:main");

    fprintf(stdout, "Loading training data...\n");

    std::vector<double> raw_X;
    std::vector<int>    raw_y;
    int nrows = 0;

    if (!load_csv_features("data/processed/train_data.csv",
                            raw_X, nrows, N_FEATURES)) exit(EXIT_FAILURE);
    if (!load_csv_labels("data/processed/train_labels.csv", raw_y))
        exit(EXIT_FAILURE);
    if ((int)raw_y.size() != nrows) {
        fprintf(stderr, "Row count mismatch: X=%d y=%zu\n", nrows, raw_y.size());
        exit(EXIT_FAILURE);
    }

    std::vector<float> X_clean;
    std::vector<int>   y_clean;
    X_clean.reserve(nrows * N_FEATURES);
    y_clean.reserve(nrows);
    for (int i = 0; i < nrows; i++) {
        int orig = raw_y[i];
        if (orig < 0 || orig >= 7) continue;
        int internal = ORIG2INT[orig];
        if (internal == -1) continue;
        for (int f = 0; f < N_FEATURES; f++)
            X_clean.push_back((float)raw_X[i * N_FEATURES + f]);
        y_clean.push_back(internal);
    }
    int N_train = (int)y_clean.size();
    fprintf(stdout, "Training samples after remap: %d\n", N_train);

    CUDA_CHECK(cudaHostAlloc(&g_h_X,
                              (size_t)N_train * N_FEATURES * sizeof(float),
                              cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&g_h_y,
                              (size_t)N_train * sizeof(int),
                              cudaHostAllocPortable));
    memcpy(g_h_X, X_clean.data(), (size_t)N_train * N_FEATURES * sizeof(float));
    memcpy(g_h_y, y_clean.data(), (size_t)N_train * sizeof(int));

    /* ── Load test data for proper validation ────────────────────────────── */
    fprintf(stdout, "Loading test data...\n");
    {
        std::vector<double> raw_X_test;
        std::vector<int>    raw_y_test;
        int nrows_test = 0;
        if (!load_csv_features("data/processed/test_data.csv",
                                raw_X_test, nrows_test, N_FEATURES))
            exit(EXIT_FAILURE);
        if (!load_csv_labels("data/processed/test_labels.csv", raw_y_test))
            exit(EXIT_FAILURE);

        std::vector<float> X_test_clean;
        std::vector<int>   y_test_clean;
        X_test_clean.reserve(nrows_test * N_FEATURES);
        y_test_clean.reserve(nrows_test);
        for (int i = 0; i < nrows_test; i++) {
            int orig = raw_y_test[i];
            if (orig < 0 || orig >= 7) continue;
            int internal = ORIG2INT[orig];
            if (internal == -1) continue;
            for (int f = 0; f < N_FEATURES; f++)
                X_test_clean.push_back((float)raw_X_test[i * N_FEATURES + f]);
            y_test_clean.push_back(internal);
        }
        g_N_test = (int)y_test_clean.size();
        fprintf(stdout, "Test samples after remap: %d\n", g_N_test);

        CUDA_CHECK(cudaHostAlloc(&g_h_X_test,
                                  (size_t)g_N_test * N_FEATURES * sizeof(float),
                                  cudaHostAllocPortable));
        CUDA_CHECK(cudaHostAlloc(&g_h_y_test,
                                  (size_t)g_N_test * sizeof(int),
                                  cudaHostAllocPortable));
        memcpy(g_h_X_test, X_test_clean.data(),
               (size_t)g_N_test * N_FEATURES * sizeof(float));
        memcpy(g_h_y_test, y_test_clean.data(),
               (size_t)g_N_test * sizeof(int));
    }

    /* Generate RFF parameters */
    fprintf(stdout, "Generating RFF parameters (N_RFF=%d, gamma=%.4f)...\n",
            N_RFF, RBF_GAMMA);
    CUDA_CHECK(cudaHostAlloc(&g_h_omega,
                              (size_t)N_RFF * N_FEATURES * sizeof(float),
                              cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&g_h_rff_b,
                              (size_t)N_RFF * sizeof(float),
                              cudaHostAllocPortable));
    {
        std::mt19937 rng(42);
        std::normal_distribution<float>      norm_dist(0.0f, sqrtf(2.0f * RBF_GAMMA));
        std::uniform_real_distribution<float> unif_dist(0.0f, 2.0f * (float)M_PI);
        for (int i = 0; i < N_RFF * N_FEATURES; i++)
            g_h_omega[i] = norm_dist(rng);
        for (int i = 0; i < N_RFF; i++)
            g_h_rff_b[i] = unif_dist(rng);
    }

    /* Allocate AllReduce staging buffers */
    for (int g = 0; g < NUM_GPUS; g++) {
        g_h_dW_staging[g] = new float[N_CLASSIFIERS * N_RFF]();
        g_h_db_staging[g] = new float[N_CLASSIFIERS]();
    }

    int base_chunk = N_train / NUM_GPUS;

    g_log_file.open("training_log.csv");
    if (!g_log_file.is_open()) {
        fprintf(stderr, "Cannot open training_log.csv\n"); exit(EXIT_FAILURE);
    }
    g_log_file << "epoch,lr,train_loss,val_accuracy,val_macro_f1,"
                  "train_ms,predict_ms,dbscan_ms,total_ms,"
                  "gflops,n_uncertain,n_clusters\n";

    Barrier bar(NUM_GPUS);
    Barrier bar_allreduce_A(NUM_GPUS);
    Barrier bar_allreduce_B(NUM_GPUS);
    g_barrier             = &bar;
    g_barrier_allreduce_A = &bar_allreduce_A;
    g_barrier_allreduce_B = &bar_allreduce_B;

    GpuContext ctx[NUM_GPUS];
    memset(ctx, 0, sizeof(ctx));
    for (int g = 0; g < NUM_GPUS; g++) {
        ctx[g].gpu_id = g;
        if (g == 0) {
            ctx[g].local_start = 0;
            ctx[g].local_n     = base_chunk + (N_train % NUM_GPUS);
        } else {
            ctx[g].local_start = (N_train % NUM_GPUS) + g * base_chunk;
            ctx[g].local_n     = base_chunk;
        }
    }

    g_steps_per_epoch = std::max(1, base_chunk / BATCH_SIZE);
    fprintf(stdout, "Steps per epoch: %d  (base_chunk=%d, BATCH_SIZE=%d)\n",
            g_steps_per_epoch, base_chunk, BATCH_SIZE);
    fprintf(stdout, "OvO classifiers: %d  (N_CLASSES=%d, N_RFF=%d)\n",
            N_CLASSIFIERS, N_CLASSES, N_RFF);

    std::thread threads[NUM_GPUS];
    auto t_train_start = std::chrono::high_resolution_clock::now();
    for (int g = 0; g < NUM_GPUS; g++)
        threads[g] = std::thread(gpu_worker, &ctx[g]);
    for (int g = 0; g < NUM_GPUS; g++)
        threads[g].join();
    auto t_train_end = std::chrono::high_resolution_clock::now();
    double total_train_ms = std::chrono::duration<double, std::milli>(
        t_train_end - t_train_start).count();

    g_log_file.close();

    /* ── Training summary ── */
    {
        double total_flops = 2.0 * g_steps_per_epoch * EPOCHS * BATCH_SIZE
                                 * N_CLASSIFIERS * N_RFF * NUM_GPUS;
        double total_gflops = total_flops / (total_train_ms * 1e-3) / 1e9;
        fprintf(stdout, "\n=== Training Summary ===\n");
        fprintf(stdout, "  Total Train Time : %.2f ms (%.2f s)\n",
                total_train_ms, total_train_ms / 1000.0);
        fprintf(stdout, "  Effective GFLOPS : %.3f\n", total_gflops);
        fprintf(stdout, "  Best val macro-F1: %.4f\n", g_best_f1);
    }

    for (int g = 0; g < NUM_GPUS; g++) {
        delete[] g_h_dW_staging[g];
        delete[] g_h_db_staging[g];
    }
    CUDA_CHECK(cudaFreeHost(g_h_X));
    CUDA_CHECK(cudaFreeHost(g_h_y));
    CUDA_CHECK(cudaFreeHost(g_h_omega));
    CUDA_CHECK(cudaFreeHost(g_h_rff_b));
    CUDA_CHECK(cudaFreeHost(g_h_X_test));
    CUDA_CHECK(cudaFreeHost(g_h_y_test));

    nvtxRangePop();
    fprintf(stdout, "Training complete.\n");
    return 0;
}
