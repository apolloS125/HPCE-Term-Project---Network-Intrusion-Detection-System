/*
 * ============================================================================
 * cuda_infer.cu  (RBF-SVM via Random Fourier Features, One-vs-One)
 * ============================================================================
 * Purpose       : Multi-GPU OvO RBF SVM inference on CICIDS2017 test set,
 *                 followed by confidence-based partitioning, GPU-accelerated
 *                 DBSCAN on the uncertain/holdout pool, cluster labelling,
 *                 and evaluation.
 *
 * Algorithm     :
 *   1. Load model.bin (nc, nf, n_rff, ncls,
 *                      W[ncls×n_rff], b[ncls], omega[n_rff×nf], rff_b[n_rff])
 *      produced by cuda_train (OvO, ncls=C(4,2)=6 binary classifiers).
 *   2. Distribute test data across 4 GPUs; per GPU:
 *        a. rff_transform_kernel: X[local_n×nf] → Z[local_n×N_RFF]
 *        b. ovo_infer_kernel: Z → 6 binary scores, majority-vote pred, conf
 *   3. Gather all scores, confidences, and predictions to GPU 0 / host.
 *   4. Partition:  confident → final prediction via INT2ORIG.
 *                  uncertain + holdout → DBSCAN pool.
 *   5. GPU DBSCAN (dbscan_neighbor_kernel) on 6-D OvO score vectors.
 *      GPU streaming BFS if n_db > 20K; else CPU BFS.
 *   6. Cluster labelling via holdout majority vote or Normal proximity.
 *   7. Evaluation: accuracy, per-class metrics + holdout detection rates.
 *   8. Write cuda_predictions.csv and cuda_log.csv.
 *
 * GPU           : 4× RTX 2080 (sm_75, 8 GB VRAM, 46 SMs)
 * Frameworks    : CUDA 12.x + NVTX3
 *
 * Build         : nvcc -O3 -arch=sm_75 -std=c++17 cuda_infer.cu \
 *                      -o cuda_infer
 *
 * Run           : ./cuda_infer
 *
 * Output Files  : cuda_predictions.csv  — original_label,predicted_label
 *                 cuda_log.csv          — accuracy,macro_f1,predict_ms,
 *                                         dbscan_ms,total_ms,gflops,...
 * ============================================================================
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <cassert>
#include <thread>
#include <mutex>
#include <atomic>

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
static constexpr int   NUM_GPUS          = 4;
static constexpr int   N_FEATURES        = 52;
static constexpr int   N_CLASSES         = 4;
static constexpr int   N_CLASSIFIERS     = 6;    /* C(4,2) OvO pairs            */
static constexpr int   N_RFF             = 1024;  /* must match cuda_train.cu    */
static constexpr int   SCORE_DIM         = N_CLASSIFIERS; /* DBSCAN embeds in 6-D OvO score space */

static constexpr float CONF_THRESHOLD    = 0.5f; /* fraction of votes for winner */
static constexpr int   N_MAX_UNCERTAIN   = 35000;
static constexpr int   MIN_SAMPLES       = 8;
static constexpr float EPS               = 0.148f;
static constexpr float NORMAL_PROX_THRESH= 4.0f;
static constexpr int   DBSCAN_GPU_THRESH = 20000;

/* Class mappings */
static constexpr int INT2ORIG[4] = { 2, 3, 4, 5 };
static constexpr int NORMAL_ORIG_ID = 4;

/* ── DBSCAN TILE size ── */
static constexpr int  DBSCAN_TILE = 32;

/* Class pair table for OvO inference (mirrors cuda_train.cu) */
__constant__ int c_pairs_infer[N_CLASSIFIERS][2] = {
    {0, 1},   /* c=0: DDoS  vs DoS      */
    {0, 2},   /* c=1: DDoS  vs Normal   */
    {0, 3},   /* c=2: DDoS  vs PortScan */
    {1, 2},   /* c=3: DoS   vs Normal   */
    {1, 3},   /* c=4: DoS   vs PortScan */
    {2, 3}    /* c=5: Normal vs PortScan*/
};

/* ==========================================================================
 * KERNEL 1: rff_transform_kernel
 *
 * Purpose : z[i][j] = sqrt(2/N_RFF) * cos( omega[j] · x[i] + rff_b[j] )
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
 * KERNEL 2: ovo_infer_kernel  (One-vs-One majority vote)
 *
 * Purpose : For each sample, compute 6 binary classifier scores, cast
 *           majority votes, and return the winning class.
 *           Also outputs all 6 raw scores as the DBSCAN embedding vector.
 *
 * Grid    : (local_n + 255) / 256 blocks, 256 threads per block.
 * Shmem   : s_W[N_CLASSIFIERS × N_RFF] ≈ 12 KB.
 *
 * Inputs  : d_Z_local [local_n × N_RFF]
 *           d_W       [N_CLASSIFIERS × N_RFF]  (6 binary SVM weight vectors)
 *           d_b       [N_CLASSIFIERS]
 * Outputs : d_scores  [local_n × N_CLASSIFIERS]  — 6 raw binary scores
 *           d_conf    [local_n]                  — vote fraction of winner
 *           d_pred    [local_n]                  — majority-vote class id
 * ==========================================================================
 */
__global__ void ovo_infer_kernel(
    const float* __restrict__ d_Z_local,
    const float* __restrict__ d_W,
    const float* __restrict__ d_b,
    float* d_scores,
    float* d_conf,
    int*   d_pred,
    int    local_n)
{
    __shared__ float s_W[N_CLASSIFIERS * N_RFF];  /* ≈ 12 KB */

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    /* Cooperatively load all 6 classifier weight vectors into shared memory */
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
        votes[sc[c] > 0.0f ? c_pairs_infer[c][0] : c_pairs_infer[c][1]]++;

    /* Argmax of votes */
    int best_k = 0;
    #pragma unroll
    for (int k = 1; k < N_CLASSES; k++)
        if (votes[k] > votes[best_k]) best_k = k;

    /* Write outputs */
    float* out_sc = d_scores + (long long)i * N_CLASSIFIERS;
    #pragma unroll
    for (int c = 0; c < N_CLASSIFIERS; c++) out_sc[c] = sc[c];
    d_conf[i] = (float)votes[best_k] / (float)N_CLASSIFIERS;
    d_pred[i] = best_k;
}

/* ==========================================================================
 * KERNEL 3: dbscan_neighbor_kernel
 *
 * Purpose : For every pair of points in the DBSCAN pool, checks if their
 *           squared L2 distance in SCORE_DIM-D space is < EPS², and if so
 *           atomicAdd(d_ncount[row], 1).  Uses shared-memory tiling.
 *
 * Grid    : 2-D grid of TILE×TILE thread blocks.
 * Shmem   : s_A[TILE][SCORE_DIM] + s_B[TILE][SCORE_DIM].
 * ==========================================================================
 */
__global__ void dbscan_neighbor_kernel(
    const float* __restrict__ d_X_db,
    int   n_db,
    float eps2,
    int*  d_ncount)
{
    __shared__ float s_A[DBSCAN_TILE][SCORE_DIM];
    __shared__ float s_B[DBSCAN_TILE][SCORE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * DBSCAN_TILE + ty;
    int col = blockIdx.x * DBSCAN_TILE + tx;

    int linear = ty * DBSCAN_TILE + tx;
    if (linear < DBSCAN_TILE * SCORE_DIM) {
        int p_in_tile = linear / SCORE_DIM;
        int dim       = linear % SCORE_DIM;
        int pr        = blockIdx.y * DBSCAN_TILE + p_in_tile;
        int pc        = blockIdx.x * DBSCAN_TILE + p_in_tile;
        s_A[p_in_tile][dim] = (pr < n_db)
            ? d_X_db[(long long)pr * SCORE_DIM + dim] : 0.0f;
        s_B[p_in_tile][dim] = (pc < n_db)
            ? d_X_db[(long long)pc * SCORE_DIM + dim] : 0.0f;
    }
    __syncthreads();

    if (row < n_db && col < n_db) {
        float d2 = 0.0f;
        #pragma unroll
        for (int d = 0; d < SCORE_DIM; d++) {
            float diff = s_A[ty][d] - s_B[tx][d];
            d2 += diff * diff;
        }
        if (d2 < eps2)
            atomicAdd(&d_ncount[row], 1);
    }
}

/* ==========================================================================
 * KERNEL 4: mark_core_kernel
 * ==========================================================================
 */
__global__ void mark_core_kernel(const int* d_ncount, int* d_is_core,
                                  int n_db, int min_samples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_db)
        d_is_core[i] = ((d_ncount[i] - 1) >= min_samples) ? 1 : 0;
}

/* ==========================================================================
 * KERNEL 5: dbscan_range_query_kernel   (GPU streaming BFS helper)
 * ==========================================================================
 */
__global__ void dbscan_range_query_kernel(
    const float* __restrict__ d_X_db,
    int   n_db,
    int   seed_idx,
    float eps2,
    int*  d_reach)
{
    __shared__ float s_seed[SCORE_DIM];

    if (threadIdx.x < SCORE_DIM)
        s_seed[threadIdx.x] = d_X_db[(long long)seed_idx * SCORE_DIM + threadIdx.x];
    __syncthreads();

    int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n_db) return;

    float d2 = 0.0f;
    #pragma unroll
    for (int d = 0; d < SCORE_DIM; d++) {
        float diff = d_X_db[(long long)i * SCORE_DIM + d] - s_seed[d];
        d2 += diff * diff;
    }
    if (d2 < eps2) d_reach[i] = 1;
}

/* ── Binary / CSV loading helpers ──────────────────────────────────────── */

/* load_model: reads binary model written by cuda_train.
 * Header: nc (int), nf (int), n_rff (int), ncls (int)
 * Data  : W[ncls×n_rff], b[ncls], omega[n_rff×nf], rff_b[n_rff]
 */
static bool load_model(const char* path,
                        std::vector<float>& W,     std::vector<float>& b,
                        std::vector<float>& omega, std::vector<float>& rff_b,
                        int& nc, int& nf, int& n_rff, int& ncls)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    if (fread(&nc,    sizeof(int), 1, fp) != 1 ||
        fread(&nf,    sizeof(int), 1, fp) != 1 ||
        fread(&n_rff, sizeof(int), 1, fp) != 1 ||
        fread(&ncls,  sizeof(int), 1, fp) != 1) {
        fclose(fp);
        fprintf(stderr, "Failed to read model header from %s\n", path);
        return false;
    }

    W.resize((size_t)ncls * n_rff);
    b.resize(ncls);
    omega.resize((size_t)n_rff * nf);
    rff_b.resize(n_rff);

    if (fread(W.data(),     sizeof(float), (size_t)ncls * n_rff, fp) != (size_t)(ncls * n_rff) ||
        fread(b.data(),     sizeof(float), ncls,                 fp) != (size_t)ncls             ||
        fread(omega.data(), sizeof(float), (size_t)n_rff * nf,   fp) != (size_t)(n_rff * nf)    ||
        fread(rff_b.data(), sizeof(float), n_rff,                 fp) != (size_t)n_rff) {
        fclose(fp);
        fprintf(stderr, "Failed to read model data from %s\n", path);
        return false;
    }
    fclose(fp);
    return true;
}

static bool load_csv_features_f(const char* path,
                                  std::vector<float>& out, int& nrows, int ncols)
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
        while (std::getline(ss, tok, ',')) { out.push_back(std::stof(tok)); col++; }
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

/* ── CPU DBSCAN BFS (n_db ≤ DBSCAN_GPU_THRESH) ─────────────────────────── */
static void cpu_dbscan(
    const float* X_db,
    int   n_db,
    float eps,
    int   min_samples,
    std::vector<int>& cluster_id)
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
            for (int d = 0; d < SCORE_DIM; d++) {
                float diff = X_db[(long long)i*SCORE_DIM+d] - X_db[(long long)j*SCORE_DIM+d];
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
static void gpu_dbscan(
    float* d_X_db,
    int    n_db,
    float  eps,
    int    min_samples,
    cudaStream_t stream,
    std::vector<int>& cluster_id)
{
    float eps2 = eps * eps;

    int* d_ncount;  CUDA_CHECK(cudaMalloc(&d_ncount,  n_db * sizeof(int)));
    int* d_is_core; CUDA_CHECK(cudaMalloc(&d_is_core, n_db * sizeof(int)));
    int* d_reach;   CUDA_CHECK(cudaMalloc(&d_reach,   n_db * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_ncount, 0, n_db * sizeof(int)));

    dim3 tile_dim(DBSCAN_TILE, DBSCAN_TILE);
    int  grid_n = (n_db + DBSCAN_TILE - 1) / DBSCAN_TILE;
    dim3 tile_grid(grid_n, grid_n);
    dbscan_neighbor_kernel<<<tile_grid, tile_dim, 0, stream>>>(
        d_X_db, n_db, eps2, d_ncount);

    int mc_blocks = (n_db + 255) / 256;
    mark_core_kernel<<<mc_blocks, 256, 0, stream>>>(
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

/* ── Per-GPU inference thread context ──────────────────────────────────── */
struct InferCtx {
    int    gpu_id;
    int    local_n;
    int    local_start;
    float* h_X_global;       /* pinned host pointer (full test set)     */
    float* h_W;              /* [N_CLASSIFIERS × N_RFF]                 */
    float* h_b;              /* [N_CLASSIFIERS]                         */
    float* h_omega;          /* [N_RFF × N_FEATURES]                    */
    float* h_rff_b;          /* [N_RFF]                                 */
    /* Device buffers (allocated by thread) */
    float* d_X_local;        /* [local_n × N_FEATURES] — freed after RFF */
    float* d_Z_local;        /* [local_n × N_RFF]                       */
    float* d_W;              /* [N_CLASSIFIERS × N_RFF]                 */
    float* d_b;              /* [N_CLASSIFIERS]                         */
    float* d_omega;          /* [N_RFF × N_FEATURES]                    */
    float* d_rff_b;          /* [N_RFF]                                 */
    float* d_scores;         /* [local_n × N_CLASSIFIERS]               */
    float* d_conf;           /* [local_n]                               */
    int*   d_pred;           /* [local_n]                               */
    /* Pinned host output buffers (allocated by main, filled by thread) */
    float* h_scores_out;
    float* h_conf_out;
    int*   h_pred_out;
    cudaStream_t stream;
};

static void infer_worker(InferCtx* ctx)
{
    int g = ctx->gpu_id;
    CUDA_CHECK(cudaSetDevice(g));

    for (int j = 0; j < NUM_GPUS; j++) {
        if (j == g) continue;
        int can = 0;
        cudaDeviceCanAccessPeer(&can, g, j);
        if (can) cudaDeviceEnablePeerAccess(j, 0);
    }

    CUDA_CHECK(cudaStreamCreate(&ctx->stream));

    int    ln    = ctx->local_n;
    size_t Xsz   = (size_t)ln  * N_FEATURES      * sizeof(float);
    size_t Zsz   = (size_t)ln  * N_RFF            * sizeof(float);
    size_t Wsz   = (size_t)N_CLASSIFIERS * N_RFF  * sizeof(float);
    size_t bsz   = (size_t)N_CLASSIFIERS          * sizeof(float);
    size_t Om_sz = (size_t)N_RFF * N_FEATURES      * sizeof(float);
    size_t Rb_sz = (size_t)N_RFF                   * sizeof(float);
    size_t Ssz   = (size_t)ln  * N_CLASSIFIERS    * sizeof(float);
    size_t Csz   = (size_t)ln                     * sizeof(float);
    size_t Psz   = (size_t)ln                     * sizeof(int);

    CUDA_CHECK(cudaMalloc(&ctx->d_X_local, Xsz));
    CUDA_CHECK(cudaMalloc(&ctx->d_Z_local, Zsz));
    CUDA_CHECK(cudaMalloc(&ctx->d_W,       Wsz));
    CUDA_CHECK(cudaMalloc(&ctx->d_b,       bsz));
    CUDA_CHECK(cudaMalloc(&ctx->d_omega,   Om_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_rff_b,   Rb_sz));
    CUDA_CHECK(cudaMalloc(&ctx->d_scores,  Ssz));
    CUDA_CHECK(cudaMalloc(&ctx->d_conf,    Csz));
    CUDA_CHECK(cudaMalloc(&ctx->d_pred,    Psz));

    CUDA_CHECK(cudaMemcpyAsync(ctx->d_W,      ctx->h_W,      Wsz,
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_b,      ctx->h_b,      bsz,
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_omega,  ctx->h_omega,  Om_sz,
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_rff_b,  ctx->h_rff_b,  Rb_sz,
                               cudaMemcpyHostToDevice, ctx->stream));

    const float* h_X_slice = ctx->h_X_global +
                              (size_t)ctx->local_start * N_FEATURES;
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_X_local, h_X_slice, Xsz,
                               cudaMemcpyHostToDevice, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    /* RFF transform: X[ln×N_FEATURES] → Z[ln×N_RFF] */
    {
        long long total_rff = (long long)ln * N_RFF;
        rff_transform_kernel<<<(int)((total_rff + 255) / 256), 256, 0, ctx->stream>>>(
            ctx->d_X_local, ctx->d_omega, ctx->d_rff_b, ctx->d_Z_local, ln);
        CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        CUDA_CHECK(cudaFree(ctx->d_X_local));
        ctx->d_X_local = nullptr;
    }

    /* OvO majority-vote scoring */
    int blocks = (ln + 255) / 256;
    ovo_infer_kernel<<<blocks, 256, 0, ctx->stream>>>(
        ctx->d_Z_local, ctx->d_W, ctx->d_b,
        ctx->d_scores, ctx->d_conf, ctx->d_pred, ln);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    CUDA_CHECK(cudaMemcpyAsync(ctx->h_scores_out, ctx->d_scores, Ssz,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_conf_out,   ctx->d_conf,   Csz,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaMemcpyAsync(ctx->h_pred_out,   ctx->d_pred,   Psz,
                               cudaMemcpyDeviceToHost, ctx->stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    if (ctx->d_X_local) CUDA_CHECK(cudaFree(ctx->d_X_local));
    CUDA_CHECK(cudaFree(ctx->d_Z_local));
    CUDA_CHECK(cudaFree(ctx->d_W));
    CUDA_CHECK(cudaFree(ctx->d_b));
    CUDA_CHECK(cudaFree(ctx->d_omega));
    CUDA_CHECK(cudaFree(ctx->d_rff_b));
    CUDA_CHECK(cudaFree(ctx->d_scores));
    CUDA_CHECK(cudaFree(ctx->d_conf));
    CUDA_CHECK(cudaFree(ctx->d_pred));
    CUDA_CHECK(cudaStreamDestroy(ctx->stream));
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main()
{
    nvtxRangePushA("cuda_infer:main");
    auto t_total_start = std::chrono::high_resolution_clock::now();

    /* ── Load model ── */
    fprintf(stdout, "Loading model.bin...\n");
    std::vector<float> h_W_raw, h_b_raw, h_omega_raw, h_rff_b_raw;
    int nc = 0, nf = 0, n_rff_model = 0, ncls_model = 0;
    if (!load_model("model.bin", h_W_raw, h_b_raw, h_omega_raw, h_rff_b_raw,
                    nc, nf, n_rff_model, ncls_model))
        exit(EXIT_FAILURE);
    if (nc != N_CLASSES || nf != N_FEATURES ||
        n_rff_model != N_RFF || ncls_model != N_CLASSIFIERS) {
        fprintf(stderr, "Model dims mismatch: nc=%d nf=%d n_rff=%d ncls=%d"
                        " (expected %d/%d/%d/%d)\n",
                nc, nf, n_rff_model, ncls_model,
                N_CLASSES, N_FEATURES, N_RFF, N_CLASSIFIERS);
        exit(EXIT_FAILURE);
    }
    fprintf(stdout, "Model loaded: N_CLASSES=%d N_FEATURES=%d N_RFF=%d N_CLASSIFIERS=%d\n",
            nc, nf, n_rff_model, ncls_model);

    /* Pinned copies of model parameters */
    float* h_W;
    float* h_b;
    float* h_omega;
    float* h_rff_b;
    CUDA_CHECK(cudaHostAlloc(&h_W,     (size_t)N_CLASSIFIERS * N_RFF * sizeof(float),
                              cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_b,     N_CLASSIFIERS * sizeof(float),
                              cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_omega, (size_t)N_RFF * N_FEATURES * sizeof(float),
                              cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_rff_b, N_RFF * sizeof(float),
                              cudaHostAllocPortable));
    memcpy(h_W,     h_W_raw.data(),     (size_t)N_CLASSIFIERS * N_RFF * sizeof(float));
    memcpy(h_b,     h_b_raw.data(),     N_CLASSIFIERS * sizeof(float));
    memcpy(h_omega, h_omega_raw.data(), (size_t)N_RFF * N_FEATURES * sizeof(float));
    memcpy(h_rff_b, h_rff_b_raw.data(), N_RFF * sizeof(float));

    /* ── Load test data ── */
    fprintf(stdout, "Loading test data...\n");
    std::vector<float> raw_X_test;
    std::vector<int>   raw_y_test;
    int N_test_rows = 0;
    if (!load_csv_features_f("data/processed/test_data.csv",
                              raw_X_test, N_test_rows, N_FEATURES)) exit(EXIT_FAILURE);
    if (!load_csv_labels("data/processed/test_labels.csv", raw_y_test))
        exit(EXIT_FAILURE);
    int N_test = N_test_rows;
    fprintf(stdout, "Test samples: %d\n", N_test);
    if ((int)raw_y_test.size() != N_test) {
        fprintf(stderr, "Label count mismatch\n"); exit(EXIT_FAILURE);
    }

    float* h_X_test;
    CUDA_CHECK(cudaHostAlloc(&h_X_test,
                              (size_t)N_test * N_FEATURES * sizeof(float),
                              cudaHostAllocPortable));
    memcpy(h_X_test, raw_X_test.data(),
           (size_t)N_test * N_FEATURES * sizeof(float));

    int* h_y_test;
    CUDA_CHECK(cudaHostAlloc(&h_y_test, N_test * sizeof(int),
                              cudaHostAllocPortable));
    memcpy(h_y_test, raw_y_test.data(), N_test * sizeof(int));

    /* ── Per-GPU partitions ── */
    int base_chunk = N_test / NUM_GPUS;
    int remainder  = N_test % NUM_GPUS;

    float* h_all_scores;   /* N_test × N_CLASSIFIERS */
    float* h_all_conf;     /* N_test                 */
    int*   h_all_pred;     /* N_test                 */
    CUDA_CHECK(cudaHostAlloc(&h_all_scores,
                              (size_t)N_test * N_CLASSIFIERS * sizeof(float),
                              cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_all_conf, N_test * sizeof(float),
                              cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_all_pred, N_test * sizeof(int),
                              cudaHostAllocPortable));

    /* ── Launch inference workers ── */
    InferCtx ctx[NUM_GPUS];
    memset(ctx, 0, sizeof(ctx));
    int offset = 0;
    for (int g = 0; g < NUM_GPUS; g++) {
        int ln = base_chunk + (g == 0 ? remainder : 0);
        ctx[g].gpu_id       = g;
        ctx[g].local_n      = ln;
        ctx[g].local_start  = offset;
        ctx[g].h_X_global   = h_X_test;
        ctx[g].h_W          = h_W;
        ctx[g].h_b          = h_b;
        ctx[g].h_omega      = h_omega;
        ctx[g].h_rff_b      = h_rff_b;
        ctx[g].h_scores_out = h_all_scores + (size_t)offset * N_CLASSIFIERS;
        ctx[g].h_conf_out   = h_all_conf   + offset;
        ctx[g].h_pred_out   = h_all_pred   + offset;
        offset += ln;
    }

    nvtxRangePushA("cuda_infer:parallel_inference");
    auto t_predict_start = std::chrono::high_resolution_clock::now();
    std::thread threads[NUM_GPUS];
    for (int g = 0; g < NUM_GPUS; g++)
        threads[g] = std::thread(infer_worker, &ctx[g]);
    for (int g = 0; g < NUM_GPUS; g++)
        threads[g].join();
    auto t_predict_end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    double predict_ms = std::chrono::duration<double, std::milli>(
        t_predict_end - t_predict_start).count();
    fprintf(stdout, "Inference complete on all GPUs. (%.2f ms)\n", predict_ms);

    /* GFLOPS for RFF + OvO scoring (all 4 GPUs in parallel) */
    double rff_flops  = 2.0 * N_test * N_RFF * N_FEATURES;
    double ovo_flops  = 2.0 * N_test * N_CLASSIFIERS * N_RFF;
    double predict_gflops = (rff_flops + ovo_flops) / (predict_ms * 1e-3) / 1e9;
    fprintf(stdout, "Predict GFLOPS: %.2f\n", predict_gflops);

    /* ═══════════════════════════════════════════════════════════════════════
     * PARTITION
     * ═══════════════════════════════════════════════════════════════════════ */
    nvtxRangePushA("cuda_infer:partition");

    auto is_holdout = [](int orig) { return orig == 0 || orig == 1 || orig == 6; };

    std::vector<int> final_pred(N_test, -999);
    std::vector<int> uncertain_pool;
    std::vector<int> holdout_pool;
    uncertain_pool.reserve(N_MAX_UNCERTAIN);
    holdout_pool.reserve(4096);

    for (int i = 0; i < N_test; i++) {
        if (is_holdout(h_y_test[i])) {
            holdout_pool.push_back(i);
        } else if (h_all_conf[i] < CONF_THRESHOLD) {
            if ((int)uncertain_pool.size() < N_MAX_UNCERTAIN)
                uncertain_pool.push_back(i);
            else
                final_pred[i] = INT2ORIG[h_all_pred[i]];
        } else {
            final_pred[i] = INT2ORIG[h_all_pred[i]];
        }
    }

    int n_holdout = (int)holdout_pool.size();
    std::vector<int> db_pool;
    db_pool.reserve(n_holdout + (int)uncertain_pool.size());
    for (int idx : holdout_pool)   db_pool.push_back(idx);
    for (int idx : uncertain_pool) db_pool.push_back(idx);
    int n_db = (int)db_pool.size();

    fprintf(stdout, "Partition: confident=%d  uncertain=%d  holdout=%d  n_db=%d\n",
            N_test - n_db, (int)uncertain_pool.size(), n_holdout, n_db);

    /* Build X_db: SCORE_DIM-dimensional OvO score vectors */
    std::vector<float> X_db((size_t)n_db * SCORE_DIM);
    for (int i = 0; i < n_db; i++) {
        int orig_idx = db_pool[i];
        for (int d = 0; d < SCORE_DIM; d++)
            X_db[(size_t)i * SCORE_DIM + d] =
                h_all_scores[(size_t)orig_idx * SCORE_DIM + d];
    }

    nvtxRangePop();

    /* ═══════════════════════════════════════════════════════════════════════
     * GPU DBSCAN   (GPU 0)
     * ═══════════════════════════════════════════════════════════════════════ */
    nvtxRangePushA("cuda_infer:dbscan");

    CUDA_CHECK(cudaSetDevice(0));

    std::vector<int> cluster_id(n_db, -1);

    auto t_dbscan_start = std::chrono::high_resolution_clock::now();
    if (n_db > 0) {
        if (n_db > DBSCAN_GPU_THRESH) {
            fprintf(stdout, "DBSCAN: GPU streaming BFS (n_db=%d > %d)\n",
                    n_db, DBSCAN_GPU_THRESH);
            float* d_X_db;
            CUDA_CHECK(cudaMalloc(&d_X_db, (size_t)n_db * SCORE_DIM * sizeof(float)));
            cudaStream_t dbscan_stream;
            CUDA_CHECK(cudaStreamCreate(&dbscan_stream));
            CUDA_CHECK(cudaMemcpyAsync(d_X_db, X_db.data(),
                                       (size_t)n_db * SCORE_DIM * sizeof(float),
                                       cudaMemcpyHostToDevice, dbscan_stream));
            CUDA_CHECK(cudaStreamSynchronize(dbscan_stream));
            gpu_dbscan(d_X_db, n_db, EPS, MIN_SAMPLES, dbscan_stream, cluster_id);
            CUDA_CHECK(cudaFree(d_X_db));
            CUDA_CHECK(cudaStreamDestroy(dbscan_stream));
        } else {
            fprintf(stdout, "DBSCAN: CPU BFS (n_db=%d <= %d)\n",
                    n_db, DBSCAN_GPU_THRESH);
            cpu_dbscan(X_db.data(), n_db, EPS, MIN_SAMPLES, cluster_id);
        }
    }
    auto t_dbscan_end = std::chrono::high_resolution_clock::now();
    double dbscan_ms = std::chrono::duration<double, std::milli>(
        t_dbscan_end - t_dbscan_start).count();
    fprintf(stdout, "DBSCAN complete. (%.2f ms)\n", dbscan_ms);

    nvtxRangePop();

    /* ═══════════════════════════════════════════════════════════════════════
     * CLUSTER LABELLING
     * ═══════════════════════════════════════════════════════════════════════ */
    nvtxRangePushA("cuda_infer:cluster_label");

    int max_cid = -1;
    for (int i = 0; i < n_db; i++)
        if (cluster_id[i] > max_cid) max_cid = cluster_id[i];
    int n_clusters = max_cid + 1;

    std::vector<int> cluster_class(n_clusters, -2);

    /* Normal centroid in SCORE_DIM-D space */
    std::vector<double> normal_centroid(SCORE_DIM, 0.0);
    int normal_count = 0;
    for (int i = 0; i < N_test; i++) {
        if (h_y_test[i] == NORMAL_ORIG_ID) {
            for (int d = 0; d < SCORE_DIM; d++)
                normal_centroid[d] += h_all_scores[(size_t)i * SCORE_DIM + d];
            normal_count++;
        }
    }
    if (normal_count > 0)
        for (int d = 0; d < SCORE_DIM; d++)
            normal_centroid[d] /= normal_count;

    for (int c = 0; c < n_clusters; c++) {
        std::unordered_map<int, int> holdout_votes;
        std::vector<double> centroid(SCORE_DIM, 0.0);
        int cnt = 0;
        for (int i = 0; i < n_db; i++) {
            if (cluster_id[i] != c) continue;
            for (int d = 0; d < SCORE_DIM; d++)
                centroid[d] += X_db[(size_t)i * SCORE_DIM + d];
            cnt++;
            if (i < n_holdout) {
                int orig = h_y_test[db_pool[i]];
                holdout_votes[orig]++;
            }
        }
        if (cnt > 0)
            for (int d = 0; d < SCORE_DIM; d++) centroid[d] /= cnt;

        if (!holdout_votes.empty()) {
            int best_class = -2, best_vote = 0;
            for (auto& kv : holdout_votes) {
                if (kv.second > best_vote) { best_vote = kv.second; best_class = kv.first; }
            }
            cluster_class[c] = best_class;
        } else {
            double dist2 = 0.0;
            for (int d = 0; d < SCORE_DIM; d++) {
                double diff = centroid[d] - normal_centroid[d];
                dist2 += diff * diff;
            }
            cluster_class[c] = (sqrt(dist2) < NORMAL_PROX_THRESH) ? NORMAL_ORIG_ID : -2;
        }
    }

    for (int i = 0; i < n_db; i++) {
        int orig_idx = db_pool[i];
        int cid      = cluster_id[i];
        final_pred[orig_idx] = (cid == -1) ? -1 : cluster_class[cid];
    }

    nvtxRangePop();

    /* ═══════════════════════════════════════════════════════════════════════
     * EVALUATION
     * ═══════════════════════════════════════════════════════════════════════ */
    nvtxRangePushA("cuda_infer:evaluation");

    /* Accuracy: over known classes (orig 2-5) */
    int acc_correct = 0, acc_total = 0;
    int tp[4]={}, fp[4]={}, fn_[4]={};
    for (int i = 0; i < N_test; i++) {
        int orig = h_y_test[i];
        if (orig < 2 || orig > 5) continue;
        int ko   = orig - 2;
        int pred = final_pred[i];
        acc_total++;
        if (pred >= 2 && pred <= 5) {
            int kp = pred - 2;
            if (kp == ko) { tp[ko]++; acc_correct++; }
            else { fp[kp]++; fn_[ko]++; }
        } else {
            fn_[ko]++;
        }
    }
    float accuracy = acc_total > 0 ? (float)acc_correct / acc_total : 0.0f;

    float macro_f1 = 0.0f;
    printf("\n%-15s  %8s  %8s  %8s\n", "Class","Precision","Recall","F1");
    printf("%-15s  %8s  %8s  %8s\n", "-----","------","------","------");
    const char* trained_names[4] = {"DDoS","DoS","NormalTraffic","PortScan"};
    float per_f1[4];
    for (int k = 0; k < 4; k++) {
        float prec = (tp[k]+fp[k]) > 0 ? (float)tp[k]/(tp[k]+fp[k]) : 0.0f;
        float rec  = (tp[k]+fn_[k]) > 0 ? (float)tp[k]/(tp[k]+fn_[k]) : 0.0f;
        float f1   = (prec+rec) > 0 ? 2*prec*rec/(prec+rec) : 0.0f;
        per_f1[k]  = f1;
        macro_f1  += f1;
        printf("%-15s  %8.4f  %8.4f  %8.4f  (tp=%d fp=%d fn=%d)\n",
               trained_names[k], prec, rec, f1, tp[k], fp[k], fn_[k]);
    }
    macro_f1 /= 4.0f;
    printf("%-15s  %8s  %8s  %8.4f\n", "Macro-F1", "", "", macro_f1);
    printf("%-15s  %8.4f\n", "Accuracy", accuracy);

    printf("\nHoldout class detection via DBSCAN:\n");
    printf("%-15s  %8s  %8s  %8s\n","Class","Correct","Total","Rate");
    int holdout_ids[3]   = { 0, 1, 6 };
    const char* holdout_names[3] = { "Bots", "BruteForce", "WebAttacks" };
    int total_holdout_correct = 0, total_holdout = 0;
    float holdout_rate[3];
    for (int hi = 0; hi < 3; hi++) {
        int hid = holdout_ids[hi];
        int cnt = 0, correct = 0;
        for (int i = 0; i < N_test; i++) {
            if (h_y_test[i] != hid) continue;
            cnt++;
            if (final_pred[i] == hid) correct++;
        }
        total_holdout         += cnt;
        total_holdout_correct += correct;
        holdout_rate[hi] = cnt > 0 ? (float)correct / cnt : 0.0f;
        printf("%-15s  %8d  %8d  %8.4f\n", holdout_names[hi], correct, cnt, holdout_rate[hi]);
    }
    float overall_holdout_rate = total_holdout > 0 ?
        (float)total_holdout_correct / total_holdout : 0.0f;
    printf("%-15s  %8d  %8d  %8.4f\n", "Overall",
           total_holdout_correct, total_holdout, overall_holdout_rate);

    printf("\nUncertain pool breakdown:\n");
    int unc_normal=0, unc_holdout=0, unc_novel=0, unc_unknown=0;
    for (int idx : uncertain_pool) {
        int p = final_pred[idx];
        if      (p == NORMAL_ORIG_ID)  unc_normal++;
        else if (p == -1)              unc_novel++;
        else if (p == -2)              unc_unknown++;
        else if (is_holdout(p))        unc_holdout++;
    }
    printf("  Reclassified Normal : %d\n", unc_normal);
    printf("  Matched holdout     : %d\n", unc_holdout);
    printf("  NovelAttack (-1)    : %d\n", unc_novel);
    printf("  UnknownAnomaly (-2) : %d\n", unc_unknown);

    nvtxRangePop();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(
        t_total_end - t_total_start).count();

    printf("\n=== Timing Summary ===\n");
    printf("  Predict (ms) : %.2f\n", predict_ms);
    printf("  DBSCAN  (ms) : %.2f\n", dbscan_ms);
    printf("  Total   (ms) : %.2f\n", total_ms);
    printf("  Predict GFLOPS: %.3f\n", predict_gflops);

    /* ═══════════════════════════════════════════════════════════════════════
     * OUTPUT FILES
     * ═══════════════════════════════════════════════════════════════════════ */
    nvtxRangePushA("cuda_infer:output");

    {
        std::ofstream pred_file("cuda_predictions.csv");
        if (!pred_file.is_open()) {
            fprintf(stderr, "Cannot open cuda_predictions.csv\n"); exit(EXIT_FAILURE);
        }
        pred_file << "original_label,predicted_label\n";
        for (int i = 0; i < N_test; i++)
            pred_file << h_y_test[i] << "," << final_pred[i] << "\n";
    }

    {
        std::ofstream log_file("cuda_log.csv");
        if (!log_file.is_open()) {
            fprintf(stderr, "Cannot open cuda_log.csv\n"); exit(EXIT_FAILURE);
        }
        log_file << "accuracy,macro_f1,"
                    "f1_ddos,f1_dos,f1_normal,f1_portscan,"
                    "holdout_detection_rate,"
                    "predict_ms,dbscan_ms,total_ms,predict_gflops,"
                    "n_test,n_db,n_clusters,"
                    "uncertain_normal,uncertain_holdout,"
                    "uncertain_novel,uncertain_unknown\n";
        log_file << accuracy              << ","
                 << macro_f1              << ","
                 << per_f1[0]             << ","
                 << per_f1[1]             << ","
                 << per_f1[2]             << ","
                 << per_f1[3]             << ","
                 << overall_holdout_rate  << ","
                 << predict_ms            << ","
                 << dbscan_ms             << ","
                 << total_ms              << ","
                 << predict_gflops        << ","
                 << N_test                << ","
                 << n_db                  << ","
                 << n_clusters            << ","
                 << unc_normal            << ","
                 << unc_holdout           << ","
                 << unc_novel             << ","
                 << unc_unknown           << "\n";
    }

    fprintf(stdout, "\nWrote cuda_predictions.csv and cuda_log.csv\n");

    nvtxRangePop();
    nvtxRangePop();

    /* ── Cleanup ── */
    CUDA_CHECK(cudaFreeHost(h_W));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_omega));
    CUDA_CHECK(cudaFreeHost(h_rff_b));
    CUDA_CHECK(cudaFreeHost(h_X_test));
    CUDA_CHECK(cudaFreeHost(h_y_test));
    CUDA_CHECK(cudaFreeHost(h_all_scores));
    CUDA_CHECK(cudaFreeHost(h_all_conf));
    CUDA_CHECK(cudaFreeHost(h_all_pred));

    return 0;
}
