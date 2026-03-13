/*
 * cuda_dbscan.cu  ---  Hybrid SVM + DBSCAN Anomaly Detector for CICIDS2017
 *
 * Pipeline
 * --------
 *  1. Load pre-trained SVM (best_model.bin: W[4×52], b[4]).
 *  2. Run SVM inference on the full test set -> decision scores D[N_te × 4].
 *  3. Partition test samples:
 *       holdout   : original class ∈ {0,1,6}           -> always into DBSCAN pool
 *                   (Bots, BruteForce, WebAttacks — SVM never saw them)
 *       uncertain : non-holdout AND max(D[i,:]) < CONF_THRESHOLD -> DBSCAN pool
 *       confident : non-holdout AND max(D[i,:]) >= CONF_THRESHOLD -> SVM argmax
 *
 *  4. Build DBSCAN pool X_db = concat(D[holdout], D[uncertain])  (4-dim SVM scores).
 *     Holdout samples come FIRST so their indices 0..n_holdout-1 are known.
 *
 *  5. Auto-tune eps via k-distance graph:
 *       - For every point in X_db, compute distance to its MIN_SAMPLES-th
 *         nearest non-self neighbour (GPU kernel, register-based top-k sort).
 *       - Sort those n_db values on CPU.
 *       - Knee = index of maximum second-difference on the sorted curve.
 *       - eps = k-dist value at the knee.
 *
 *  6. DBSCAN on GPU (two-pass CSR) + CPU BFS:
 *       a. count_neighbors_kernel  -> nbr_count[n_db]  (pass 1, O(n²d) GPU)
 *       b. Mark core points: is_core[i] iff nbr_count[i] >= MIN_SAMPLES
 *       c. Build CSR row_ptr from prefix-sum of nbr_count (CPU)
 *       d. fill_neighbors_kernel   -> col_idx[total_edges]  (pass 2, O(n²d) GPU)
 *       e. CPU BFS cluster expansion with CSR downloaded from GPU
 *
 *  7. Label each cluster by majority vote over its holdout members:
 *       - Cluster contains Bots/BruteForce/WebAttacks samples -> that class
 *       - No holdout members, centroid near Normal centroid    -> reclassify benign
 *       - No holdout members, centroid far from Normal         -> unknown anomaly
 *       - DBSCAN noise (-1)                                    -> novel/unknown attack
 *
 *  8. Assign final predictions to all test samples.
 *
 *  9. Evaluate: per-class precision/recall/F1, holdout detection breakdown,
 *     uncertain-sample reclassification breakdown.
 *
 * 10. Save: dbscan_clusters.csv, hybrid_predictions.csv, hybrid_log.csv
 *
 * Hardware target
 * ---------------
 *  GPU 0 only (inference + DBSCAN).  4×RTX 2080, sm_75.
 *  Typical pool size n_db ≈ 10K–80K in 4D -> all ops fit in 8 GB VRAM.
 *
 * Build
 * -----
 *  nvcc -O3 -arch=sm_75 -std=c++17 -lcublas -o cuda_dbscan cuda_dbscan.cu
 *
 * Prerequisites
 * -------------
 *  best_model.bin                      (output of cuda_svm)     [linear SVM mode]
 *  best_rff_model.bin                  (output of cuda_rff_svm) [RFF mode]
 *  best_rff_Omega.bin, best_rff_phi.bin                         [RFF mode]
 *  data/processed/test_data.csv
 *  data/processed/test_labels.csv      (original label IDs 0-6, no header)
 *
 * Switching models
 * ----------------
 *  Set USE_RFF_MODEL = true  in the constants section to use the RFF SVM.
 *  Set USE_RFF_MODEL = false (default) for the linear SVM.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Compile-time configuration
// ---------------------------------------------------------------------------
static constexpr int   N_FEATURES      = 52;
static constexpr int   N_SVM_K         = 4;    // SVM output classes
static constexpr int   N_DBSCAN_FEAT   = 4;    // feature dim fed to DBSCAN (= N_SVM_K)
static constexpr int   N_ALL_CLASSES   = 7;    // original class IDs 0-6

// Hybrid pipeline parameters
static constexpr float CONF_THRESHOLD      = 0.5f;  // SVM max-score cutoff
static constexpr int   MIN_SAMPLES         = 8;     // kept at 8: smaller values shift eps lower via k-dist graph, fragmenting clusters
static constexpr float NORMAL_PROX_THRESH  = 4.0f;  // raised 2→4: diagnostic shows unclaimed cluster min-dist=1.5, p25=3.9

// Inference chunk (limits GPU memory for D matrix during SVM pass)
static constexpr int   INF_CHUNK           = 65536;
// Safety cap on total CSR edges  (guards against eps too large)
static constexpr long  MAX_EDGES           = 200000000L;  // 200 M  (~800 MB as int)
// Cap uncertain pool size before DBSCAN.  CICIDS2017 shows ~88 K uncertain
// at CONF_THRESHOLD=0.5; with density variation that many points in 4D causes
// O(n²) edge blowup.  We keep ALL holdout (anchors) and subsample uncertain.
static constexpr int   N_MAX_UNCERTAIN     = 35000;
// Manual eps override (0.0 = auto-tune via k-distance graph).
// Useful when the auto-knee lands in the wrong place.
// Example: set to 0.10f to force eps=0.10
static constexpr float EPS_OVERRIDE        = 0.148f; // pin to original eps: auto-knee drifts lower with larger 35K pool (0.134 vs 0.148)

// ---------------------------------------------------------------------------
// RFF comparison mode
//   Set USE_RFF_MODEL = true   to load best_rff_model.bin + Omega/phi files
//   produced by cuda_rff_svm.cu and run the RBF-kernel SVM inference instead
//   of the linear SVM (best_model.bin).  Everything downstream (DBSCAN, cluster
//   labeling, evaluation) is identical; only the 4D decision scores change.
//
//   Memory for RFF mode on RTX 2080 (8 GB):
//     d_X_chunk : 8192 × 52 × 4B  =    1.6 MB
//     d_Z_chunk : 8192 × 4096 × 4B = 128   MB   <- dominant term
//     d_W       :    4 × 4096 × 4B =   64  KB
//     d_Omega   : 4096 × 52  × 4B  =  832  KB
//     total GPU 0 peak  ≈ 130 MB -> comfortably fits
// ---------------------------------------------------------------------------
static constexpr bool  USE_RFF_MODEL  = false;   // flip to true to use RFF SVM
static constexpr int   RFF_N_RFF      = 4096;    // must match cuda_rff_svm.cu N_RFF
static constexpr int   INF_CHUNK_RFF  = 8192;    // chunk for RFF path (128 MB Z buffer)

// Class metadata
// Original IDs: 0=Bots 1=BruteForce 2=DDoS 3=DoS 4=NormalTraffic 5=PortScan 6=WebAttacks
static constexpr int ORIG2INT[N_ALL_CLASSES] = { -1, -1,  0,  1,  2,  3, -1 };
static constexpr int INT2ORIG[N_SVM_K]       = {  2,  3,  4,  5 };
static constexpr int NORMAL_ORIG_ID          = 4;
static const char*   CLASS_NAMES[N_ALL_CLASSES] = {
    "Bots", "BruteForce", "DDoS", "DoS",
    "NormalTraffic", "PortScan", "WebAttacks"
};

// Final-pred sentinel values (stored in final_pred[] / CSV)
// >= 0  : original class ID (0-6)
// -1    : DBSCAN noise     -> novel / unknown attack
// -2    : unclaimed cluster far from Normal -> unknown anomaly
static constexpr int PRED_NOVEL   = -1;
static constexpr int PRED_UNKNOWN = -2;

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr) do {                                                 \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                  \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
} while (0)

#define CUBLAS_CHECK(expr) do {                                               \
    cublasStatus_t _s = (expr);                                               \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
        fprintf(stderr, "[cuBLAS ERROR] %s:%d  status=%d\n",                 \
                __FILE__, __LINE__, (int)_s);                                 \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
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
// Data loaders
// ---------------------------------------------------------------------------
static std::vector<float> load_features(const std::string& path)
{
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    std::vector<float> data;
    std::string line, tok;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        while (std::getline(ss, tok, ','))
            data.push_back(std::stof(tok));
    }
    return data;
}

static std::vector<int> load_labels(const std::string& path)
{
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    std::vector<int> labels;
    std::string line;
    while (std::getline(f, line))
        labels.push_back(std::stoi(line));
    return labels;
}

// ---------------------------------------------------------------------------
// Load SVM model (best_model.bin)
//   Format: int[2]{K,F}  +  float W[K*F]  +  float b[K]
// ---------------------------------------------------------------------------
struct SvmModel {
    int K, F;
    std::vector<float> W;   // [K × F] row-major
    std::vector<float> b;   // [K]
};

static SvmModel load_model(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    SvmModel m;
    int dims[2];
    f.read(reinterpret_cast<char*>(dims), sizeof(dims));
    m.K = dims[0]; m.F = dims[1];
    m.W.resize((size_t)m.K * m.F);
    m.b.resize(m.K);
    f.read(reinterpret_cast<char*>(m.W.data()), (size_t)m.K * m.F * sizeof(float));
    f.read(reinterpret_cast<char*>(m.b.data()), m.K * sizeof(float));
    printf("[%s] Loaded SVM model: K=%d  F=%d\n", ts().c_str(), m.K, m.F);
    return m;
}

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

// Add bias in-place: D[n][k] += b[k]
__global__ void add_bias_kernel(float* D, const float* b, int N, int K)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N * K) D[tid] += b[tid % K];
}

// Extract per-sample confidence = max(D[i,:]) and argmax SVM prediction.
__global__ void extract_conf_kernel(const float* __restrict__ D,
                                     float*       __restrict__ conf,
                                     int*         __restrict__ svm_pred,
                                     int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* row = D + (size_t)i * K;
    int   best  = 0;
    float bval  = row[0];
    for (int k = 1; k < K; ++k)
        if (row[k] > bval) { bval = row[k]; best = k; }
    conf[i]     = bval;
    svm_pred[i] = best;
}

// Compute the distance to the MIN_SAMPLES-th nearest non-self neighbour
// for each point in X_db (4-dim, row-major).
// Maintains a sorted top-MIN_SAMPLES array in registers (insertion sort).
// MIN_SAMPLES is compile-time constant = 8.
__global__ void compute_kdist_kernel(const float* __restrict__ X, int n,
                                      float*       __restrict__ kdist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi0 = X[i*4+0], xi1 = X[i*4+1],
          xi2 = X[i*4+2], xi3 = X[i*4+3];

    // Top-MIN_SAMPLES (=8) non-self distances, sorted ascending in registers
    float top[MIN_SAMPLES];
    for (int t = 0; t < MIN_SAMPLES; ++t) top[t] = 1e30f;

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        float d0 = xi0 - X[j*4+0], d1 = xi1 - X[j*4+1],
              d2 = xi2 - X[j*4+2], d3 = xi3 - X[j*4+3];
        float dist2 = d0*d0 + d1*d1 + d2*d2 + d3*d3;
        if (dist2 < top[MIN_SAMPLES - 1]) {
            int pos = MIN_SAMPLES - 1;
            while (pos > 0 && top[pos - 1] > dist2) {
                top[pos] = top[pos - 1];
                --pos;
            }
            top[pos] = dist2;
        }
    }
    // Return distance (not squared) to MIN_SAMPLES-th nearest non-self neighbour
    kdist[i] = sqrtf(top[MIN_SAMPLES - 1]);
}

// Count neighbours within eps (squared) for each point in X_db.
// Distance is computed in 4D.  Includes point i itself (dist=0 <= eps²).
__global__ void count_neighbors_kernel(const float* __restrict__ X, int n,
                                        float eps2,
                                        int*  __restrict__ nbr_count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi0 = X[i*4+0], xi1 = X[i*4+1],
          xi2 = X[i*4+2], xi3 = X[i*4+3];
    int cnt = 0;
    for (int j = 0; j < n; ++j) {
        float d0 = xi0 - X[j*4+0], d1 = xi1 - X[j*4+1],
              d2 = xi2 - X[j*4+2], d3 = xi3 - X[j*4+3];
        if (d0*d0 + d1*d1 + d2*d2 + d3*d3 <= eps2) ++cnt;
    }
    nbr_count[i] = cnt;
}

// Fill CSR column-index array (second pass, same O(n²d) scan).
// For each point i, write neighbour indices into col_idx[row_ptr[i]..].
__global__ void fill_neighbors_kernel(const float* __restrict__ X, int n,
                                       float eps2,
                                       const int* __restrict__ row_ptr,
                                       int*       __restrict__ col_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi0 = X[i*4+0], xi1 = X[i*4+1],
          xi2 = X[i*4+2], xi3 = X[i*4+3];
    int base = row_ptr[i], fill = 0;
    for (int j = 0; j < n; ++j) {
        float d0 = xi0 - X[j*4+0], d1 = xi1 - X[j*4+1],
              d2 = xi2 - X[j*4+2], d3 = xi3 - X[j*4+3];
        if (d0*d0 + d1*d1 + d2*d2 + d3*d3 <= eps2)
            col_idx[base + fill++] = j;
    }
}

// ---------------------------------------------------------------------------
// RFF transform: Z[n, d] = scale * cos(Omega[d,:] · X[n,:] + phi[d])
//   N:    batch size
//   IN_F: raw feature dimension (52)
//   D:    RFF output dimension (RFF_N_RFF)
//   scale = sqrt(2/D)
// Copied verbatim from cuda_rff_svm.cu so inference is byte-identical.
// ---------------------------------------------------------------------------
__global__ void rff_transform_kernel(
    const float* __restrict__ X,      // N × IN_F
    const float* __restrict__ Omega,  // D × IN_F
    const float* __restrict__ phi,    // D
    float*       __restrict__ Z,      // N × D
    int N, int IN_F, int D, float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)N * D) return;
    int n = (int)(idx / D);
    int d = (int)(idx % D);
    float dot = 0.f;
    const float* x  = X     + (long long)n * IN_F;
    const float* om = Omega + (long long)d * IN_F;
    for (int f = 0; f < IN_F; ++f) dot += om[f] * x[f];
    Z[idx] = scale * cosf(dot + phi[d]);
}

// ---------------------------------------------------------------------------
// Auto-tune eps: Kneedle algorithm on sorted k-distance curve.
//
// Root cause of the original failure: max second-difference is dominated by
// the extreme outlier in the tail (max=292 vs p99=1.79), so the "knee" lands
// at the very last index rather than at the true elbow.
//
// Fix:
//   1. Clip the sorted k-dist at p90 to remove the fat tail before analysis.
//   2. Apply Kneedle: on a normalised ascending curve, the elbow is the point
//      of maximum deviation BELOW the diagonal, i.e. argmax(x_norm - y_norm).
//      For a curve that stays flat long then rises steeply the maximum of
//      (x_norm - y_norm) is close to where the steep rise begins.
//   3. Clamp the result to [p10, p87] of the FULL distribution so we never
//      pick an eps below the 10th percentile (too small, most points become
//      noise) or above the 87th (too large, core-point explosion).
//   4. If EPS_OVERRIDE != 0, skip all of the above and use it directly.
// ---------------------------------------------------------------------------
static float auto_eps(std::vector<float>& kdist)
{
    int n = (int)kdist.size();
    std::sort(kdist.begin(), kdist.end());

    printf("[%s] k-distance graph (k=%d, n=%d):\n", ts().c_str(), MIN_SAMPLES, n);
    printf("  p10=%.5f  p25=%.5f  p50=%.5f  p75=%.5f  p90=%.5f  p99=%.5f  max=%.5f\n",
           (double)kdist[n/10],    (double)kdist[n/4],
           (double)kdist[n/2],     (double)kdist[3*n/4],
           (double)kdist[9*n/10],  (double)kdist[99*n/100],
           (double)kdist[n-1]);

    if (EPS_OVERRIDE > 0.f) {
        printf("  EPS_OVERRIDE active  ->  eps = %.5f\n", (double)EPS_OVERRIDE);
        return EPS_OVERRIDE;
    }

    if (n < 4) return kdist[n / 2];

    // Clip at p90 to discard the extreme fat tail
    int   n_clip = std::max(4, (int)(n * 0.90));
    float v0     = kdist[0];
    float vn     = kdist[n_clip - 1];
    float range  = vn - v0 + 1e-9f;

    // Kneedle: argmax(x_norm - y_norm)
    // x_norm = i / (n_clip-1),  y_norm = (kdist[i] - v0) / range
    // For a concave/ascending curve that stays near zero then rises:
    //   x_norm - y_norm is large in the flat region and decreases once
    //   y_norm catches up -- maximum is right at the start of the steep rise.
    int   knee    = n_clip / 2;
    float max_dev = -1e30f;
    for (int i = 0; i < n_clip; ++i) {
        float x_n = (float)i / (float)(n_clip - 1);
        float y_n = (kdist[i] - v0) / range;
        float dev = x_n - y_n;
        if (dev > max_dev) { max_dev = dev; knee = i; }
    }

    // Clamp to [p10, p87] of full distribution
    int lo = n / 10;
    int hi = (int)(n * 0.87f);
    knee = std::max(lo, std::min(knee, hi));

    printf("  Kneedle knee at index %d/%d  ->  eps = %.5f\n",
           knee, n, (double)kdist[knee]);
    printf("  To override: set EPS_OVERRIDE = %.5ff in the constants section.\n",
           (double)kdist[knee]);

    return kdist[knee];
}

// ---------------------------------------------------------------------------
// BFS cluster expansion (standard DBSCAN, CPU side)
//   Visits all neighbours of each core point; labels border points (non-core
//   but within eps of a core) with the cluster ID but does not expand them.
//   Returns number of clusters found (noise points remain labelled -1).
// ---------------------------------------------------------------------------
static int bfs_clusters(int n_db,
                         const std::vector<int>& row_ptr,
                         const std::vector<int>& col_idx,
                         const std::vector<bool>& is_core,
                         std::vector<int>& labels)
{
    labels.assign(n_db, -1);
    int n_clusters = 0;
    std::queue<int> q;

    for (int i = 0; i < n_db; ++i) {
        if (!is_core[i] || labels[i] >= 0) continue;

        labels[i] = n_clusters;
        q.push(i);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
                int v = col_idx[e];
                if (labels[v] >= 0) continue;   // already assigned
                labels[v] = n_clusters;
                if (is_core[v]) q.push(v);       // expand through core points
            }
        }
        ++n_clusters;
    }
    return n_clusters;
}

// ---------------------------------------------------------------------------
// Label clusters by holdout majority vote + Normal proximity fallback
//   cluster_names[c] = orig class ID (0,1,6)  -> known holdout class
//                    = NORMAL_ORIG_ID (4)       -> benign (near-Normal centroid)
//                    = PRED_UNKNOWN  (-2)        -> unknown anomaly
// ---------------------------------------------------------------------------
static void label_clusters(int n_clusters, int n_holdout,
                            const std::vector<int>&   db_labels,
                            const std::vector<int>&   holdout_orig,
                            const std::vector<float>& X_db_h,
                            const std::vector<float>& normal_centroid,
                            std::vector<int>& cluster_names)
{
    cluster_names.resize(n_clusters, PRED_UNKNOWN);
    if (n_clusters == 0) return;

    int n_db = (int)db_labels.size();

    // Per-cluster holdout class vote counts
    std::vector<std::unordered_map<int,int>> votes(n_clusters);
    for (int i = 0; i < n_holdout; ++i) {
        int c = db_labels[i];
        if (c < 0) continue;   // holdout point is noise
        votes[c][holdout_orig[i]]++;
    }

    // Per-cluster centroid (4D) for Normal proximity check
    std::vector<std::array<double, N_DBSCAN_FEAT>> centroid(
        n_clusters, {0.0, 0.0, 0.0, 0.0});
    std::vector<int> sz(n_clusters, 0);
    for (int i = 0; i < n_db; ++i) {
        int c = db_labels[i];
        if (c < 0) continue;
        for (int d = 0; d < N_DBSCAN_FEAT; ++d)
            centroid[c][d] += X_db_h[i * N_DBSCAN_FEAT + d];
        ++sz[c];
    }
    for (int c = 0; c < n_clusters; ++c)
        if (sz[c] > 0)
            for (int d = 0; d < N_DBSCAN_FEAT; ++d)
                centroid[c][d] /= sz[c];

    int n_by_holdout = 0, n_benign = 0, n_unknown = 0;

    for (int c = 0; c < n_clusters; ++c) {
        if (!votes[c].empty()) {
            // Majority vote from holdout members
            int best_cls = -1, best_cnt = 0;
            for (auto& [cls, cnt] : votes[c])
                if (cnt > best_cnt) { best_cnt = cnt; best_cls = cls; }
            cluster_names[c] = best_cls;
            ++n_by_holdout;
        } else {
            // No holdout anchor: check proximity to Normal class centroid
            double dist2 = 0.0;
            for (int d = 0; d < N_DBSCAN_FEAT; ++d) {
                double diff = centroid[c][d] - normal_centroid[d];
                dist2 += diff * diff;
            }
            if (dist2 < (double)NORMAL_PROX_THRESH * NORMAL_PROX_THRESH) {
                cluster_names[c] = NORMAL_ORIG_ID;  // reclassify as benign
                ++n_benign;
            } else {
                cluster_names[c] = PRED_UNKNOWN;    // novel anomaly
                ++n_unknown;
            }
        }
    }

    printf("[%s] Cluster labeling: %d via holdout vote, %d reclassified benign,"
           " %d unknown anomaly\n",
           ts().c_str(), n_by_holdout, n_benign, n_unknown);

    // Print centroid-distance distribution for unclaimed clusters to help
    // calibrate NORMAL_PROX_THRESH.  If p25 << current threshold, lower it;
    // if p50 > threshold, existing benign clusters are being labelled unknown.
    {
        std::vector<double> dists;
        dists.reserve(n_clusters);
        for (int c = 0; c < n_clusters; ++c) {
            if (!votes[c].empty()) continue;  // already labelled by holdout vote
            double dist2 = 0.0;
            for (int d = 0; d < N_DBSCAN_FEAT; ++d) {
                double diff = centroid[c][d] - normal_centroid[d];
                dist2 += diff * diff;
            }
            dists.push_back(std::sqrt(dist2));
        }
        if (!dists.empty()) {
            std::sort(dists.begin(), dists.end());
            int nd = (int)dists.size();
            printf("  Unclaimed-cluster centroid-to-Normal distances (n=%d):\n", nd);
            printf("    min=%.3f  p25=%.3f  p50=%.3f  p75=%.3f  max=%.3f\n",
                   dists[0], dists[nd/4], dists[nd/2],
                   dists[3*nd/4], dists[nd - 1]);
            printf("    NORMAL_PROX_THRESH=%.3f  (clusters with dist < thresh -> benign)\n",
                   (double)NORMAL_PROX_THRESH);
        }
    }
}

// ---------------------------------------------------------------------------
// Save DBSCAN cluster centroids for inductive inference.
//
// File: best_dbscan_model.bin  (linear SVM)
//    or best_rff_dbscan_model.bin  (RFF SVM)
//
// Format (little-endian):
//   int   nc              number of clusters
//   float eps             neighbourhood radius that was used
//   float norm_ctr[4]     Normal-class centroid in 4-D SVM-score space
//   repeat nc times:
//     int   label         orig class ID (0-6), NORMAL_ORIG_ID(4), PRED_UNKNOWN(-2)
//     float centroid[4]   mean position of all cluster members in 4-D score space
//
// Inference (nearest-centroid):
//   for each uncertain sample s (4-D SVM scores):
//     c* = argmin_c  dist(s, centroid[c])
//     if dist < eps  →  label[c*]   else  →  PRED_NOVEL
// ---------------------------------------------------------------------------
static void save_dbscan_model(const char* path,
                               int n_clusters, float eps,
                               const std::vector<float>& normal_centroid,
                               const std::vector<int>&   cluster_names,
                               const std::vector<float>& X_db_h,
                               const std::vector<int>&   db_labels,
                               int n_db)
{
    // Compute per-cluster centroids in 4-D SVM-score space
    std::vector<std::array<float, N_DBSCAN_FEAT>> ctrs(
        n_clusters, {0.f, 0.f, 0.f, 0.f});
    std::vector<int> sz(n_clusters, 0);
    for (int i = 0; i < n_db; ++i) {
        int c = db_labels[i];
        if (c < 0) continue;
        for (int d = 0; d < N_DBSCAN_FEAT; ++d)
            ctrs[c][d] += X_db_h[(size_t)i * N_DBSCAN_FEAT + d];
        ++sz[c];
    }
    for (int c = 0; c < n_clusters; ++c)
        if (sz[c] > 0)
            for (int d = 0; d < N_DBSCAN_FEAT; ++d)
                ctrs[c][d] /= sz[c];

    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[save_dbscan] Cannot open %s\n", path); return; }
    f.write(reinterpret_cast<const char*>(&n_clusters),          sizeof(int));
    f.write(reinterpret_cast<const char*>(&eps),                  sizeof(float));
    f.write(reinterpret_cast<const char*>(normal_centroid.data()), N_DBSCAN_FEAT * sizeof(float));
    for (int c = 0; c < n_clusters; ++c) {
        f.write(reinterpret_cast<const char*>(&cluster_names[c]), sizeof(int));
        f.write(reinterpret_cast<const char*>(ctrs[c].data()),    N_DBSCAN_FEAT * sizeof(float));
    }
    printf("[%s] Saved DBSCAN model -> %s  (%d clusters, eps=%.5f)\n",
           ts().c_str(), path, n_clusters, (double)eps);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    // ── 1. Load SVM model ────────────────────────────────────────────────
    const char* model_path = USE_RFF_MODEL ? "best_rff_model.bin" : "best_model.bin";
    SvmModel model = load_model(model_path);
    assert(model.K == N_SVM_K);
    const int model_F = model.F;  // 52 (linear SVM) or 4096 (RFF SVM)
    printf("[%s] Model: K=%d  F=%d  RFF_mode=%s\n",
           ts().c_str(), model.K, model_F, USE_RFF_MODEL ? "yes" : "no");

    // ── 2. Load test data ────────────────────────────────────────────────
    printf("[%s] Loading test data...\n", ts().c_str());
    auto h_Xte      = load_features("data/processed/test_data.csv");
    auto h_yte_orig = load_labels("data/processed/test_labels.csv"); // original 0-6
    const int N_te  = (int)h_yte_orig.size();
    assert((int)h_Xte.size() == (size_t)N_te * N_FEATURES);  // raw 52-dim always
    printf("[%s] N_test = %d\n", ts().c_str(), N_te);

    // ── Load RFF projection params (only when USE_RFF_MODEL=true) ────────
    float *d_Omega = nullptr, *d_phi_rff = nullptr;
    if (USE_RFF_MODEL) {
        // Omega.bin: int[2]{D, IN_F}  +  float Omega[D × IN_F]
        std::ifstream fo("best_rff_Omega.bin", std::ios::binary);
        if (!fo) { fprintf(stderr, "Cannot open best_rff_Omega.bin\n"); exit(1); }
        int odims[2];
        fo.read(reinterpret_cast<char*>(odims), sizeof(odims));
        assert(odims[0] == RFF_N_RFF && odims[1] == N_FEATURES);
        std::vector<float> h_Omega((long long)RFF_N_RFF * N_FEATURES);
        fo.read(reinterpret_cast<char*>(h_Omega.data()),
                (long long)RFF_N_RFF * N_FEATURES * sizeof(float));

        // phi.bin: int D  +  float phi[D]
        std::ifstream fp("best_rff_phi.bin", std::ios::binary);
        if (!fp) { fprintf(stderr, "Cannot open best_rff_phi.bin\n"); exit(1); }
        int pD; fp.read(reinterpret_cast<char*>(&pD), sizeof(int));
        assert(pD == RFF_N_RFF);
        std::vector<float> h_phi(RFF_N_RFF);
        fp.read(reinterpret_cast<char*>(h_phi.data()), RFF_N_RFF * sizeof(float));

        CUDA_CHECK(cudaMalloc(&d_Omega,   (long long)RFF_N_RFF * N_FEATURES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_phi_rff, RFF_N_RFF * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Omega,   h_Omega.data(),
                              (long long)RFF_N_RFF * N_FEATURES * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_phi_rff, h_phi.data(),
                              RFF_N_RFF * sizeof(float), cudaMemcpyHostToDevice));
        printf("[%s] RFF params loaded: Omega[%d×%d]  phi[%d]\n",
               ts().c_str(), RFF_N_RFF, N_FEATURES, RFF_N_RFF);
    }

    // Internal SVM labels: holdout classes -> -1
    std::vector<int> h_yte_int(N_te);
    for (int i = 0; i < N_te; ++i) {
        int o = h_yte_orig[i];
        h_yte_int[i] = (o >= 0 && o < N_ALL_CLASSES) ? ORIG2INT[o] : -1;
    }

    // ── 3. SVM inference on GPU (linear or RFF) ──────────────────────────
    printf("[%s] %s inference (GPU 0)...\n",
           ts().c_str(), USE_RFF_MODEL ? "RFF-SVM" : "Linear-SVM");

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Upload model
    float *d_W, *d_b;
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)N_SVM_K * model_F * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N_SVM_K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W, model.W.data(),
                          (size_t)N_SVM_K * model_F * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, model.b.data(),
                          N_SVM_K * sizeof(float), cudaMemcpyHostToDevice));

    // Inference scratch buffers
    // Use smaller chunks in RFF mode to cap Z buffer at ~128 MB
    const int inf_chunk = USE_RFF_MODEL ? INF_CHUNK_RFF : INF_CHUNK;

    float *d_X_chunk, *d_Z_chunk, *d_D_chunk, *d_conf_chunk;
    int   *d_pred_chunk;
    // Raw input: always 52-dim
    CUDA_CHECK(cudaMalloc(&d_X_chunk,    (size_t)inf_chunk * N_FEATURES * sizeof(float)));
    // RFF output buffer (inf_chunk × model_F); nullptr in linear mode
    if (USE_RFF_MODEL)
        CUDA_CHECK(cudaMalloc(&d_Z_chunk, (size_t)inf_chunk * model_F * sizeof(float)));
    else
        d_Z_chunk = nullptr;
    CUDA_CHECK(cudaMalloc(&d_D_chunk,    (size_t)inf_chunk * N_SVM_K   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conf_chunk, inf_chunk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pred_chunk, inf_chunk * sizeof(int)));

    // In linear mode sgemm reads raw X; in RFF mode it reads Z (post-transform)
    float* d_sgemm_in = USE_RFF_MODEL ? d_Z_chunk : d_X_chunk;

    std::vector<float> h_D       ((size_t)N_te * N_SVM_K);
    std::vector<float> h_conf    (N_te);
    std::vector<int>   h_svm_pred(N_te);

    for (int off = 0; off < N_te; off += inf_chunk) {
        int chunk = std::min(inf_chunk, N_te - off);

        CUDA_CHECK(cudaMemcpy(d_X_chunk,
                              h_Xte.data() + (size_t)off * N_FEATURES,
                              (size_t)chunk * N_FEATURES * sizeof(float),
                              cudaMemcpyHostToDevice));

        if (USE_RFF_MODEL) {
            // Project raw 52-dim → 4096-dim RFF space
            long long total = (long long)chunk * model_F;
            int th = 256, bl = (int)((total + th - 1) / th);
            rff_transform_kernel<<<bl, th>>>(
                d_X_chunk, d_Omega, d_phi_rff,
                d_Z_chunk,
                chunk, N_FEATURES, model_F, sqrtf(2.f / model_F));
        }

        // D[chunk×K] = sgemm_in[chunk×model_F] @ W^T[model_F×K]
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N_SVM_K, chunk, model_F,
            &alpha, d_W,         model_F,
                    d_sgemm_in,  model_F,
            &beta,  d_D_chunk,   N_SVM_K));

        const int th = 256;
        add_bias_kernel   <<<(chunk * N_SVM_K + th - 1) / th, th>>>(
            d_D_chunk, d_b, chunk, N_SVM_K);
        extract_conf_kernel<<<(chunk + th - 1) / th, th>>>(
            d_D_chunk, d_conf_chunk, d_pred_chunk, chunk, N_SVM_K);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_D.data()        + (size_t)off * N_SVM_K,
                              d_D_chunk,
                              (size_t)chunk * N_SVM_K * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conf.data()     + off, d_conf_chunk,
                              chunk * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_svm_pred.data() + off, d_pred_chunk,
                              chunk * sizeof(int),   cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_X_chunk));
    if (d_Z_chunk) CUDA_CHECK(cudaFree(d_Z_chunk));
    CUDA_CHECK(cudaFree(d_D_chunk));
    CUDA_CHECK(cudaFree(d_conf_chunk));
    CUDA_CHECK(cudaFree(d_pred_chunk));
    if (d_Omega)   CUDA_CHECK(cudaFree(d_Omega));
    if (d_phi_rff) CUDA_CHECK(cudaFree(d_phi_rff));

    // ── 4. Partition test samples ─────────────────────────────────────────
    auto is_holdout_class = [](int orig) {
        return orig == 0 || orig == 1 || orig == 6;
    };

    std::vector<int> holdout_idx;    // indices into test set (holdout)
    std::vector<int> uncertain_idx;  // indices into test set (uncertain, non-holdout)

    for (int i = 0; i < N_te; ++i) {
        if (is_holdout_class(h_yte_orig[i])) {
            holdout_idx.push_back(i);
        } else if (h_conf[i] < CONF_THRESHOLD) {
            uncertain_idx.push_back(i);
        }
    }

    int n_holdout   = (int)holdout_idx.size();
    int n_uncertain = (int)uncertain_idx.size();
    int n_confident = N_te - n_holdout - n_uncertain;

    printf("[%s] Partition (CONF_THRESHOLD=%.3f):\n",
           ts().c_str(), (double)CONF_THRESHOLD);
    printf("  holdout (Bots/BruteForce/WebAttacks) : %d\n", n_holdout);
    printf("  uncertain (non-holdout, low conf)    : %d  (raw)\n", n_uncertain);
    printf("  confident (SVM, high conf)           : %d\n", n_confident);

    // ── Subsample uncertain pool ──────────────────────────────────────────
    // CICIDS2017 at CONF_THRESHOLD=0.5 produces ~88 K uncertain samples.
    // Many are Normal-boundary traffic that forms an *extremely* dense cluster
    // in 4D SVM-score space; any reasonable eps results in O(n²) CSR edges
    // and OOM.  We keep ALL holdout anchors and cap uncertain at N_MAX_UNCERTAIN.
    if (n_uncertain > N_MAX_UNCERTAIN) {
        std::mt19937 rng(42);
        std::shuffle(uncertain_idx.begin(), uncertain_idx.end(), rng);
        uncertain_idx.resize(N_MAX_UNCERTAIN);
        n_uncertain = N_MAX_UNCERTAIN;
        printf("  [subsample] uncertain capped to %d\n", N_MAX_UNCERTAIN);
    }

    int n_db = n_holdout + n_uncertain;
    printf("  total DBSCAN pool (n_db)             : %d\n", n_db);

    if (n_db == 0) {
        fprintf(stderr, "[ERROR] DBSCAN pool is empty. "
                        "Lower CONF_THRESHOLD or check model output.\n");
        return 1;
    }

    // ── 5. Build X_db: 4D SVM scores, holdout first ──────────────────────
    std::vector<float> X_db_h((size_t)n_db * N_DBSCAN_FEAT);

    auto copy_scores = [&](int db_pos, int te_idx) {
        for (int k = 0; k < N_SVM_K; ++k)
            X_db_h[(size_t)db_pos * N_DBSCAN_FEAT + k] =
                h_D[(size_t)te_idx * N_SVM_K + k];
    };
    for (int i = 0; i < n_holdout;   ++i) copy_scores(i,            holdout_idx[i]);
    for (int i = 0; i < n_uncertain; ++i) copy_scores(n_holdout + i, uncertain_idx[i]);

    // True original labels for holdout points (for cluster voting)
    std::vector<int> holdout_orig(n_holdout);
    for (int i = 0; i < n_holdout; ++i)
        holdout_orig[i] = h_yte_orig[holdout_idx[i]];

    // Normal-class centroid for DBSCAN cluster relabeling.
    //
    // WHY NOT use all test Normal samples:
    //   Confident Normal samples (628 K) have scores ≈ [-9.6, -8.7, 3.1, -5.2].
    //   Uncertain boundary Normal samples have max score < 0.5, so scores ≈ [0, 0, ~0.3, 0].
    //   L2 gap between the two regions ≈ 14 units.  NORMAL_PROX_THRESH=1.0 (or even 5.0)
    //   would never fire → 0 reclassified benign, 127 unknown anomaly (observed bug).
    //
    // FIX: compute centroid only from uncertain-pool samples whose SVM argmax = Normal
    //   (internal class 2).  These are the true boundary Normal samples already in
    //   the DBSCAN pool.  Their centroid lives near [0, 0, ~0.3, 0], which is exactly
    //   where uncertain Normal clusters land.  NORMAL_PROX_THRESH=2.0 then works.
    static constexpr int NORMAL_INT = 2;  // INT2ORIG[2] = 4 = NORMAL_ORIG_ID
    std::array<double, N_DBSCAN_FEAT> normal_sum = {0.0, 0.0, 0.0, 0.0};
    int n_normal = 0;
    for (int ui = 0; ui < n_uncertain; ++ui) {
        int te_idx = uncertain_idx[ui];
        if (h_svm_pred[te_idx] == NORMAL_INT) {
            for (int k = 0; k < N_SVM_K; ++k)
                normal_sum[k] += h_D[(size_t)te_idx * N_SVM_K + k];
            ++n_normal;
        }
    }
    std::vector<float> normal_centroid(N_DBSCAN_FEAT, 0.f);
    if (n_normal > 0)
        for (int k = 0; k < N_DBSCAN_FEAT; ++k)
            normal_centroid[k] = (float)(normal_sum[k] / n_normal);
    printf("[%s] Normal centroid (uncertain-pool argmax=Normal, n=%d): "
           "[%.3f, %.3f, %.3f, %.3f]\n",
           ts().c_str(), n_normal,
           (double)normal_centroid[0], (double)normal_centroid[1],
           (double)normal_centroid[2], (double)normal_centroid[3]);

    // ── 6. Upload X_db to GPU ────────────────────────────────────────────
    float* d_Xdb;
    CUDA_CHECK(cudaMalloc(&d_Xdb, (size_t)n_db * N_DBSCAN_FEAT * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Xdb, X_db_h.data(),
                          (size_t)n_db * N_DBSCAN_FEAT * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ── 7. Auto-tune eps via k-distance graph ─────────────────────────────
    printf("[%s] Computing k-distance graph (k=%d, n_db=%d)...\n",
           ts().c_str(), MIN_SAMPLES, n_db);

    float* d_kdist;
    CUDA_CHECK(cudaMalloc(&d_kdist, n_db * sizeof(float)));
    {
        int th = 256;
        compute_kdist_kernel<<<(n_db + th - 1) / th, th>>>(d_Xdb, n_db, d_kdist);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<float> h_kdist(n_db);
    CUDA_CHECK(cudaMemcpy(h_kdist.data(), d_kdist, n_db * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_kdist));

    float eps  = auto_eps(h_kdist);  // sorts h_kdist in place
    // Save percentile handles for the overflow error message below
    float h_kdist_p10 = h_kdist[n_db / 10];
    float h_kdist_p25 = h_kdist[n_db / 4];
    float eps2 = eps * eps;
    printf("[%s] DBSCAN  eps=%.5f  min_samples=%d\n",
           ts().c_str(), (double)eps, MIN_SAMPLES);

    // ── 8. DBSCAN Pass 1: count neighbours ───────────────────────────────
    printf("[%s] DBSCAN pass 1: counting neighbours...\n", ts().c_str());

    int* d_nbr_cnt;
    CUDA_CHECK(cudaMalloc(&d_nbr_cnt, n_db * sizeof(int)));
    {
        int th = 256;
        count_neighbors_kernel<<<(n_db + th - 1) / th, th>>>(
            d_Xdb, n_db, eps2, d_nbr_cnt);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<int> h_nbr_cnt(n_db);
    CUDA_CHECK(cudaMemcpy(h_nbr_cnt.data(), d_nbr_cnt,
                          n_db * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_nbr_cnt));

    // Mark core points
    std::vector<bool> is_core(n_db);
    int n_core = 0;
    for (int i = 0; i < n_db; ++i) {
        is_core[i] = (h_nbr_cnt[i] >= MIN_SAMPLES);
        if (is_core[i]) ++n_core;
    }
    printf("  Core points: %d / %d  (%.1f%%)\n",
           n_core, n_db, 100.0 * n_core / n_db);

    // ── Build CSR row_ptr: long pre-check BEFORE touching int array ──────────
    // Bug fixed: the original code did `std::vector<int> h_row_ptr` and then
    // read h_row_ptr[n_db] as `long total_edges`.  With n_db=92K and each point
    // having ~92K neighbours the sum exceeds INT_MAX, silently wraps to a large
    // negative number, passes the "> MAX_EDGES" guard and causes OOM on cudaMalloc.
    // Fix: accumulate in `long` first; only build the int row_ptr if safe.
    {
        long acc = 0;
        for (int i = 0; i < n_db; ++i) {
            acc += h_nbr_cnt[i];
            if (acc > MAX_EDGES) {
                fprintf(stderr,
                    "\n[ERROR] Projected total CSR edges (%ld+) exceeds MAX_EDGES=%ld.\n"
                    "  eps=%.5f is too large for this pool density.\n"
                    "  Options:\n"
                    "    1. Set EPS_OVERRIDE to a smaller value (e.g. %.5f or %.5f).\n"
                    "    2. Lower N_MAX_UNCERTAIN (currently %d).\n"
                    "  k-dist percentiles were printed above.\n",
                    acc, MAX_EDGES, (double)eps,
                    (double)h_kdist_p25, (double)h_kdist_p10,
                    N_MAX_UNCERTAIN);
                CUDA_CHECK(cudaFree(d_Xdb));
                return 1;
            }
        }
    }

    // Safe to build int row_ptr
    std::vector<int> h_row_ptr(n_db + 1, 0);
    for (int i = 0; i < n_db; ++i)
        h_row_ptr[i + 1] = h_row_ptr[i] + h_nbr_cnt[i];
    long total_edges = (long)h_row_ptr[n_db];

    printf("  Total adjacency edges: %ld  (avg %.1f/point)\n",
           total_edges, (double)total_edges / n_db);

    // ── 9. DBSCAN Pass 2: fill CSR neighbour indices ──────────────────────
    printf("[%s] DBSCAN pass 2: filling CSR adjacency...\n", ts().c_str());

    int *d_row_ptr, *d_col_idx;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n_db + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, total_edges  * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr.data(),
                          (n_db + 1) * sizeof(int), cudaMemcpyHostToDevice));
    {
        int th = 256;
        fill_neighbors_kernel<<<(n_db + th - 1) / th, th>>>(
            d_Xdb, n_db, eps2, d_row_ptr, d_col_idx);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<int> h_col_idx(total_edges);
    CUDA_CHECK(cudaMemcpy(h_col_idx.data(), d_col_idx,
                          total_edges * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_Xdb));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));

    // ── 10. BFS cluster expansion (CPU) ───────────────────────────────────
    printf("[%s] BFS cluster expansion...\n", ts().c_str());

    std::vector<int> db_labels;
    int n_clusters = bfs_clusters(n_db, h_row_ptr, h_col_idx, is_core, db_labels);

    int n_noise = 0;
    for (int i = 0; i < n_db; ++i) if (db_labels[i] < 0) ++n_noise;

    std::vector<int> cluster_sz(n_clusters, 0);
    for (int i = 0; i < n_db; ++i)
        if (db_labels[i] >= 0) ++cluster_sz[db_labels[i]];

    printf("  Clusters found : %d\n", n_clusters);
    printf("  Noise points   : %d / %d  (%.1f%%)\n",
           n_noise, n_db, 100.0 * n_noise / n_db);
    if (n_clusters > 0) {
        int sz_min = *std::min_element(cluster_sz.begin(), cluster_sz.end());
        int sz_max = *std::max_element(cluster_sz.begin(), cluster_sz.end());
        double sz_mean = std::accumulate(cluster_sz.begin(), cluster_sz.end(), 0.0) / n_clusters;
        printf("  Cluster sizes  : min=%d  max=%d  mean=%.1f\n",
               sz_min, sz_max, sz_mean);
    }

    // ── 11. Label clusters by holdout majority vote ───────────────────────
    printf("[%s] Labeling clusters...\n", ts().c_str());

    std::vector<int> cluster_names;
    label_clusters(n_clusters, n_holdout, db_labels, holdout_orig,
                   X_db_h, normal_centroid, cluster_names);

    // ── 12. Assign final predictions ─────────────────────────────────────
    // final_pred[i]: original class ID (0-6), PRED_NOVEL, or PRED_UNKNOWN
    std::vector<int> final_pred(N_te, PRED_NOVEL);

    // Confident non-holdout: SVM argmax -> map to original class ID
    for (int i = 0; i < N_te; ++i) {
        if (!is_holdout_class(h_yte_orig[i]) && h_conf[i] >= CONF_THRESHOLD)
            final_pred[i] = INT2ORIG[h_svm_pred[i]];
    }

    // Uncertain: use DBSCAN cluster label
    for (int ui = 0; ui < n_uncertain; ++ui) {
        int te_idx = uncertain_idx[ui];
        int db_pos = n_holdout + ui;
        int cid    = db_labels[db_pos];

        if (cid < 0) {
            final_pred[te_idx] = PRED_NOVEL;    // noise -> novel attack
        } else {
            int cname = cluster_names[cid];
            // cname is either an orig class ID (0-6), NORMAL_ORIG_ID, or PRED_UNKNOWN
            final_pred[te_idx] = cname;
        }
    }

    // Holdout: also assign from DBSCAN (for reporting; these are "unknown" at inference time)
    for (int hi = 0; hi < n_holdout; ++hi) {
        int te_idx = holdout_idx[hi];
        int cid    = db_labels[hi];
        if (cid < 0) {
            final_pred[te_idx] = PRED_NOVEL;
        } else {
            final_pred[te_idx] = cluster_names[cid];
        }
    }

    // ── 13. Evaluation ────────────────────────────────────────────────────
    printf("\n[%s] ══════════ EVALUATION ══════════\n", ts().c_str());

    // (A) SVM confident accuracy (only known-class samples above threshold)
    long cm[N_SVM_K][N_SVM_K] = {};
    long conf_total = 0;
    for (int i = 0; i < N_te; ++i) {
        int ti = h_yte_int[i];
        if (ti < 0 || h_conf[i] < CONF_THRESHOLD) continue;
        int pi = ORIG2INT[final_pred[i] >= 0 && final_pred[i] < N_ALL_CLASSES
                          ? final_pred[i] : INT2ORIG[0]];
        if (pi < 0) continue;
        cm[ti][pi]++;
        ++conf_total;
    }
    long conf_correct = 0;
    for (int k = 0; k < N_SVM_K; ++k) conf_correct += cm[k][k];
    printf("\n  SVM confident accuracy: %.4f  (%ld / %ld)\n",
           conf_total > 0 ? (double)conf_correct / conf_total : 0.0,
           conf_correct, conf_total);

    // Per-class P/R/F1
    printf("\n  Per-class metrics (SVM-known classes, confident samples):\n");
    printf("  %-16s  %8s  %8s  %8s  %8s\n",
           "Class", "Prec", "Recall", "F1", "Support");
    float macro_p = 0, macro_r = 0, macro_f1 = 0;
    for (int k = 0; k < N_SVM_K; ++k) {
        long tp = cm[k][k], fp = 0, fn = 0;
        for (int j = 0; j < N_SVM_K; ++j) {
            if (j != k) fp += cm[j][k];
            if (j != k) fn += cm[k][j];
        }
        float p  = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.f;
        float r  = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.f;
        float f1 = (p + r > 1e-9f) ? 2.f * p * r / (p + r) : 0.f;
        macro_p  += p; macro_r  += r; macro_f1  += f1;
        printf("  %-16s  %8.4f  %8.4f  %8.4f  %8ld\n",
               CLASS_NAMES[INT2ORIG[k]], (double)p, (double)r, (double)f1, tp + fn);
    }
    macro_p /= N_SVM_K; macro_r /= N_SVM_K; macro_f1 /= N_SVM_K;
    printf("  %-16s  %8.4f  %8.4f  %8.4f\n",
           "macro", (double)macro_p, (double)macro_r, (double)macro_f1);

    // (B) Holdout detection: how well does DBSCAN identify each holdout class?
    printf("\n  Holdout detection (DBSCAN clustering):\n");
    printf("  %-16s  %8s  %8s  %8s\n", "Class", "Total", "Correct", "Noise");
    for (int hc : {0, 1, 6}) {
        long total = 0, correct = 0, noise = 0;
        for (int hi = 0; hi < n_holdout; ++hi) {
            if (holdout_orig[hi] != hc) continue;
            ++total;
            int c = db_labels[hi];
            if (c < 0) { ++noise; continue; }
            if (cluster_names[c] == hc) ++correct;
        }
        printf("  %-16s  %8ld  %8ld  %8ld\n",
               CLASS_NAMES[hc], total, correct, noise);
    }

    // (C) Uncertain sample reclassification breakdown
    long unc_novel = 0, unc_unknown = 0, unc_benign = 0;
    long unc_holdout_cls = 0, unc_svm_cls = 0;
    for (int ui = 0; ui < n_uncertain; ++ui) {
        int p = final_pred[uncertain_idx[ui]];
        if      (p == PRED_NOVEL)    ++unc_novel;
        else if (p == PRED_UNKNOWN)  ++unc_unknown;
        else if (p == NORMAL_ORIG_ID) ++unc_benign;
        else if (p == 0 || p == 1 || p == 6) ++unc_holdout_cls;
        else    ++unc_svm_cls;
    }
    printf("\n  Uncertain sample reclassification (%d total):\n", n_uncertain);
    printf("  %-38s  %d\n", "Reclassified Normal (near-Normal cluster)", (int)unc_benign);
    printf("  %-38s  %d\n", "Matched holdout pattern (Bots/BF/WebAtk)", (int)unc_holdout_cls);
    printf("  %-38s  %d\n", "Known SVM class (near-anchor cluster)",    (int)unc_svm_cls);
    printf("  %-38s  %d\n", "Novel attack (DBSCAN noise -1)",           (int)unc_novel);
    printf("  %-38s  %d\n", "Unknown anomaly (unclaimed cluster)",       (int)unc_unknown);

    // ── 14. Save results ──────────────────────────────────────────────────
    printf("\n[%s] Saving results...\n", ts().c_str());

    // dbscan_clusters.csv
    {
        std::vector<int> holdout_in_cluster(n_clusters, 0);
        for (int i = 0; i < n_holdout; ++i)
            if (db_labels[i] >= 0) ++holdout_in_cluster[db_labels[i]];

        std::ofstream f("dbscan_clusters.csv");
        f << "cluster_id,assigned_orig_class,class_name,size,holdout_members\n";
        for (int c = 0; c < n_clusters; ++c) {
            int cls = cluster_names[c];
            const char* cname =
                (cls >= 0 && cls < N_ALL_CLASSES) ? CLASS_NAMES[cls] :
                (cls == PRED_UNKNOWN)              ? "UnknownAnomaly"  :
                                                     "NovelAttack";
            f << c      << ","
              << cls     << ","
              << cname   << ","
              << cluster_sz[c] << ","
              << holdout_in_cluster[c] << "\n";
        }
        printf("  -> dbscan_clusters.csv  (%d clusters)\n", n_clusters);
    }

    // hybrid_predictions.csv
    {
        // Build te_idx -> db_pos lookup for CSV
        std::unordered_map<int,int> te_to_dbpos;
        te_to_dbpos.reserve(n_db);
        for (int i = 0; i < n_holdout;   ++i) te_to_dbpos[holdout_idx[i]]   = i;
        for (int i = 0; i < n_uncertain; ++i) te_to_dbpos[uncertain_idx[i]] = n_holdout + i;

        std::ofstream f("hybrid_predictions.csv");
        f << "te_idx,true_orig,true_class,"
             "svm_pred_int,confidence,route,"
             "dbscan_cluster,final_pred_orig,final_class\n";

        for (int i = 0; i < N_te; ++i) {
            int  true_o   = h_yte_orig[i];
            int  svm_k    = h_svm_pred[i];
            float conf    = h_conf[i];
            int  fp       = final_pred[i];

            const char* true_name = (true_o >= 0 && true_o < N_ALL_CLASSES)
                                    ? CLASS_NAMES[true_o] : "Unknown";
            const char* fp_name   = (fp >= 0 && fp < N_ALL_CLASSES)
                                    ? CLASS_NAMES[fp]
                                    : (fp == PRED_NOVEL   ? "NovelAttack"
                                    : (fp == PRED_UNKNOWN ? "UnknownAnomaly"
                                    : "?"));

            // Route: SVM | DBSCAN_holdout | DBSCAN_uncertain
            const char* route = is_holdout_class(true_o) ? "DBSCAN_holdout"
                              : (conf < CONF_THRESHOLD)   ? "DBSCAN_uncertain"
                              :                             "SVM";

            auto it = te_to_dbpos.find(i);
            int db_cluster = (it != te_to_dbpos.end())
                             ? db_labels[it->second] : -99;  // -99 = not in pool

            f << i        << "," << true_o   << "," << true_name  << ","
              << svm_k    << "," << conf     << "," << route      << ","
              << db_cluster << "," << fp      << "," << fp_name    << "\n";
        }
        printf("  -> hybrid_predictions.csv  (%d rows)\n", N_te);
    }

    // hybrid_log.csv
    {
        std::ofstream f("hybrid_log.csv");
        f << "metric,value\n"
          << "n_test,"            << N_te              << "\n"
          << "conf_threshold,"    << CONF_THRESHOLD     << "\n"
          << "min_samples,"       << MIN_SAMPLES        << "\n"
          << "eps,"               << eps                << "\n"
          << "normal_prox_thresh,"<< NORMAL_PROX_THRESH << "\n"
          << "n_holdout,"         << n_holdout          << "\n"
          << "n_uncertain,"       << n_uncertain        << "\n"
          << "n_confident,"       << n_confident        << "\n"
          << "n_db,"              << n_db               << "\n"
          << "n_clusters,"        << n_clusters         << "\n"
          << "n_core_points,"     << n_core             << "\n"
          << "n_noise_db,"        << n_noise            << "\n"
          << "svm_confident_acc," << (conf_total > 0 ? (double)conf_correct / conf_total : 0.0) << "\n"
          << "macro_precision,"   << macro_p            << "\n"
          << "macro_recall,"      << macro_r            << "\n"
          << "macro_f1,"          << macro_f1           << "\n";
        printf("  -> hybrid_log.csv\n");
    }

    // ── 15. Save DBSCAN cluster model ────────────────────────────────────
    {
        const char* dbscan_path = USE_RFF_MODEL ? "best_rff_dbscan_model.bin"
                                                : "best_dbscan_model.bin";
        save_dbscan_model(dbscan_path, n_clusters, eps, normal_centroid,
                          cluster_names, X_db_h, db_labels, n_db);
    }

    // ── Cleanup ────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_b));
    CUBLAS_CHECK(cublasDestroy(cublas));

    printf("\n[%s] Done.\n", ts().c_str());
    return 0;
}
