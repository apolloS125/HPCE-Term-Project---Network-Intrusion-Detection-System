/*
 * cuda_hybrid.cu  ---  Hybrid RFF-SVM + DBSCAN inference pipeline
 *
 * Pipeline:
 * ──────────────────────────────────────────────────────────────────────────
 *  1. RFF-SVM inference  →  D[N × 4]  (decision_function scores)
 *     RFF projection: Z = sqrt(2/D) * cos(Omega @ x + phi)  (52-D → 4096-D)
 *     conf = max(D[i,:])   ← confidence for each sample
 *
 *  2. Partition:
 *       original class ∈ {0,1,6}   → holdout pool  (Bots/BruteForce/WebAttacks)
 *       conf < CONF_THRESHOLD       → uncertain pool (SVM not confident)
 *       conf >= CONF_THRESHOLD      → SVM argmax    (confident, skip DBSCAN)
 *
 *  3. DBSCAN pool  = [holdout | uncertain_subsample]  in 4-D SVM-score space
 *       Holdout always first → their indices 0..n_holdout-1 are known anchors
 *       eps = 0.148  (fixed override, tuned for CICIDS2017 at N_MAX_UNCERTAIN=35K)
 *       min_samples = 8
 *
 *  4. Label each cluster by majority vote of its holdout members:
 *       has holdout members                  →  that class (0=Bots / 1=BruteForce / 6=WebAttacks)
 *       no holdout, centroid near Normal      →  NormalTraffic  (reclassify benign)
 *       no holdout, centroid far from Normal  →  UnknownAnomaly
 *       DBSCAN noise point (-1)              →  NovelAttack
 *
 * Files required:
 *   best_rff_model.bin    (RFF SVM: W[4×4096], b[4])
 *   best_rff_Omega.bin    (Omega[4096×52])
 *   best_rff_phi.bin      (phi[4096])
 *   best_rff_dbscan_model.bin (DBSCAN centroids — written at end of run)
 *   data/processed/mini_test_data.csv    ← when USE_MINI_TEST=true  (run make_mini_test.py first)
 *   data/processed/mini_test_labels.csv  ← when USE_MINI_TEST=true
 *   data/processed/test_data.csv         ← when USE_MINI_TEST=false
 *   data/processed/test_labels.csv       ← when USE_MINI_TEST=false
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -std=c++17 -lcublas -o cuda_hybrid cuda_hybrid.cu
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
// Configuration
// ---------------------------------------------------------------------------
static constexpr int   N_FEATURES         = 52;
static constexpr int   N_SVM_K            = 4;
static constexpr int   N_ALL_CLASSES      = 7;
static constexpr float CONF_THRESHOLD     = 0.5f;
static constexpr int   MIN_SAMPLES        = 8;
static constexpr float NORMAL_PROX_THRESH = 4.0f;
static constexpr int   N_MAX_UNCERTAIN    = 35000;
static constexpr float EPS_OVERRIDE       = 0.148f;   // tuned for this dataset
static constexpr long  MAX_EDGES          = 200000000L;
static constexpr int   INF_CHUNK          = 65536;     // linear SVM chunk
static constexpr int   INF_CHUNK_RFF      = 8192;      // RFF SVM chunk (128 MB)
static constexpr int   RFF_N_RFF          = 4096;

// ── Pipeline switches ───────────────────────────────────────────────────────
//
// USE_MINI_TEST — set to false to score the full 756K-row test set.
//   When true, reads mini_test_data.csv / mini_test_labels.csv which is a
//   small stratified sample created by make_mini_test.py (run that first).
static constexpr bool  USE_MINI_TEST    = false;
//
// Data CSV paths — derived from USE_MINI_TEST, do not edit directly.
static const char* TEST_DATA_CSV   = USE_MINI_TEST
    ? "data/processed/mini_test_data.csv"
    : "data/processed/test_data.csv";
static const char* TEST_LABELS_CSV = USE_MINI_TEST
    ? "data/processed/mini_test_labels.csv"
    : "data/processed/test_labels.csv";

// Original IDs: 0=Bots 1=BruteForce 2=DDoS 3=DoS 4=NormalTraffic 5=PortScan 6=WebAttacks
static constexpr int ORIG2INT[N_ALL_CLASSES] = { -1, -1,  0,  1,  2,  3, -1 };
static constexpr int INT2ORIG[N_SVM_K]       = {  2,  3,  4,  5 };
static constexpr int NORMAL_ORIG_ID          = 4;
static constexpr int NORMAL_INT              = 2;   // INT2ORIG[2] = 4 = Normal
static const char*   CLASS_NAMES[N_ALL_CLASSES] = {
    "Bots","BruteForce","DDoS","DoS","NormalTraffic","PortScan","WebAttacks"
};
static constexpr int PRED_NOVEL   = -1;
static constexpr int PRED_UNKNOWN = -2;

// ---------------------------------------------------------------------------
// Error macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(expr) do {                                                 \
    cudaError_t _e=(expr);                                                    \
    if(_e!=cudaSuccess){                                                      \
        fprintf(stderr,"[CUDA] %s:%d  %s\n",__FILE__,__LINE__,               \
                cudaGetErrorString(_e)); exit(1); }                           \
} while(0)

#define CUBLAS_CHECK(expr) do {                                               \
    cublasStatus_t _s=(expr);                                                 \
    if(_s!=CUBLAS_STATUS_SUCCESS){                                            \
        fprintf(stderr,"[cuBLAS] %s:%d  status=%d\n",__FILE__,__LINE__,      \
                (int)_s); exit(1); }                                          \
} while(0)

static std::string ts() {
    auto now=std::chrono::system_clock::now();
    std::time_t t=std::chrono::system_clock::to_time_t(now);
    char buf[32]; std::strftime(buf,sizeof(buf),"%H:%M:%S",std::localtime(&t));
    return buf;
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------
struct SvmModel { int K, F; std::vector<float> W, b; };

struct Metrics {
    const char* name;
    float  svm_acc, macro_f1;
    float  eps;
    int    n_clusters;
    float  noise_pct;
    long   bots_total,  bots_correct,  bots_noise;
    long   bf_total,    bf_correct,    bf_noise;
    long   wa_total,    wa_correct,    wa_noise;
    long   unc_benign, unc_holdout, unc_novel, unc_unknown;
};

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------
static SvmModel load_model(const char* path) {
    std::ifstream f(path,std::ios::binary);
    if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    SvmModel m; int dims[2];
    f.read(reinterpret_cast<char*>(dims),sizeof(dims));
    m.K=dims[0]; m.F=dims[1];
    m.W.resize((size_t)m.K*m.F); m.b.resize(m.K);
    f.read(reinterpret_cast<char*>(m.W.data()),(size_t)m.K*m.F*sizeof(float));
    f.read(reinterpret_cast<char*>(m.b.data()),m.K*sizeof(float));
    printf("[%s] Loaded %s: K=%d F=%d\n",ts().c_str(),path,m.K,m.F);
    return m;
}

static std::vector<float> load_csv_float(const char* path) {
    std::ifstream f(path);
    if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    std::vector<float> d; std::string line,tok;
    while(std::getline(f,line)){
        std::istringstream ss(line);
        while(std::getline(ss,tok,',')) d.push_back(std::stof(tok));
    }
    return d;
}

static std::vector<int> load_csv_int(const char* path) {
    std::ifstream f(path);
    if(!f){fprintf(stderr,"Cannot open %s\n",path);exit(1);}
    std::vector<int> d; std::string line;
    while(std::getline(f,line)) d.push_back(std::stoi(line));
    return d;
}

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------
__global__ void add_bias_kernel(float* D, const float* b, int N, int K) {
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    if(tid<N*K) D[tid]+=b[tid%K];
}

__global__ void extract_conf_kernel(const float* __restrict__ D,
                                     float* __restrict__ conf,
                                     int*   __restrict__ pred,
                                     int N, int K) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return;
    const float* row=D+(size_t)i*K;
    int best=0; float bval=row[0];
    for(int k=1;k<K;++k) if(row[k]>bval){bval=row[k];best=k;}
    conf[i]=bval; pred[i]=best;
}

__global__ void compute_kdist_kernel(const float* __restrict__ X, int n,
                                      float* __restrict__ kdist) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float xi0=X[i*4+0],xi1=X[i*4+1],xi2=X[i*4+2],xi3=X[i*4+3];
    float top[MIN_SAMPLES];
    for(int t=0;t<MIN_SAMPLES;++t) top[t]=1e30f;
    for(int j=0;j<n;++j){
        if(j==i) continue;
        float d0=xi0-X[j*4+0],d1=xi1-X[j*4+1],d2=xi2-X[j*4+2],d3=xi3-X[j*4+3];
        float dist2=d0*d0+d1*d1+d2*d2+d3*d3;
        if(dist2<top[MIN_SAMPLES-1]){
            int pos=MIN_SAMPLES-1;
            while(pos>0&&top[pos-1]>dist2){top[pos]=top[pos-1];--pos;}
            top[pos]=dist2;
        }
    }
    kdist[i]=sqrtf(top[MIN_SAMPLES-1]);
}

__global__ void count_neighbors_kernel(const float* __restrict__ X, int n,
                                        float eps2, int* __restrict__ nbr_count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float xi0=X[i*4+0],xi1=X[i*4+1],xi2=X[i*4+2],xi3=X[i*4+3];
    int cnt=0;
    for(int j=0;j<n;++j){
        float d0=xi0-X[j*4+0],d1=xi1-X[j*4+1],d2=xi2-X[j*4+2],d3=xi3-X[j*4+3];
        if(d0*d0+d1*d1+d2*d2+d3*d3<=eps2) ++cnt;
    }
    nbr_count[i]=cnt;
}

__global__ void fill_neighbors_kernel(const float* __restrict__ X, int n, float eps2,
                                       const int* __restrict__ row_ptr,
                                       int* __restrict__ col_idx) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float xi0=X[i*4+0],xi1=X[i*4+1],xi2=X[i*4+2],xi3=X[i*4+3];
    int base=row_ptr[i],fill=0;
    for(int j=0;j<n;++j){
        float d0=xi0-X[j*4+0],d1=xi1-X[j*4+1],d2=xi2-X[j*4+2],d3=xi3-X[j*4+3];
        if(d0*d0+d1*d1+d2*d2+d3*d3<=eps2) col_idx[base+fill++]=j;
    }
}

__global__ void rff_transform_kernel(const float* __restrict__ X,
                                      const float* __restrict__ Omega,
                                      const float* __restrict__ phi,
                                      float* __restrict__ Z,
                                      int N, int IN_F, int D, float scale) {
    long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=(long long)N*D) return;
    int n=(int)(idx/D),d=(int)(idx%D); float dot=0.f;
    const float* x=X+(long long)n*IN_F;
    const float* om=Omega+(long long)d*IN_F;
    for(int f=0;f<IN_F;++f) dot+=om[f]*x[f];
    Z[idx]=scale*cosf(dot+phi[d]);
}

// ---------------------------------------------------------------------------
// Auto-eps (Kneedle, p90 clip)
// ---------------------------------------------------------------------------
static float auto_eps(std::vector<float>& kdist) {
    int n=(int)kdist.size();
    std::sort(kdist.begin(),kdist.end());
    printf("  k-dist: p10=%.5f  p25=%.5f  p50=%.5f  p75=%.5f  p90=%.5f\n",
           (double)kdist[n/10],(double)kdist[n/4],(double)kdist[n/2],
           (double)kdist[3*n/4],(double)kdist[9*n/10]);
    if(EPS_OVERRIDE>0.f){
        printf("  eps=%.5f (EPS_OVERRIDE)\n",(double)EPS_OVERRIDE);
        return EPS_OVERRIDE;
    }
    int n_clip=std::max(4,(int)(n*0.90f));
    float v0=kdist[0],vn=kdist[n_clip-1],range=vn-v0+1e-9f;
    int knee=n_clip/2; float max_dev=-1e30f;
    for(int i=0;i<n_clip;++i){
        float xn=(float)i/(n_clip-1),yn=(kdist[i]-v0)/range;
        float dev=xn-yn; if(dev>max_dev){max_dev=dev;knee=i;}
    }
    knee=std::max(n/10,std::min(knee,(int)(n*0.87f)));
    printf("  eps=%.5f (auto-knee)\n",(double)kdist[knee]);
    return kdist[knee];
}

// ---------------------------------------------------------------------------
// BFS cluster expansion
// ---------------------------------------------------------------------------
static int bfs_clusters(int n, const std::vector<int>& rp, const std::vector<int>& ci,
                         const std::vector<bool>& is_core, std::vector<int>& labels) {
    labels.assign(n,-1); int nc=0; std::queue<int> q;
    for(int i=0;i<n;++i){
        if(!is_core[i]||labels[i]>=0) continue;
        labels[i]=nc; q.push(i);
        while(!q.empty()){
            int u=q.front(); q.pop();
            for(int e=rp[u];e<rp[u+1];++e){
                int v=ci[e]; if(labels[v]>=0)continue;
                labels[v]=nc; if(is_core[v])q.push(v);
            }
        }
        ++nc;
    }
    return nc;
}

// ---------------------------------------------------------------------------
// Save DBSCAN cluster centroids for inductive inference.
//
// Format (little-endian):
//   int   nc              number of clusters
//   float eps             neighbourhood radius used
//   float norm_ctr[4]     Normal centroid in 4-D SVM-score space
//   repeat nc times:
//     int   label         orig class ID, NORMAL_ORIG_ID(4), PRED_UNKNOWN(-2)
//     float centroid[4]   mean position of all cluster members
//
// Inference: assign uncertain sample to nearest centroid within eps,
//            else PRED_NOVEL.
// ---------------------------------------------------------------------------
static void save_dbscan_model(const char* path, int nc, float eps,
                               const std::vector<float>& norm_ctr,
                               const std::vector<int>&   cname,
                               const std::vector<float>& X_db,
                               const std::vector<int>&   db_labels,
                               int n_db)
{
    std::vector<std::array<float,4>> ctrs(nc, {0,0,0,0});
    std::vector<int> sz(nc, 0);
    for (int i = 0; i < n_db; ++i) {
        int c = db_labels[i]; if (c < 0) continue;
        for (int d = 0; d < 4; ++d) ctrs[c][d] += X_db[(size_t)i*4+d];
        ++sz[c];
    }
    for (int c = 0; c < nc; ++c)
        if (sz[c] > 0) for (int d = 0; d < 4; ++d) ctrs[c][d] /= sz[c];

    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr,"[save_dbscan] Cannot open %s\n",path); return; }
    f.write(reinterpret_cast<const char*>(&nc),   sizeof(int));
    f.write(reinterpret_cast<const char*>(&eps),  sizeof(float));
    f.write(reinterpret_cast<const char*>(norm_ctr.data()), 4*sizeof(float));
    for (int c = 0; c < nc; ++c) {
        f.write(reinterpret_cast<const char*>(&cname[c]),     sizeof(int));
        f.write(reinterpret_cast<const char*>(ctrs[c].data()), 4*sizeof(float));
    }
    printf("[%s] Saved DBSCAN model -> %s  (%d clusters, eps=%.5f)\n",
           ts().c_str(), path, nc, (double)eps);
}

// ---------------------------------------------------------------------------
// Step 1 — SVM inference
//   Loads W/b to GPU, runs chunked sgemm + bias + argmax.
//   Returns h_D [N×4], h_conf [N], h_pred [N] (internal class 0-3).
// ---------------------------------------------------------------------------
static void svm_inference(const SvmModel& model, bool use_rff,
                           const float* h_Xte, int N_te,
                           float* d_Omega, float* d_phi_rff,
                           std::vector<float>& h_D,
                           std::vector<float>& h_conf,
                           std::vector<int>&   h_pred) {
    const int mF    = model.F;
    const int chunk = use_rff ? INF_CHUNK_RFF : INF_CHUNK;

    float *d_W, *d_b;
    CUDA_CHECK(cudaMalloc(&d_W,(size_t)N_SVM_K*mF*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N_SVM_K*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_W,model.W.data(),(size_t)N_SVM_K*mF*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,model.b.data(), N_SVM_K*sizeof(float),           cudaMemcpyHostToDevice));

    float *d_X, *d_Z=nullptr, *d_D, *d_conf; int *d_pred_g;
    CUDA_CHECK(cudaMalloc(&d_X,   (size_t)chunk*N_FEATURES*sizeof(float)));
    if(use_rff) CUDA_CHECK(cudaMalloc(&d_Z,(size_t)chunk*mF*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D,   (size_t)chunk*N_SVM_K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conf, chunk*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pred_g,chunk*sizeof(int)));

    float* d_in = use_rff ? d_Z : d_X;
    cublasHandle_t cb; CUBLAS_CHECK(cublasCreate(&cb));

    h_D.resize((size_t)N_te*N_SVM_K);
    h_conf.resize(N_te); h_pred.resize(N_te);

    for(int off=0;off<N_te;off+=chunk){
        int c=std::min(chunk,N_te-off);
        CUDA_CHECK(cudaMemcpy(d_X,h_Xte+(size_t)off*N_FEATURES,
                              (size_t)c*N_FEATURES*sizeof(float),cudaMemcpyHostToDevice));
        if(use_rff){
            long long tot=(long long)c*mF;
            int th=256,bl=(int)((tot+th-1)/th);
            rff_transform_kernel<<<bl,th>>>(d_X,d_Omega,d_phi_rff,d_Z,
                                            c,N_FEATURES,mF,sqrtf(2.f/mF));
        }
        const float alpha=1.f,beta=0.f;
        CUBLAS_CHECK(cublasSgemm(cb,CUBLAS_OP_T,CUBLAS_OP_N,N_SVM_K,c,mF,
                                 &alpha,d_W,mF,d_in,mF,&beta,d_D,N_SVM_K));
        int th=256;
        add_bias_kernel    <<<(c*N_SVM_K+th-1)/th,th>>>(d_D,d_b,c,N_SVM_K);
        extract_conf_kernel<<<(c+th-1)/th,th>>>(d_D,d_conf,d_pred_g,c,N_SVM_K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_D.data()+(size_t)off*N_SVM_K,d_D,
                              (size_t)c*N_SVM_K*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_conf.data()+off,  d_conf,  c*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pred.data()+off,  d_pred_g,c*sizeof(int),  cudaMemcpyDeviceToHost));
    }
    CUBLAS_CHECK(cublasDestroy(cb));
    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_X)); if(d_Z) CUDA_CHECK(cudaFree(d_Z));
    CUDA_CHECK(cudaFree(d_D)); CUDA_CHECK(cudaFree(d_conf)); CUDA_CHECK(cudaFree(d_pred_g));
}

// ---------------------------------------------------------------------------
// Steps 2-4 — Partition → DBSCAN → Label → Evaluate
//   Takes the 4-D decision scores from either SVM and runs the full DBSCAN
//   pipeline.  Returns Metrics for the comparison table.
// ---------------------------------------------------------------------------
static Metrics run_dbscan_pipeline(const char* name,
                                    const std::vector<float>& h_D,
                                    const std::vector<float>& h_conf,
                                    const std::vector<int>&   h_svm_pred,
                                    const std::vector<int>&   h_yte_orig,
                                    int N_te,
                                    const char* save_path = nullptr) {
    printf("\n[%s] ────────── Pipeline: %s ──────────\n",ts().c_str(),name);
    auto is_holdout=[](int o){return o==0||o==1||o==6;};

    // ── Step 2: Partition ─────────────────────────────────────────────────
    std::vector<int> ho_idx, unc_idx;
    for(int i=0;i<N_te;++i){
        if(is_holdout(h_yte_orig[i]))      ho_idx.push_back(i);
        else if(h_conf[i]<CONF_THRESHOLD)  unc_idx.push_back(i);
    }
    // Subsample uncertain to cap CSR edges
    if((int)unc_idx.size()>N_MAX_UNCERTAIN){
        std::mt19937 rng(42);
        std::shuffle(unc_idx.begin(),unc_idx.end(),rng);
        unc_idx.resize(N_MAX_UNCERTAIN);
    }
    int nh=(int)ho_idx.size(), nu=(int)unc_idx.size(), n_db=nh+nu;
    printf("  Partition: holdout=%d  uncertain=%d (capped)  confident=%d\n",
           nh,nu,N_te-nh-(int)unc_idx.size());
    printf("  DBSCAN pool size: %d\n",n_db);

    // ── Step 3a: Build X_db in 4-D SVM-score space ───────────────────────
    // Holdout FIRST so indices 0..nh-1 are known class anchors
    std::vector<float> X_db((size_t)n_db*4);
    for(int i=0;i<nh;++i)
        for(int k=0;k<4;++k) X_db[(size_t)i*4+k]=h_D[(size_t)ho_idx[i]*4+k];
    for(int i=0;i<nu;++i)
        for(int k=0;k<4;++k) X_db[(size_t)(nh+i)*4+k]=h_D[(size_t)unc_idx[i]*4+k];

    std::vector<int> ho_orig(nh);
    for(int i=0;i<nh;++i) ho_orig[i]=h_yte_orig[ho_idx[i]];

    // Normal centroid: computed only from uncertain-pool samples whose SVM
    // argmax = Normal (internal class 2).  These live near the decision
    // boundary in 4-D score space, which is where uncertain Normal clusters
    // form.  Using all-test Normal centroid would be ~14 L2 units away and
    // NORMAL_PROX_THRESH would never fire.
    std::array<double,4> nsum={0,0,0,0}; int n_norm=0;
    for(int ui=0;ui<nu;++ui){
        int ti=unc_idx[ui];
        if(h_svm_pred[ti]==NORMAL_INT){
            for(int k=0;k<4;++k) nsum[k]+=h_D[(size_t)ti*4+k];
            ++n_norm;
        }
    }
    std::vector<float> norm_ctr(4,0.f);
    if(n_norm>0) for(int k=0;k<4;++k) norm_ctr[k]=(float)(nsum[k]/n_norm);
    printf("  Normal centroid (uncertain-pool argmax=Normal, n=%d): "
           "[%.3f,%.3f,%.3f,%.3f]\n",n_norm,
           (double)norm_ctr[0],(double)norm_ctr[1],(double)norm_ctr[2],(double)norm_ctr[3]);

    // ── Step 3b: GPU DBSCAN — two-pass CSR ───────────────────────────────
    float* d_Xdb;
    CUDA_CHECK(cudaMalloc(&d_Xdb,(size_t)n_db*4*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Xdb,X_db.data(),(size_t)n_db*4*sizeof(float),cudaMemcpyHostToDevice));

    // k-distance graph → auto-tune eps
    float* d_kdist; CUDA_CHECK(cudaMalloc(&d_kdist,n_db*sizeof(float)));
    {int th=256; compute_kdist_kernel<<<(n_db+th-1)/th,th>>>(d_Xdb,n_db,d_kdist);
     CUDA_CHECK(cudaDeviceSynchronize());}
    std::vector<float> h_kdist(n_db);
    CUDA_CHECK(cudaMemcpy(h_kdist.data(),d_kdist,n_db*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_kdist));
    float eps=auto_eps(h_kdist), eps2=eps*eps;
    printf("  DBSCAN  eps=%.5f  min_samples=%d\n",(double)eps,MIN_SAMPLES);

    // Pass 1: count neighbours
    int* d_nbr; CUDA_CHECK(cudaMalloc(&d_nbr,n_db*sizeof(int)));
    {int th=256; count_neighbors_kernel<<<(n_db+th-1)/th,th>>>(d_Xdb,n_db,eps2,d_nbr);
     CUDA_CHECK(cudaDeviceSynchronize());}
    std::vector<int> h_nbr(n_db);
    CUDA_CHECK(cudaMemcpy(h_nbr.data(),d_nbr,n_db*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_nbr));

    std::vector<bool> is_core(n_db);
    for(int i=0;i<n_db;++i) is_core[i]=(h_nbr[i]>=MIN_SAMPLES);

    // Edge overflow pre-check (avoids silent INT_MAX wrap)
    {long acc=0;
     for(int i=0;i<n_db;++i){
         acc+=h_nbr[i];
         if(acc>MAX_EDGES){
             fprintf(stderr,"\n[ERROR] CSR edges exceed MAX_EDGES=%ld "
                     "(eps=%.5f too large for this pool).  "
                     "Lower EPS_OVERRIDE.\n",MAX_EDGES,(double)eps);
             CUDA_CHECK(cudaFree(d_Xdb)); exit(1);
         }
     }}

    std::vector<int> h_rp(n_db+1,0);
    for(int i=0;i<n_db;++i) h_rp[i+1]=h_rp[i]+h_nbr[i];
    long total_edges=(long)h_rp[n_db];
    printf("  Adjacency edges: %ld  (avg %.1f/point)\n",
           total_edges,(double)total_edges/n_db);

    // Pass 2: fill CSR column indices
    int *d_rp, *d_ci;
    CUDA_CHECK(cudaMalloc(&d_rp,(n_db+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ci, total_edges*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rp,h_rp.data(),(n_db+1)*sizeof(int),cudaMemcpyHostToDevice));
    {int th=256; fill_neighbors_kernel<<<(n_db+th-1)/th,th>>>(d_Xdb,n_db,eps2,d_rp,d_ci);
     CUDA_CHECK(cudaDeviceSynchronize());}
    std::vector<int> h_ci(total_edges);
    CUDA_CHECK(cudaMemcpy(h_ci.data(),d_ci,total_edges*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_Xdb)); CUDA_CHECK(cudaFree(d_rp)); CUDA_CHECK(cudaFree(d_ci));

    // CPU BFS cluster expansion
    std::vector<int> db_labels;
    int nc=bfs_clusters(n_db,h_rp,h_ci,is_core,db_labels);
    int n_noise=0;
    for(int i=0;i<n_db;++i) if(db_labels[i]<0) ++n_noise;
    printf("  Clusters=%d  Noise=%d (%.1f%%)\n",nc,n_noise,100.0*n_noise/n_db);

    // ── Step 4: Label each cluster ────────────────────────────────────────
    // 4a) Collect holdout vote per cluster
    std::vector<std::unordered_map<int,int>> votes(nc);
    for(int i=0;i<nh;++i){
        int c=db_labels[i]; if(c>=0) votes[c][ho_orig[i]]++;
    }
    // 4b) Compute cluster centroids for Normal proximity fallback
    std::vector<std::array<double,4>> centroid(nc,{0,0,0,0});
    std::vector<int> csz(nc,0);
    for(int i=0;i<n_db;++i){
        int c=db_labels[i]; if(c<0) continue;
        for(int d=0;d<4;++d) centroid[c][d]+=X_db[(size_t)i*4+d];
        ++csz[c];
    }
    for(int c=0;c<nc;++c) if(csz[c]>0)
        for(int d=0;d<4;++d) centroid[c][d]/=csz[c];

    // 4c) Assign label to each cluster
    std::vector<int> cname(nc,PRED_UNKNOWN);
    int n_by_holdout=0, n_benign=0, n_unknown=0;
    for(int c=0;c<nc;++c){
        if(!votes[c].empty()){
            // majority vote wins → class identity for this cluster
            int bc=-1,bv=0;
            for(auto&[cl,v]:votes[c]) if(v>bv){bv=v;bc=cl;}
            cname[c]=bc; ++n_by_holdout;
        } else {
            // no holdout anchor → distance to Normal centroid decides
            double d2=0;
            for(int d=0;d<4;++d){
                double diff=centroid[c][d]-norm_ctr[d]; d2+=diff*diff;
            }
            if(d2<(double)NORMAL_PROX_THRESH*NORMAL_PROX_THRESH){
                cname[c]=NORMAL_ORIG_ID; ++n_benign;
            } else {
                cname[c]=PRED_UNKNOWN; ++n_unknown;
            }
        }
    }
    printf("  Cluster labeling: %d holdout-voted, %d reclassified Normal, "
           "%d unknown anomaly\n", n_by_holdout, n_benign, n_unknown);

    // ── Assign final predictions ──────────────────────────────────────────
    std::vector<int> fp(N_te,PRED_NOVEL);
    // Confident non-holdout → SVM argmax
    for(int i=0;i<N_te;++i)
        if(!is_holdout(h_yte_orig[i]) && h_conf[i]>=CONF_THRESHOLD)
            fp[i]=INT2ORIG[h_svm_pred[i]];
    // Uncertain → DBSCAN cluster label
    for(int ui=0;ui<nu;++ui){
        int ti=unc_idx[ui]; int c=db_labels[nh+ui];
        fp[ti]=(c<0) ? PRED_NOVEL : cname[c];
    }
    // Holdout → DBSCAN cluster label (for evaluation)
    for(int hi=0;hi<nh;++hi){
        int ti=ho_idx[hi]; int c=db_labels[hi];
        fp[ti]=(c<0) ? PRED_NOVEL : cname[c];
    }

    // ── Evaluate ─────────────────────────────────────────────────────────
    // (A) SVM confident accuracy
    long cm[N_SVM_K][N_SVM_K]={},cf_tot=0,cf_ok=0;
    for(int i=0;i<N_te;++i){
        int oi=h_yte_orig[i];
        if(oi<0||oi>=N_ALL_CLASSES) continue;
        int ti=ORIG2INT[oi]; if(ti<0||h_conf[i]<CONF_THRESHOLD) continue;
        int fpi=fp[i]; if(fpi<0||fpi>=N_ALL_CLASSES) continue;
        int pi=ORIG2INT[fpi]; if(pi<0) continue;
        cm[ti][pi]++; ++cf_tot;
    }
    for(int k=0;k<N_SVM_K;++k) cf_ok+=cm[k][k];

    // (B) Per-class P/R/F1 → macro F1
    printf("\n  Per-class (SVM confident):\n");
    printf("  %-16s  %8s  %8s  %8s  %8s\n","Class","Prec","Recall","F1","Support");
    float mf1=0;
    for(int k=0;k<N_SVM_K;++k){
        long tp=cm[k][k],fp2=0,fn=0;
        for(int j=0;j<N_SVM_K;++j){if(j!=k)fp2+=cm[j][k]; if(j!=k)fn+=cm[k][j];}
        float p=(tp+fp2>0)?(float)tp/(tp+fp2):0.f;
        float r=(tp+fn>0)?(float)tp/(tp+fn):0.f;
        float f1=(p+r>1e-9f)?2.f*p*r/(p+r):0.f; mf1+=f1;
        printf("  %-16s  %8.4f  %8.4f  %8.4f  %8ld\n",
               CLASS_NAMES[INT2ORIG[k]],(double)p,(double)r,(double)f1,tp+fn);
    }
    mf1/=N_SVM_K;
    printf("  %-16s                          %8.4f\n","macro",(double)mf1);

    // (C) Holdout detection
    printf("\n  Holdout detection:\n");
    printf("  %-16s  %8s  %8s  %8s\n","Class","Total","Correct","Noise");
    Metrics m; m.name=name;
    for(int hc:{0,1,6}){
        long tot=0,cor=0,noi=0;
        for(int hi=0;hi<nh;++hi){
            if(ho_orig[hi]!=hc) continue; ++tot;
            int c=db_labels[hi]; if(c<0){++noi;continue;}
            if(cname[c]==hc) ++cor;
        }
        printf("  %-16s  %8ld  %8ld  %8ld\n",CLASS_NAMES[hc],tot,cor,noi);
        if(hc==0){m.bots_total=tot;m.bots_correct=cor;m.bots_noise=noi;}
        else if(hc==1){m.bf_total=tot;m.bf_correct=cor;m.bf_noise=noi;}
        else{m.wa_total=tot;m.wa_correct=cor;m.wa_noise=noi;}
    }

    // (D) Uncertain reclassification breakdown
    long ub=0,uh=0,un=0,uu=0;
    for(int ui=0;ui<nu;++ui){
        int p=fp[unc_idx[ui]];
        if(p==PRED_NOVEL)++un; else if(p==PRED_UNKNOWN)++uu;
        else if(p==NORMAL_ORIG_ID)++ub; else if(p==0||p==1||p==6)++uh;
    }
    printf("\n  Uncertain reclassification (%d total):\n",nu);
    printf("  %-38s  %d\n","Reclassified Normal (near-Normal cluster)",(int)ub);
    printf("  %-38s  %d\n","Matched holdout pattern (Bots/BF/WebAtk)", (int)uh);
    printf("  %-38s  %d\n","Novel attack (DBSCAN noise -1)",           (int)un);
    printf("  %-38s  %d\n","Unknown anomaly (unclaimed cluster)",       (int)uu);

    m.svm_acc    = cf_tot>0 ? (float)cf_ok/cf_tot : 0.f;
    m.macro_f1   = mf1;
    m.eps        = eps;
    m.n_clusters = nc;
    m.noise_pct  = 100.f*n_noise/n_db;
    m.unc_benign=ub; m.unc_holdout=uh; m.unc_novel=un; m.unc_unknown=uu;
    if (save_path)
        save_dbscan_model(save_path, nc, eps, norm_ctr, cname, X_db, db_labels, n_db);
    return m;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    CUDA_CHECK(cudaSetDevice(0));

    // ── Load test data ────────────────────────────────────────────────────
    printf("[%s] Loading test data...\n",ts().c_str());
    auto h_Xte      = load_csv_float(TEST_DATA_CSV);
    auto h_yte_orig = load_csv_int  (TEST_LABELS_CSV);
    int N_te=(int)h_yte_orig.size();
    assert((int)h_Xte.size()==(size_t)N_te*N_FEATURES);
    printf("[%s] N_test=%d\n",ts().c_str(),N_te);

    // ══════════════════════════════════════════════════════════════════════
    // RFF-SVM + DBSCAN pipeline
    // ══════════════════════════════════════════════════════════════════════
    printf("\n[%s] ═══ RFF-SVM + DBSCAN ═══\n",ts().c_str());
    SvmModel rff = load_model("best_rff_model.bin");
    assert(rff.K==N_SVM_K && rff.F==RFF_N_RFF);

    // Load Omega and phi for the RFF projection
    float *d_Omega=nullptr, *d_phi=nullptr;
    {
        std::ifstream fo("best_rff_Omega.bin",std::ios::binary);
        if(!fo){fprintf(stderr,"Cannot open best_rff_Omega.bin\n");exit(1);}
        int odims[2]; fo.read(reinterpret_cast<char*>(odims),sizeof(odims));
        assert(odims[0]==RFF_N_RFF && odims[1]==N_FEATURES);
        std::vector<float> h_Om((long long)RFF_N_RFF*N_FEATURES);
        fo.read(reinterpret_cast<char*>(h_Om.data()),(long long)RFF_N_RFF*N_FEATURES*sizeof(float));
        CUDA_CHECK(cudaMalloc(&d_Omega,(long long)RFF_N_RFF*N_FEATURES*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_Omega,h_Om.data(),(long long)RFF_N_RFF*N_FEATURES*sizeof(float),
                              cudaMemcpyHostToDevice));

        std::ifstream fp("best_rff_phi.bin",std::ios::binary);
        if(!fp){fprintf(stderr,"Cannot open best_rff_phi.bin\n");exit(1);}
        int pD; fp.read(reinterpret_cast<char*>(&pD),sizeof(int)); assert(pD==RFF_N_RFF);
        std::vector<float> h_phi(RFF_N_RFF);
        fp.read(reinterpret_cast<char*>(h_phi.data()),RFF_N_RFF*sizeof(float));
        CUDA_CHECK(cudaMalloc(&d_phi,RFF_N_RFF*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_phi,h_phi.data(),RFF_N_RFF*sizeof(float),cudaMemcpyHostToDevice));
        printf("[%s] RFF params: Omega[%d×%d]  phi[%d]\n",ts().c_str(),RFF_N_RFF,N_FEATURES,RFF_N_RFF);
    }

    std::vector<float> D_rff, conf_rff; std::vector<int> pred_rff;
    svm_inference(rff, /*use_rff=*/true, h_Xte.data(), N_te,
                  d_Omega, d_phi,
                  D_rff, conf_rff, pred_rff);
    CUDA_CHECK(cudaFree(d_Omega)); CUDA_CHECK(cudaFree(d_phi));

    run_dbscan_pipeline("RFF-SVM",
                        D_rff, conf_rff, pred_rff,
                        h_yte_orig, N_te,
                        "best_rff_dbscan_model.bin");

    printf("\n[%s] Done.\n",ts().c_str());
    return 0;
}
