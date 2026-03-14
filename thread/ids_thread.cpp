/*
 * Network Intrusion Detection System — C++ std::thread
 * =====================================================
 * Pipeline based on omp_infer.cpp logic:
 *
 *   [Stage 1] Load processed data
 *   [Stage 2] Train 6 RBF-SVM pairs in parallel (1 thread/pair)
 *   [Stage 3] Predict test set in parallel (N threads, chunked)
 *               → score_mat[N×4], conf[N], pred[N]
 *   [Stage 4] DBSCAN — pool construction (from omp_infer):
 *               holdout                            → DBSCAN pool (anchors)
 *               conf < CONF_THRESHOLD, pred≠Normal → DBSCAN pool (uncertain attack)
 *               conf < CONF_THRESHOLD, pred=Normal → Direct Normal (skip DBSCAN)
 *               conf >= CONF_THRESHOLD             → SVM final
 *             2-pass CSR DBSCAN (O(n²)) on 4-D SVM score space
 *             Cluster labelling: holdout majority vote +
 *                                centroid argmax=Normal AND dist<NORMAL_PROX_THRESH
 *   [Stage 5] Metrics + GFLOPS
 *
 * Compile:
 *   g++ -O3 -std=c++17 -pthread -o ids_thread ids_thread.cpp
 *
 * Usage:
 *   ./ids_thread [--test]
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include <set>
#include <unordered_map>
#include <queue>
#include <array>
#include <iomanip>

// ===================== Constants =====================
const std::string DATA_DIR        = "scripts/data/processed";

// SVM training
const double GAMMA          = 0.1;
const double LR             = 0.01;
const int    BATCH_SIZE     = 500;
const int    MAX_SV         = 2000;
const int    MAX_ITER       = 200;
const int    MAX_TRAIN_PAIR = 5000;   // per class per pair

// DBSCAN (from omp_infer)
const float  CONF_THRESHOLD     = 0.5f;
const int    N_MAX_UNCERTAIN    = 110000;
const float  EPS                = 0.148f;  // fixed like omp_infer
const int    MIN_SAMPLES        = 8;
const float  NORMAL_PROX_THRESH = 7.0f;

// Class mapping (identical to omp_infer)
const int N_ALL     = 7;
const int N_CLASSES = 4;
const int ORIG2INT[7]  = { -1, -1,  0,  1,  2,  3, -1 };
const int INT2ORIG[4]  = {  2,  3,  4,  5 };
const int NORMAL_INT   = 2;   // INT2ORIG[2] = 4
const int NORMAL_ORIG  = 4;
const int PRED_NOVEL   = -1;
const int PRED_UNKNOWN = -2;
const char* CLASS_NAMES[7] = {
    "Bots","BruteForce","DDoS","DoS","NormalTraffic","PortScan","WebAttacks"
};

// Test mode
const int TEST_TRAIN_PER = 2000;
const int TEST_TEST_PER  = 500;

using Matrix = std::vector<std::vector<double>>;
using Vec    = std::vector<double>;
using IVec   = std::vector<int>;

// ===================== Timer =====================
struct Timer {
    std::chrono::high_resolution_clock::time_point t0;
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now()-t0).count();
    }
};

static std::string ts() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[16]; std::strftime(buf,sizeof(buf),"%H:%M:%S",std::localtime(&t));
    return buf;
}

// ===================== Loaders =====================
Matrix load_csv(const std::string& path) {
    Matrix data; std::ifstream f(path);
    if (!f) { std::cerr<<"Cannot open: "<<path<<"\n"; return data; }
    std::string line;
    while (std::getline(f,line)) {
        if (line.empty()) continue;
        Vec row; std::stringstream ss(line); std::string tok;
        while (std::getline(ss,tok,',')) row.push_back(std::stod(tok));
        data.push_back(row);
    }
    return data;
}
IVec load_labels(const std::string& path) {
    IVec v; std::ifstream f(path);
    if (!f) { std::cerr<<"Cannot open: "<<path<<"\n"; return v; }
    std::string line;
    while (std::getline(f,line)) if (!line.empty()) v.push_back(std::stoi(line));
    return v;
}
std::map<int,std::string> load_label_map(const std::string& dir) {
    std::map<int,std::string> lm;
    std::ifstream f(dir+"/label_mapping.txt"); if (!f) return lm;
    std::string line;
    while (std::getline(f,line)) {
        auto p=line.find(':'); if (p==std::string::npos) continue;
        int id=std::stoi(line.substr(0,p));
        std::string nm=line.substr(p+1);
        auto h=nm.find(" [HOLDOUT]"); if (h!=std::string::npos) nm=nm.substr(0,h);
        lm[id]=nm;
    }
    return lm;
}
struct Config { IVec holdout_ids, svm_ids; };
Config load_config(const std::string& dir) {
    Config cfg; std::ifstream f(dir+"/holdout_config.txt"); if (!f) return cfg;
    std::string line;
    while (std::getline(f,line)) {
        if (line.find('=')==std::string::npos) continue;
        auto eq=line.find('=');
        std::string k=line.substr(0,eq), v=line.substr(eq+1);
        auto parse=[](const std::string& s){
            IVec r; std::stringstream ss(s); std::string t;
            while(std::getline(ss,t,',')) if(!t.empty()) r.push_back(std::stoi(t));
            return r;
        };
        if (k=="holdout_class_ids") cfg.holdout_ids=parse(v);
        if (k=="svm_class_ids")     cfg.svm_ids    =parse(v);
    }
    return cfg;
}

// ===================== RBF Kernel =====================
inline double rbf(const Vec& a, const Vec& b, double gamma) {
    double d=0; for (size_t i=0;i<a.size();i++){double t=a[i]-b[i];d+=t*t;}
    return std::exp(-gamma*d);
}

// ===================== SVM (Pegasos-style, same as before) =====================
struct SVMModel { int ca,cb; Matrix sv; Vec alpha; double bias; long long flops; };

SVMModel train_svm_pair(int ca,int cb,const Matrix& X,const IVec& y,
                         double gamma,double lr,int bs,int max_sv,int max_iter,
                         unsigned seed=42)
{
    int n=X.size(), D=X[0].size();
    std::mt19937 rng(seed);
    Matrix sv_arr; Vec al_arr; double bias=0.0;
    IVec idx(n); std::iota(idx.begin(),idx.end(),0);
    for (int ep=0;ep<max_iter;ep++){
        std::shuffle(idx.begin(),idx.end(),rng);
        double eta=lr/(1.0+lr*0.001*ep);
        int b=std::min(bs,n), upd=0;
        for (int bi=0;bi<b;bi++){
            int i=idx[bi]; double yi=(y[i]==ca)?1.0:-1.0, sc=bias;
            for (int si=0;si<(int)sv_arr.size();si++) sc+=al_arr[si]*rbf(sv_arr[si],X[i],gamma);
            if (yi*sc<1.0){sv_arr.push_back(X[i]);al_arr.push_back(eta*yi);bias+=eta*yi*0.1;upd++;}
        }
        if ((int)sv_arr.size()>max_sv){
            IVec ord(sv_arr.size()); std::iota(ord.begin(),ord.end(),0);
            std::partial_sort(ord.begin(),ord.begin()+max_sv,ord.end(),
                [&](int a,int b){return std::abs(al_arr[a])>std::abs(al_arr[b]);});
            ord.resize(max_sv); Matrix nsv; Vec nal;
            for (int i:ord){nsv.push_back(sv_arr[i]);nal.push_back(al_arr[i]);}
            sv_arr=std::move(nsv); al_arr=std::move(nal);
        }
        if (upd==0&&ep>10) break;
    }
    long long flops=(long long)max_iter*bs*sv_arr.size()*(3*D+2);
    return {ca,cb,sv_arr,al_arr,bias,flops};
}

void train_thread_fn(int ca,int cb,const Matrix& X,const IVec& y,
                     double gamma,double lr,int bs,int max_sv,int max_iter,
                     SVMModel& out,std::mutex& mtx)
{
    auto t=Timer(); out=train_svm_pair(ca,cb,X,y,gamma,lr,bs,max_sv,max_iter);
    std::lock_guard<std::mutex> lk(mtx);
    std::cout<<"  ["<<ts()<<"] Pair ("<<ca<<"v"<<cb<<"): "<<out.sv.size()
             <<" SVs ("<<std::fixed<<std::setprecision(1)<<t.ms()<<" ms)\n";
}

// ===================== Save / Load SVM Model =====================
// Format: int n_pairs | per pair: int ca, cb, n_sv, D | double sv[n_sv*D] | double alpha[n_sv] | double bias
void save_models(const std::vector<SVMModel>& models, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cerr<<"Cannot save model: "<<path<<"\n"; return; }
    int n=(int)models.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(int));
    for (auto& m:models) {
        int n_sv=(int)m.sv.size(), D=n_sv>0?(int)m.sv[0].size():0;
        f.write(reinterpret_cast<const char*>(&m.ca), sizeof(int));
        f.write(reinterpret_cast<const char*>(&m.cb), sizeof(int));
        f.write(reinterpret_cast<const char*>(&n_sv), sizeof(int));
        f.write(reinterpret_cast<const char*>(&D),    sizeof(int));
        for (auto& sv:m.sv)
            f.write(reinterpret_cast<const char*>(sv.data()), D*sizeof(double));
        f.write(reinterpret_cast<const char*>(m.alpha.data()), n_sv*sizeof(double));
        f.write(reinterpret_cast<const char*>(&m.bias), sizeof(double));
    }
    std::cout<<"  Model saved -> "<<path<<"\n";
}

std::vector<SVMModel> load_models(const std::string& path) {
    std::vector<SVMModel> models;
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr<<"Cannot load model: "<<path<<"\n"; return models; }
    int n; f.read(reinterpret_cast<char*>(&n), sizeof(int));
    models.resize(n);
    for (auto& m:models) {
        int n_sv,D;
        f.read(reinterpret_cast<char*>(&m.ca),  sizeof(int));
        f.read(reinterpret_cast<char*>(&m.cb),  sizeof(int));
        f.read(reinterpret_cast<char*>(&n_sv),  sizeof(int));
        f.read(reinterpret_cast<char*>(&D),     sizeof(int));
        m.sv.resize(n_sv, Vec(D));
        for (auto& sv:m.sv)
            f.read(reinterpret_cast<char*>(sv.data()), D*sizeof(double));
        m.alpha.resize(n_sv);
        f.read(reinterpret_cast<char*>(m.alpha.data()), n_sv*sizeof(double));
        f.read(reinterpret_cast<char*>(&m.bias), sizeof(double));
        m.flops=0;
    }
    std::cout<<"  Model loaded <- "<<path<<" ("<<n<<" pairs)\n";
    return models;
}

// ===================== Prediction result =====================
struct PredResult {
    int   pred;
    float conf;
    std::array<float,4> scores;  // 4-dim SVM scores (same as omp_infer)
};

PredResult predict_one(const Vec& x, const std::vector<SVMModel>& models,
                       const IVec& class_ids, double gamma)
{
    int K=class_ids.size();
    std::map<int,int>    votes;
    std::map<int,double> score_sum;
    for (int c:class_ids){votes[c]=0;score_sum[c]=0.0;}

    for (auto& m:models){
        double s=m.bias;
        for (size_t si=0;si<m.sv.size();si++) s+=m.alpha[si]*rbf(m.sv[si],x,gamma);
        bool pos=(s>=0);
        votes[pos?m.ca:m.cb]++;
        score_sum[pos?m.ca:m.cb]+=std::abs(s);
    }

    int best=class_ids[0];
    for (int c:class_ids)
        if (votes[c]>votes[best]||(votes[c]==votes[best]&&score_sum[c]>score_sum[best]))
            best=c;

    // confidence = max score (like omp_infer conf_all[i] = best raw score)
    float conf=(float)(score_sum[best] / std::max(votes[best],1));

    // map class_ids → 4-dim score array aligned to ORIG2INT order
    std::array<float,4> sc{};
    for (int i=0;i<K;i++){
        int orig=class_ids[i];
        int ii=ORIG2INT[orig];
        if (ii>=0&&ii<4) sc[ii]=(float)score_sum[orig];
    }
    return {best, conf, sc};
}

void predict_chunk_fn(const Matrix& data, const std::vector<int>& idx,
                      const std::vector<SVMModel>& models, const IVec& class_ids,
                      double gamma,
                      std::vector<PredResult>& out, std::atomic<long long>& fc)
{
    int D=data[0].size(); long long tsv=0;
    for (auto& m:models) tsv+=m.sv.size();
    for (int i:idx){
        out[i]=predict_one(data[i],models,class_ids,gamma);
        fc.fetch_add(tsv*(3*D+2),std::memory_order_relaxed);
    }
}

// ===================== DBSCAN 2-pass CSR (from omp_infer) =====================
inline float dist4_sq(const float* a, const float* b){
    float d0=a[0]-b[0],d1=a[1]-b[1],d2=a[2]-b[2],d3=a[3]-b[3];
    return d0*d0+d1*d1+d2*d2+d3*d3;
}

// Pass 1 + Pass 2 parallelised with std::thread (chunk per thread)
// BFS serial (same as omp_infer)
IVec run_dbscan_csr(const std::vector<float>& X_db, int n_db,
                    float eps, int min_samples,
                    int& n_clusters_out, int n_threads)
{
    const float eps2=eps*eps;

    // --- Pass 1: count neighbours (parallel) ---
    std::cout<<"  ["<<ts()<<"] DBSCAN pass 1: counting neighbours (n="
             <<n_db<<", eps="<<std::fixed<<std::setprecision(5)<<eps<<")...\n";
    std::vector<int> nbr_count(n_db,0);

    auto pass1_fn=[&](int from,int to){
        for (int i=from;i<to;i++){
            const float* xi=X_db.data()+(size_t)i*4;
            int cnt=0;
            for (int j=0;j<n_db;j++)
                if (dist4_sq(xi,X_db.data()+(size_t)j*4)<=eps2) cnt++;
            nbr_count[i]=cnt;
        }
    };
    {
        std::vector<std::thread> thr;
        int chunk=(n_db+n_threads-1)/n_threads;
        for (int t=0;t<n_threads;t++){
            int from=t*chunk, to=std::min(from+chunk,n_db);
            if (from<n_db) thr.emplace_back(pass1_fn,from,to);
        }
        for (auto& t:thr) t.join();
    }

    // core points & CSR row_ptr
    std::vector<bool> is_core(n_db);
    for (int i=0;i<n_db;i++) is_core[i]=(nbr_count[i]>=min_samples);
    std::vector<int> row_ptr(n_db+1,0);
    for (int i=0;i<n_db;i++) row_ptr[i+1]=row_ptr[i]+nbr_count[i];
    long total_edges=(long)row_ptr[n_db];
    std::cout<<"  ["<<ts()<<"] DBSCAN pass 2: filling adjacency (edges="
             <<total_edges<<", avg="<<std::fixed<<std::setprecision(1)
             <<(double)total_edges/n_db<<")...\n";

    // --- Pass 2: fill col_idx (parallel) ---
    std::vector<int> col_idx(total_edges);
    auto pass2_fn=[&](int from,int to){
        for (int i=from;i<to;i++){
            const float* xi=X_db.data()+(size_t)i*4;
            int pos=row_ptr[i];
            for (int j=0;j<n_db;j++)
                if (dist4_sq(xi,X_db.data()+(size_t)j*4)<=eps2)
                    col_idx[pos++]=j;
        }
    };
    {
        std::vector<std::thread> thr;
        int chunk=(n_db+n_threads-1)/n_threads;
        for (int t=0;t<n_threads;t++){
            int from=t*chunk, to=std::min(from+chunk,n_db);
            if (from<n_db) thr.emplace_back(pass2_fn,from,to);
        }
        for (auto& t:thr) t.join();
    }

    // --- BFS cluster expansion (serial, same as omp_infer) ---
    IVec labels(n_db,-1); int nc=0;
    std::queue<int> q;
    for (int i=0;i<n_db;i++){
        if (!is_core[i]||labels[i]>=0) continue;
        labels[i]=nc; q.push(i);
        while (!q.empty()){
            int u=q.front();q.pop();
            for (int e=row_ptr[u];e<row_ptr[u+1];e++){
                int v=col_idx[e];
                if (labels[v]>=0) continue;
                labels[v]=nc;
                if (is_core[v]) q.push(v);
            }
        }
        nc++;
    }
    int n_noise=0; for (int l:labels) if (l<0) n_noise++;
    std::cout<<"  ["<<ts()<<"] DBSCAN done: clusters="<<nc<<" noise="<<n_noise
             <<" ("<<std::fixed<<std::setprecision(1)<<100.0*n_noise/n_db<<"%)\n";
    n_clusters_out=nc;
    return labels;
}

// ===================== Helpers =====================
void stratified_sample(const Matrix& X,const IVec& y,const IVec& cls,int n_per,
                       Matrix& Xo,IVec& yo,unsigned seed=42){
    std::mt19937 rng(seed); std::map<int,IVec> idx;
    for (int i=0;i<(int)y.size();i++) idx[y[i]].push_back(i);
    for (int c:cls){auto& v=idx[c];std::shuffle(v.begin(),v.end(),rng);
        int t=std::min(n_per,(int)v.size());
        for (int i=0;i<t;i++){Xo.push_back(X[v[i]]);yo.push_back(c);}}
}

inline bool is_holdout_orig(int orig){return orig==0||orig==1||orig==6;}

// ===================== Metrics (same layout as omp_infer eval) =====================
void print_metrics(const IVec& y_test, const IVec& final_pred,
                   const IVec& ho_idx, const IVec& ho_orig,
                   const IVec& unc_idx, const std::vector<int>& cluster_class,
                   const IVec& cluster_id_pool, int nh, int n_full_unc,
                   int n_unc_normal, int n_unc_attack, int nu,
                   double train_ms, double predict_ms, double dbscan_ms,
                   double total_ms, long long tot_flops, int N_te,
                   const std::map<int,std::string>& lm)
{
    // per-class confusion (trained classes only)
    long cm[N_CLASSES][N_CLASSES]={};
    for (int i=0;i<N_te;i++){
        int orig=y_test[i]; if (orig<0||orig>=N_ALL) continue;
        int ti=ORIG2INT[orig]; if (ti<0) continue;
        int fp2=final_pred[i]; if (fp2<0||fp2>=N_ALL) continue;
        int pi=ORIG2INT[fp2]; if (pi<0) continue;
        cm[ti][pi]++;
    }
    std::cout<<"\n"<<std::string(60,'=')<<"\n"
             <<"  RESULTS — C++ Thread\n"
             <<std::string(60,'=')<<"\n";
    std::cout<<"\n  [SVM — Known Attack Classification]\n";
    std::cout<<"  "<<std::left<<std::setw(16)<<"Class"
             <<std::right<<std::setw(9)<<"Precision"
             <<std::setw(9)<<"Recall"<<std::setw(9)<<"F1"
             <<std::setw(9)<<"Support\n";
    std::cout<<"  "<<std::string(52,'-')<<"\n";
    float macro_f1=0;
    for (int k=0;k<N_CLASSES;k++){
        long tp=cm[k][k],fp2=0,fn=0;
        for (int j=0;j<N_CLASSES;j++){if(j!=k)fp2+=cm[j][k];if(j!=k)fn+=cm[k][j];}
        float p=(tp+fp2>0)?(float)tp/(tp+fp2):0;
        float r=(tp+fn>0)?(float)tp/(tp+fn):0;
        float f1=(p+r>1e-9f)?2*p*r/(p+r):0;
        macro_f1+=f1;
        std::cout<<"  "<<std::left<<std::setw(16)<<CLASS_NAMES[INT2ORIG[k]]
                 <<std::right<<std::setw(9)<<std::fixed<<std::setprecision(4)<<p
                 <<std::setw(9)<<r<<std::setw(9)<<f1<<std::setw(9)<<tp+fn<<"\n";
    }
    macro_f1/=N_CLASSES;
    std::cout<<"  "<<std::left<<std::setw(16)<<"macro_F1"
             <<std::right<<std::setw(27)<<std::fixed<<std::setprecision(4)<<macro_f1
             <<" (hybrid)\n";

    // holdout detection
    std::cout<<"\n  [DBSCAN — Holdout Detection]\n";
    std::cout<<"  "<<std::left<<std::setw(16)<<"Class"
             <<std::right<<std::setw(8)<<"Total"
             <<std::setw(8)<<"Correct"<<std::setw(8)<<"Noise\n";
    std::cout<<"  "<<std::string(40,'-')<<"\n";
    long ho_total=0,ho_correct=0;
    for (int i=0;i<(int)ho_idx.size();i++){
        int orig=ho_orig[i], c=cluster_id_pool[i];
        bool correct=(c>=0&&(int)cluster_class.size()>c&&cluster_class[c]==orig);
        bool noise=(c<0);
        ho_total++; if(correct) ho_correct++;
        (void)noise;
    }
    // per holdout class
    for (int hc:{0,1,6}){
        long tot=0,cor=0,noi=0;
        for (int i=0;i<(int)ho_idx.size();i++){
            if (ho_orig[i]!=hc) continue; tot++;
            int c=cluster_id_pool[i];
            if (c<0) noi++;
            else if ((int)cluster_class.size()>c&&cluster_class[c]==hc) cor++;
        }
        std::cout<<"  "<<std::left<<std::setw(16)<<CLASS_NAMES[hc]
                 <<std::right<<std::setw(8)<<tot<<std::setw(8)<<cor<<std::setw(8)<<noi<<"\n";
    }
    float dr=ho_total>0?(float)ho_correct/ho_total:0;
    std::cout<<"  Detection rate: "<<std::fixed<<std::setprecision(4)<<dr
             <<" ("<<ho_correct<<"/"<<ho_total<<")\n";

    // false alarm on Normal
    long n_norm=0,n_fa=0;
    for (int i=0;i<N_te;i++) if(y_test[i]==NORMAL_ORIG){n_norm++;if(final_pred[i]==PRED_NOVEL||final_pred[i]==PRED_UNKNOWN)n_fa++;}
    std::cout<<"  False alarm rate: "<<std::setprecision(4)<<(n_norm>0?(float)n_fa/n_norm:0)
             <<" ("<<n_fa<<"/"<<n_norm<<")\n";

    // uncertain breakdown
    long ub=0,uh=0,un=0,uu=0;
    for (int ui=0;ui<(int)unc_idx.size();ui++){
        int p=final_pred[unc_idx[ui]];
        if(p==PRED_NOVEL)un++; else if(p==PRED_UNKNOWN)uu++;
        else if(p==NORMAL_ORIG)ub++; else if(p==0||p==1||p==6)uh++;
    }
    std::cout<<"\n  [Uncertain pool breakdown ("<<n_full_unc<<" total)]\n";
    std::cout<<"  Direct Normal (skip DBSCAN)              : "<<n_unc_normal<<"\n";
    std::cout<<"  Attack-uncertain → DBSCAN                : "<<n_unc_attack
             <<" (capped "<<nu<<")\n";
    std::cout<<"    Reclassified Normal (near-Normal)       : "<<ub<<"\n";
    std::cout<<"    Matched holdout pattern                 : "<<uh<<"\n";
    std::cout<<"    Novel attack (noise -1)                 : "<<un<<"\n";
    std::cout<<"    Unknown anomaly (unclaimed)             : "<<uu<<"\n";

    // throughput
    double gflops=(tot_flops/1e9)/(total_ms/1000.0);
    double tput=N_te/(total_ms/1000.0);
    std::cout<<"\n  [Throughput]\n"
             <<"  Train (parallel) : "<<std::setw(12)<<std::setprecision(1)<<train_ms<<" ms\n"
             <<"  Predict          : "<<std::setw(12)<<predict_ms<<" ms\n"
             <<"  DBSCAN           : "<<std::setw(12)<<dbscan_ms<<" ms\n"
             <<"  Total            : "<<std::setw(12)<<total_ms<<" ms\n"
             <<"  FLOP count       : "<<std::setw(20)<<tot_flops<<"\n"
             <<"  GFLOPS           : "<<std::setw(12)<<std::setprecision(4)<<gflops<<"\n"
             <<"  Throughput       : "<<std::setw(12)<<std::setprecision(1)<<tput<<" samples/sec\n"
             <<std::string(60,'=')<<"\n";
}

// ===================== Main =====================
int main(int argc,char* argv[]){
    bool test_mode=false, load_model_flag=false;
    std::string model_path="thread_svm_model.bin";
    for (int i=1;i<argc;i++){
        std::string a=argv[i];
        if (a=="--test")       test_mode=true;
        if (a=="--load-model") load_model_flag=true;
        if (a=="--model" && i+1<argc) model_path=argv[++i];
    }
    int n_threads=(int)std::thread::hardware_concurrency(); if(!n_threads) n_threads=4;

    std::cout<<"========================================================\n"
             <<"  Network IDS — C++ std::thread\n"
             <<"  Threads: "<<n_threads<<"\n";
    if (test_mode)       std::cout<<"  [TEST MODE]\n";
    if (load_model_flag) std::cout<<"  [LOAD MODEL: "<<model_path<<"]\n";
    std::cout<<"========================================================\n";
    auto t_total=Timer();

    // ── Stage 1: Load ─────────────────────────────────────
    std::cout<<"\n["<<ts()<<"] [Stage 1] Loading data from "<<DATA_DIR<<"...\n";
    Matrix train_data  =load_csv(DATA_DIR+"/train_data.csv");
    IVec   train_labels=load_labels(DATA_DIR+"/train_labels.csv");
    Matrix test_data   =load_csv(DATA_DIR+"/test_data.csv");
    IVec   test_labels =load_labels(DATA_DIR+"/test_labels.csv");
    auto   lm          =load_label_map(DATA_DIR);
    auto   cfg         =load_config(DATA_DIR);
    IVec& svm_ids    =cfg.svm_ids;
    IVec& holdout_ids=cfg.holdout_ids;
    if (!load_model_flag && train_data.empty()){std::cerr<<"Load train failed\n";return 1;}
    if (test_data.empty()){std::cerr<<"Load test failed\n";return 1;}

    if (test_mode){
        Matrix Xt;IVec yt; stratified_sample(train_data,train_labels,svm_ids,TEST_TRAIN_PER,Xt,yt);
        train_data=Xt;train_labels=yt;
        IVec all=svm_ids; for(int h:holdout_ids) all.push_back(h);
        Matrix Xe;IVec ye; stratified_sample(test_data,test_labels,all,TEST_TEST_PER,Xe,ye);
        test_data=Xe;test_labels=ye;
    }
    int N_tr=train_data.size(),N_te=test_data.size(),D=test_data[0].size();
    std::cout<<"  Train="<<N_tr<<" | Test="<<N_te<<" | Features="<<D<<"\n";

    // ── Stage 2: Train or Load SVM Model ──────────────────
    int K=svm_ids.size();
    std::vector<std::pair<int,int>> pairs;
    for (int i=0;i<K;i++) for (int j=i+1;j<K;j++) pairs.push_back({svm_ids[i],svm_ids[j]});

    std::vector<SVMModel> models;
    double train_ms=0; long long train_flops=0,total_sv=0;

    if (load_model_flag) {
        // ── Load pre-trained model ─────────────────────────
        std::cout<<"\n["<<ts()<<"] [Stage 2] Loading model from "<<model_path<<"...\n";
        auto t_load=Timer();
        models=load_models(model_path);
        train_ms=t_load.ms();
        if (models.empty()){std::cerr<<"Model load failed\n";return 1;}
        for (auto& m:models) total_sv+=m.sv.size();
        std::cout<<"  Loaded "<<models.size()<<" pairs | total_sv="<<total_sv
                 <<" ("<<std::fixed<<std::setprecision(1)<<train_ms<<" ms)\n";
    } else {
        // ── Train in parallel ─────────────────────────────
        std::cout<<"\n["<<ts()<<"] [Stage 2] SVM Training ("<<pairs.size()<<" pairs)...\n";
        std::map<int,IVec> cidx; for (int i=0;i<N_tr;i++) cidx[train_labels[i]].push_back(i);
        std::mt19937 rng_main(42);
        std::vector<Matrix> px(pairs.size()); std::vector<IVec> py(pairs.size());
        for (int p=0;p<(int)pairs.size();p++){
            int ca=pairs[p].first,cb=pairs[p].second;
            auto ia=cidx[ca],ib=cidx[cb];
            std::shuffle(ia.begin(),ia.end(),rng_main); std::shuffle(ib.begin(),ib.end(),rng_main);
            int np=std::min(MAX_TRAIN_PAIR,std::min((int)ia.size(),(int)ib.size()));
            ia.resize(np);ib.resize(np);
            for (int i:ia){px[p].push_back(train_data[i]);py[p].push_back(ca);}
            for (int i:ib){px[p].push_back(train_data[i]);py[p].push_back(cb);}
            std::cout<<"  Pair ("<<ca<<"v"<<cb<<"): "<<np<<"/class → "<<px[p].size()<<" total\n";
        }
        models.resize(pairs.size());
        std::mutex mtx;
        auto t_train=Timer();
        {
            std::vector<std::thread> thr;
            for (int p=0;p<(int)pairs.size();p++)
                thr.emplace_back(train_thread_fn,pairs[p].first,pairs[p].second,
                    std::cref(px[p]),std::cref(py[p]),GAMMA,LR,BATCH_SIZE,MAX_SV,MAX_ITER,
                    std::ref(models[p]),std::ref(mtx));
            for (auto& t:thr) t.join();
        }
        train_ms=t_train.ms();
        for (auto& m:models){train_flops+=m.flops;total_sv+=m.sv.size();}
        std::cout<<"  ["<<ts()<<"] Training done: "<<std::fixed<<std::setprecision(1)
                 <<train_ms<<" ms | total_sv="<<total_sv<<"\n";
        // Save model
        save_models(models, model_path);
    }

    // ── Stage 3: Parallel Prediction ──────────────────────
    std::cout<<"\n["<<ts()<<"] [Stage 3] Parallel Prediction ("<<n_threads<<" threads)...\n";
    std::vector<PredResult> pred_results(N_te);
    std::atomic<long long> pred_flops{0};
    std::vector<std::vector<int>> tidx(n_threads);
    for (int i=0;i<N_te;i++) tidx[i%n_threads].push_back(i);
    auto t_pred=Timer();
    {
        std::vector<std::thread> thr;
        for (int t=0;t<n_threads;t++)
            thr.emplace_back(predict_chunk_fn,
                std::cref(test_data),std::cref(tidx[t]),
                std::cref(models),std::cref(svm_ids),GAMMA,
                std::ref(pred_results),std::ref(pred_flops));
        for (auto& t:thr) t.join();
    }
    double predict_ms=t_pred.ms();
    std::cout<<"  ["<<ts()<<"] Predict done: "<<std::fixed<<std::setprecision(1)<<predict_ms<<" ms\n";
    for (int c:svm_ids){
        int cnt=0; for (auto& r:pred_results) if(r.pred==c) cnt++;
        std::cout<<"    class "<<c<<" ("<<lm[c]<<"): "<<cnt
                 <<" ("<<std::setprecision(1)<<100.0*cnt/N_te<<"%)\n";
    }

    // ── Stage 4: DBSCAN — omp_infer pool logic ────────────
    std::cout<<"\n["<<ts()<<"] [Stage 4] DBSCAN — Hybrid pool (omp_infer logic)...\n";

    // Partition: holdout | uncertain-attack | uncertain-Normal | confident
    IVec ho_idx, unc_idx, unc_normal_idx;
    int n_full_uncertain=0;
    for (int i=0;i<N_te;i++){
        int orig=test_labels[i];
        if (is_holdout_orig(orig)){
            ho_idx.push_back(i);
        } else if (pred_results[i].conf < CONF_THRESHOLD){
            n_full_uncertain++;
            if (pred_results[i].pred==INT2ORIG[NORMAL_INT])
                unc_normal_idx.push_back(i);  // uncertain-Normal → direct
            else
                unc_idx.push_back(i);         // uncertain-attack → DBSCAN
        }
    }
    // cap uncertain-attack pool
    if ((int)unc_idx.size()>N_MAX_UNCERTAIN) unc_idx.resize(N_MAX_UNCERTAIN);

    int nh=(int)ho_idx.size(), nu=(int)unc_idx.size();
    int n_db=nh+nu;
    int n_confident=N_te-nh-n_full_uncertain;
    int n_unc_normal=(int)unc_normal_idx.size();
    int n_unc_attack=n_full_uncertain-n_unc_normal;

    std::cout<<"  Holdout samples  (anchors)           : "<<nh<<"\n"
             <<"  Confident samples (SVM final)         : "<<n_confident<<"\n"
             <<"  Uncertain Normal → direct              : "<<n_unc_normal<<"  (skip DBSCAN)\n"
             <<"  Uncertain attack → DBSCAN              : "<<n_unc_attack
             <<" (capped "<<nu<<")\n"
             <<"  DBSCAN pool total                      : "<<n_db<<"\n";

    // Build X_db (4-dim SVM scores, holdout first)
    std::vector<float> X_db((size_t)n_db*4);
    for (int i=0;i<nh;i++)
        for (int k=0;k<4;k++) X_db[(size_t)i*4+k]=pred_results[ho_idx[i]].scores[k];
    for (int i=0;i<nu;i++)
        for (int k=0;k<4;k++) X_db[(size_t)(nh+i)*4+k]=pred_results[unc_idx[i]].scores[k];

    IVec ho_orig(nh); for (int i=0;i<nh;i++) ho_orig[i]=test_labels[ho_idx[i]];

    // Normal centroid (from all Normal test samples, like omp_infer)
    std::array<double,4> nsum={0,0,0,0}; int n_norm=0;
    for (int i=0;i<N_te;i++) if(test_labels[i]==NORMAL_ORIG){
        for (int k=0;k<4;k++) nsum[k]+=pred_results[i].scores[k]; n_norm++;
    }
    std::vector<float> norm_ctr(4,0.f);
    if (n_norm>0) for (int k=0;k<4;k++) norm_ctr[k]=(float)(nsum[k]/n_norm);
    std::cout<<"  Normal centroid (n="<<n_norm<<"): ["
             <<std::fixed<<std::setprecision(3)
             <<norm_ctr[0]<<", "<<norm_ctr[1]<<", "<<norm_ctr[2]<<", "<<norm_ctr[3]<<"]\n";

    // Run DBSCAN
    long long dbscan_flops=0; double dbscan_ms=0;
    IVec final_pred(N_te,PRED_NOVEL);
    IVec cluster_id_pool; std::vector<int> cluster_class;
    int n_clusters=0;

    if (n_db>1){
        auto t_db=Timer();
        IVec db_labels=run_dbscan_csr(X_db,n_db,EPS,MIN_SAMPLES,n_clusters,n_threads);
        dbscan_ms=t_db.ms();
        dbscan_flops=(long long)n_db*(n_db-1)/2*(3*4+1);
        cluster_id_pool=db_labels;

        // Cluster centroids
        std::vector<std::array<double,4>> centroid(n_clusters,{0,0,0,0});
        std::vector<int> csz(n_clusters,0);
        for (int i=0;i<n_db;i++){
            int c=db_labels[i]; if(c<0) continue;
            for (int d=0;d<4;d++) centroid[c][d]+=X_db[(size_t)i*4+d];
            csz[c]++;
        }
        for (int c=0;c<n_clusters;c++) if(csz[c]>0)
            for (int d=0;d<4;d++) centroid[c][d]/=csz[c];

        // Cluster labelling (omp_infer Step 2f)
        std::vector<std::unordered_map<int,int>> votes(n_clusters);
        for (int i=0;i<nh;i++){int c=db_labels[i];if(c>=0)votes[c][ho_orig[i]]++;}

        cluster_class.resize(n_clusters,PRED_UNKNOWN);
        int cnt_ho=0,cnt_norm=0,cnt_unk=0;
        for (int c=0;c<n_clusters;c++){
            if (!votes[c].empty()){
                int bc=-1,bv=0;
                for (auto&[cl,v]:votes[c]) if(v>bv){bv=v;bc=cl;}
                cluster_class[c]=bc; cnt_ho++;
            } else {
                // argmax of centroid
                int ctr_am=0;
                for (int d=1;d<4;d++)
                    if (centroid[c][d]>centroid[c][ctr_am]) ctr_am=d;
                double d2=0;
                for (int d=0;d<4;d++){double diff=centroid[c][d]-norm_ctr[d];d2+=diff*diff;}
                if (ctr_am==NORMAL_INT && d2<(double)NORMAL_PROX_THRESH*NORMAL_PROX_THRESH){
                    cluster_class[c]=NORMAL_ORIG; cnt_norm++;
                } else {
                    cluster_class[c]=PRED_UNKNOWN; cnt_unk++;
                }
            }
        }
        std::cout<<"  Cluster labelling: "<<cnt_ho<<" holdout-voted, "
                 <<cnt_norm<<" reclassified Normal, "<<cnt_unk<<" unknown\n";

        // Build sample→pool position map
        std::vector<int> sample_to_pool(N_te,-1);
        for (int i=0;i<nh;i++) sample_to_pool[ho_idx[i]]=i;
        for (int i=0;i<nu;i++) sample_to_pool[unc_idx[i]]=nh+i;

        // Final prediction assignment (parallel with threads)
        auto assign_fn=[&](int from,int to){
            for (int i=from;i<to;i++){
                int orig=test_labels[i];
                if (!is_holdout_orig(orig)){
                    if (pred_results[i].conf>=CONF_THRESHOLD){
                        // confident → SVM argmax
                        int pk=pred_results[i].pred;
                        final_pred[i]=(pk>=0&&pk<N_ALL&&ORIG2INT[pk]>=0)?pk:PRED_NOVEL;
                        continue;
                    }
                    if (pred_results[i].pred==INT2ORIG[NORMAL_INT]){
                        // uncertain-Normal → direct Normal
                        final_pred[i]=NORMAL_ORIG; continue;
                    }
                }
                // holdout or uncertain-attack → DBSCAN
                int pp=sample_to_pool[i];
                if (pp>=0){
                    int c=db_labels[pp];
                    final_pred[i]=(c<0)?PRED_NOVEL:cluster_class[c];
                }
                // overflow uncertain stays PRED_NOVEL
            }
        };
        {
            std::vector<std::thread> thr;
            int chunk=(N_te+n_threads-1)/n_threads;
            for (int t=0;t<n_threads;t++){
                int from=t*chunk,to=std::min(from+chunk,N_te);
                if (from<N_te) thr.emplace_back(assign_fn,from,to);
            }
            for (auto& t:thr) t.join();
        }
    }

    // ── Stage 5: Metrics & Output ──────────────────────────
    double total_ms=t_total.ms();
    long long tot_flops=train_flops+pred_flops.load()+dbscan_flops;

    print_metrics(test_labels,final_pred,ho_idx,ho_orig,unc_idx,
                  cluster_class,cluster_id_pool,nh,n_full_uncertain,
                  n_unc_normal,n_unc_attack,nu,
                  train_ms,predict_ms,dbscan_ms,total_ms,tot_flops,N_te,lm);

    // Save outputs
    {
        std::ofstream fp("thread_predictions.csv");
        for (int p:final_pred) fp<<p<<"\n";
    }
    {
        double gflops=(tot_flops/1e9)/(total_ms/1000.0);
        double tput=N_te/(total_ms/1000.0);
        std::ofstream fs("thread_summary.csv");
        fs<<"technique,cppthread\ntotal_ms,"<<total_ms
          <<"\ngflops,"<<gflops<<"\nthroughput,"<<tput
          <<"\ntrain_ms,"<<train_ms<<"\npredict_ms,"<<predict_ms
          <<"\ndbscan_ms,"<<dbscan_ms<<"\nflops,"<<tot_flops<<"\n";
    }
    std::cout<<"  Predictions -> thread_predictions.csv\n"
             <<"  Summary     -> thread_summary.csv\n"
             <<"  SVM model   -> "<<model_path<<"\n";
    return 0;
}