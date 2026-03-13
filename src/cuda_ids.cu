// cuda_ids.cu - CUDA GPU accelerated SVM + DBSCAN pipeline
// Upgraded: RFF transformer kernel + One-Vs-Rest Linear SVM (from cuda/src techniques)
// Compile: nvcc -O2 -std=c++17 -o cuda_ids cuda_ids.cu -lm
// Requires: NVIDIA GPU with CUDA toolkit

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <random>

using namespace std;

// ===== Constants =====
const int D_RFF      = 1024;   // RFF dimension
const double GAMMA   = 0.1;    // RBF gamma
const int MAX_DBSCAN = 20000;  // Cap for GPU distance matrix

// ===== CUDA Error Check =====
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
}

// ===== Data I/O =====
vector<vector<double>> load_csv(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Cannot open " << filename << endl; exit(1); }
    string line;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string val;
        while (getline(ss, val, ',')) row.push_back(stod(val));
        if (!row.empty()) data.push_back(row);
    }
    return data;
}

vector<int> load_labels(const string& filename) {
    vector<int> labels;
    ifstream file(filename);
    string line;
    while (getline(file, line)) labels.push_back(stoi(line));
    return labels;
}

struct Timer {
    chrono::high_resolution_clock::time_point t;
    void start() { t = chrono::high_resolution_clock::now(); }
    double ms() { return chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t).count(); }
};

int detect_n_classes(const vector<int>& labels) {
    int mx = 0;
    for (int l : labels) if (l > mx) mx = l;
    return mx + 1;
}

// ===== CUDA Kernels =====

// --- RFF Transform Kernel ---
// Each thread computes one output RFF feature for one sample.
// data:   [N, D_in] row-major (float)
// Omega:  [D_rff, D_in] row-major (float)
// phi:    [D_rff] (float)
// out:    [N, D_rff] row-major (float)
__global__ void rff_transform_kernel(const float* data, const float* Omega, const float* phi,
                                      float* out, int N, int D_in, int D_rff) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    int feat   = blockIdx.y * blockDim.y + threadIdx.y;
    if (sample >= N || feat >= D_rff) return;

    float dot = phi[feat];
    for (int d = 0; d < D_in; d++)
        dot += Omega[feat * D_in + d] * data[sample * D_in + d];
    float norm = sqrtf(2.0f / (float)D_rff);
    out[sample * D_rff + feat] = norm * cosf(dot);
}

// --- OvR Linear SVM Predict Kernel ---
// rff_data:   [N, D_rff] (float)
// W:          [n_classes, D_rff+1] (float, +1 for bias)
// preds:      [N] (int output)
// confs:      [N] (float confidence = max_score - 2nd_max)
__global__ void ovr_predict_kernel(const float* rff_data, const float* W,
                                    int* preds, float* confs,
                                    int N, int D_rff, int n_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float best_score  = -1e30f;
    float second_score = -1e30f;
    int   best_class  = 0;

    for (int c = 0; c < n_classes; c++) {
        float score = W[c * (D_rff + 1) + D_rff]; // bias
        for (int d = 0; d < D_rff; d++)
            score += W[c * (D_rff + 1) + d] * rff_data[i * D_rff + d];

        if (score > best_score) {
            second_score = best_score;
            best_score   = score;
            best_class   = c;
        } else if (score > second_score) {
            second_score = score;
        }
    }
    preds[i] = best_class;
    confs[i]  = best_score - second_score;
}

// --- Distance Matrix Kernel (for DBSCAN) ---
__global__ void distance_matrix_kernel(const float* data, float* dist, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N || i >= j) return;
    float sum = 0.0f;
    for (int k = 0; k < D; k++) {
        float d = data[i * D + k] - data[j * D + k];
        sum += d * d;
    }
    float dv = sqrtf(sum);
    dist[i * N + j] = dv;
    dist[j * N + i] = dv;
}

// ===== CPU DBSCAN (using GPU-computed distance matrix) =====
struct DBSCANResult {
    vector<int> cluster_labels;
    int n_clusters, n_noise;
};

DBSCANResult cpu_dbscan_from_matrix(const vector<float>& dist, int N, double eps, int min_pts) {
    DBSCANResult res;
    res.cluster_labels.assign(N, -2);
    res.n_clusters = 0;

    auto region_query = [&](int p) {
        vector<int> nb;
        for (int i = 0; i < N; i++)
            if ((double)dist[p * N + i] <= eps) nb.push_back(i);
        return nb;
    };

    for (int i = 0; i < N; i++) {
        if (res.cluster_labels[i] != -2) continue;
        auto nb = region_query(i);
        if ((int)nb.size() < min_pts) { res.cluster_labels[i] = -1; continue; }
        int cid = res.n_clusters++;
        res.cluster_labels[i] = cid;
        vector<int> seed(nb.begin(), nb.end());
        for (size_t s = 0; s < seed.size(); s++) {
            int q = seed[s];
            if (res.cluster_labels[q] == -1) res.cluster_labels[q] = cid;
            if (res.cluster_labels[q] != -2) continue;
            res.cluster_labels[q] = cid;
            auto qn = region_query(q);
            if ((int)qn.size() >= min_pts)
                for (int nn : qn)
                    if (find(seed.begin(), seed.end(), nn) == seed.end())
                        seed.push_back(nn);
        }
    }
    res.n_noise = count(res.cluster_labels.begin(), res.cluster_labels.end(), -1);
    return res;
}

// ===== CPU OvR Linear SVM Training (SGD + Cosine LR + Class Weights) =====
// (Training is done on CPU with RFF-transformed float features for simplicity.
//  Inference is on GPU via ovr_predict_kernel above.)
struct OvrModel {
    int n_classes, D_rff;
    vector<float> W;    // [n_classes * (D_rff+1)] flat, row-major
    float conf_threshold = 0.3f;

    void train(const vector<vector<float>>& rff_data, const vector<int>& labels,
               int max_epochs = 30, float lr = 0.05f, float reg = 1e-4f) {
        int N = rff_data.size();
        W.assign((size_t)n_classes * (D_rff + 1), 0.0f);

        // Compute class weights
        vector<int> counts(n_classes, 0);
        for (int l : labels) if (l >= 0 && l < n_classes) counts[l]++;
        vector<float> cw(n_classes, 1.0f);
        for (int c = 0; c < n_classes; c++)
            if (counts[c] > 0) cw[c] = (float)N / (n_classes * counts[c]);

        for (int ep = 0; ep < max_epochs; ep++) {
            float eta = lr * 0.5f * (1.0f + cosf(M_PI * (float)ep / max_epochs));
            for (int c = 0; c < n_classes; c++) {
                for (int i = 0; i < N; i++) {
                    float yi = (labels[i] == c) ? 1.0f : -1.0f;
                    float score = W[c * (D_rff + 1) + D_rff]; // bias
                    for (int d = 0; d < D_rff; d++)
                        score += W[c * (D_rff + 1) + d] * rff_data[i][d];
                    float hinge = max(0.0f, 1.0f - yi * score);
                    if (hinge > 0.0f) {
                        float g = -yi * cw[c] * eta;
                        for (int d = 0; d < D_rff; d++)
                            W[c * (D_rff + 1) + d] -= g * rff_data[i][d];
                        W[c * (D_rff + 1) + D_rff] -= g; // bias
                    }
                }
                // L2 weight decay
                for (int d = 0; d < D_rff; d++)
                    W[c * (D_rff + 1) + d] *= (1.0f - eta * reg);
            }
            if ((ep + 1) % 10 == 0)
                cout << "  Epoch " << (ep+1) << "/" << max_epochs << " done" << endl;
        }
    }

    void save(const string& filename) const {
        ofstream f(filename);
        f << n_classes << " " << D_rff << " " << conf_threshold << "\n";
        for (float v : W) f << v << " ";
        f << "\n";
        cout << "OvR model saved to " << filename << "\n";
    }
};

// ===== GPU Pipeline =====
void gpu_rff_transform(const vector<vector<double>>& data,
                        const vector<float>& Omega_h, const vector<float>& phi_h,
                        vector<vector<float>>& out, int D_in, int D_rff) {
    int N = data.size();
    // Flatten input to float
    vector<float> flat_data(N * D_in);
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D_in; d++)
            flat_data[i * D_in + d] = (float)data[i][d];

    // GPU allocations
    float *d_data, *d_Omega, *d_phi, *d_out;
    CUDA_CHECK(cudaMalloc(&d_data,  N * D_in  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Omega, D_rff * D_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_phi,   D_rff         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,   N * D_rff     * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data,  flat_data.data(),  N * D_in * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Omega, Omega_h.data(),    D_rff * D_in * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi,   phi_h.data(),      D_rff * sizeof(float),      cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (D_rff + 15) / 16);
    rff_transform_kernel<<<grid, block>>>(d_data, d_Omega, d_phi, d_out, N, D_in, D_rff);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    vector<float> flat_out(N * D_rff);
    CUDA_CHECK(cudaMemcpy(flat_out.data(), d_out, N * D_rff * sizeof(float), cudaMemcpyDeviceToHost));
    out.resize(N, vector<float>(D_rff));
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D_rff; d++)
            out[i][d] = flat_out[i * D_rff + d];

    cudaFree(d_data); cudaFree(d_Omega); cudaFree(d_phi); cudaFree(d_out);
}

void gpu_ovr_predict(const vector<vector<float>>& rff_data, const OvrModel& model,
                      vector<int>& preds, vector<float>& confs) {
    int N = rff_data.size();
    int D_rff = model.D_rff, nc = model.n_classes;

    vector<float> flat_rff(N * D_rff);
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D_rff; d++)
            flat_rff[i * D_rff + d] = rff_data[i][d];

    float *d_rff, *d_W, *d_confs; int *d_preds;
    CUDA_CHECK(cudaMalloc(&d_rff,   N * D_rff * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W,     nc * (D_rff + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_preds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_confs, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rff, flat_rff.data(),    N * D_rff * sizeof(float),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,   model.W.data(), nc * (D_rff + 1) * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (N + block - 1) / block;
    ovr_predict_kernel<<<grid, block>>>(d_rff, d_W, d_preds, d_confs, N, D_rff, nc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    preds.resize(N); confs.resize(N);
    CUDA_CHECK(cudaMemcpy(preds.data(), d_preds, N * sizeof(int),   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(confs.data(), d_confs, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_rff); cudaFree(d_W); cudaFree(d_preds); cudaFree(d_confs);
}

DBSCANResult gpu_dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int N = data.size(), D = data[0].size();
    vector<float> flat(N * D);
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++)
            flat[i * D + d] = (float)data[i][d];

    float *d_data, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_data, N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dist, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dist, 0, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, flat.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    distance_matrix_kernel<<<grid, block>>>(d_data, d_dist, N, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> dist_h((size_t)N * N);
    CUDA_CHECK(cudaMemcpy(dist_h.data(), d_dist, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_data); cudaFree(d_dist);

    return cpu_dbscan_from_matrix(dist_h, N, eps, min_pts);
}

// ===== Main =====
int main() {
    cout << "=== Network IDS: CUDA GPU ===" << endl;
    cout << "  Algorithm: GPU RFF Transform + OvR Linear SVM + GPU DBSCAN" << endl;

    auto train_data   = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data    = load_csv("data/test_data.csv");
    auto test_labels  = load_labels("data/test_labels.csv");

    int N_train = train_data.size(), N_test = test_data.size();
    int D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    cout << "Train: " << N_train << " | Test: " << N_test
         << " | Features: " << D << " | Classes: " << n_classes << endl;

    // ===== Step 1: Generate RFF parameters (CPU) =====
    cout << "\n--- Generating RFF Parameters (D_rff=" << D_RFF << ", gamma=" << GAMMA << ") ---" << endl;
    mt19937 rng(42);
    normal_distribution<float>  norm_dist(0.0f, (float)sqrt(2.0 * GAMMA));
    uniform_real_distribution<float> unif_dist(0.0f, (float)(2.0 * M_PI));

    vector<float> Omega_h(D_RFF * D), phi_h(D_RFF);
    for (auto& v : Omega_h) v = norm_dist(rng);
    for (auto& v : phi_h)   v = unif_dist(rng);

    // ===== Step 2: GPU RFF Transform =====
    cout << "\n--- GPU RFF Transform ---" << endl;
    Timer t_rff; t_rff.start();
    vector<vector<float>> rff_train, rff_test;
    gpu_rff_transform(train_data, Omega_h, phi_h, rff_train, D, D_RFF);
    gpu_rff_transform(test_data,  Omega_h, phi_h, rff_test,  D, D_RFF);
    double rff_time = t_rff.ms();
    long long total_flops = (long long)(N_train + N_test) * D_RFF * (D + 1);
    cout << "RFF transform: " << fixed << setprecision(1) << rff_time << " ms" << endl;

    // ===== Step 3: CPU OvR SVM Training (on RFF features) =====
    cout << "\n--- OvR Linear SVM Training (SGD + Cosine LR + Class Weights) ---" << endl;
    OvrModel model;
    model.n_classes = n_classes;
    model.D_rff = D_RFF;
    Timer t1; t1.start();
    model.train(rff_train, train_labels, /*epochs=*/30, /*lr=*/0.05f);
    double svm_train_time = t1.ms();
    total_flops += (long long)30 * N_train * n_classes * (2 * D_RFF + 3);
    cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    // ===== Step 4: GPU OvR Predict =====
    cout << "\n--- GPU OvR Predict ---" << endl;
    Timer t2; t2.start();
    vector<int> all_preds;
    vector<float> all_confs;
    gpu_ovr_predict(rff_test, model, all_preds, all_confs);
    double svm_pred_time = t2.ms();
    total_flops += (long long)N_test * n_classes * (D_RFF + 1);
    cout << "SVM prediction: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;

    // ===== Step 5: Split confident vs uncertain =====
    float threshold = model.conf_threshold;
    vector<int> final_predictions(N_test);
    vector<int> uncertain_idx;
    vector<vector<double>> uncertain_data;

    for (int i = 0; i < N_test; i++) {
        if (all_confs[i] >= threshold) {
            final_predictions[i] = all_preds[i];
        } else {
            final_predictions[i] = -1;
            uncertain_idx.push_back(i);
            uncertain_data.push_back(test_data[i]);
        }
    }

    if ((int)uncertain_data.size() > MAX_DBSCAN) {
        for (size_t i = MAX_DBSCAN; i < uncertain_idx.size(); i++)
            final_predictions[uncertain_idx[i]] = all_preds[uncertain_idx[i]];
        uncertain_idx.resize(MAX_DBSCAN);
        uncertain_data.resize(MAX_DBSCAN);
    }
    cout << "Confident: " << (N_test - (int)uncertain_idx.size())
         << " | Uncertain: " << uncertain_idx.size() << endl;

    // ===== Step 6: GPU DBSCAN on uncertain subset =====
    double dbscan_time = 0;
    if (uncertain_data.size() > 1) {
        cout << "\n--- GPU DBSCAN (distance matrix on GPU) ---" << endl;
        Timer t3; t3.start();
        double dbscan_eps = 2.0;
        int dbscan_min_pts = max(3, D / 10);
        auto db = gpu_dbscan(uncertain_data, dbscan_eps, dbscan_min_pts);
        dbscan_time = t3.ms();
        total_flops += (long long)uncertain_data.size() * uncertain_data.size() * (3 * D + 1) / 2;
        cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms"
             << " | Clusters: " << db.n_clusters << " | Noise: " << db.n_noise << endl;

        for (size_t i = 0; i < uncertain_idx.size(); i++)
            final_predictions[uncertain_idx[i]] = (db.cluster_labels[i] == -1)
                                                   ? -1 : all_preds[uncertain_idx[i]];
    }

    // ===== Results =====
    int correct = 0, classified = 0;
    for (int i = 0; i < N_test; i++)
        if (final_predictions[i] >= 0) { classified++; if (final_predictions[i] == test_labels[i]) correct++; }

    double acc = classified > 0 ? 100.0 * correct / classified : 0;
    cout << "\n=== Results ===" << endl;
    cout << "Accuracy: " << fixed << setprecision(2) << acc << "%" << endl;
    cout << "Timing: RFF=" << fixed << setprecision(1) << rff_time << "ms"
         << " Train="  << svm_train_time << "ms"
         << " Predict=" << svm_pred_time  << "ms"
         << " DBSCAN=" << dbscan_time    << "ms" << endl;

    // Train accuracy (on subset of RFF-transformed train data)
    int sample_n = min(N_train, 20000);
    int step = max(1, N_train / sample_n);
    vector<int> train_preds_sample, train_labels_sample;
    vector<vector<float>> rff_train_sample;
    for (int i = 0; i < N_train && (int)rff_train_sample.size() < sample_n; i += step) {
        rff_train_sample.push_back(rff_train[i]);
        train_labels_sample.push_back(train_labels[i]);
    }
    vector<float> dummy_confs;
    gpu_ovr_predict(rff_train_sample, model, train_preds_sample, dummy_confs);
    int tr_correct = 0;
    for (int i = 0; i < (int)train_labels_sample.size(); i++)
        if (train_preds_sample[i] == train_labels_sample[i]) tr_correct++;
    double train_acc = 100.0 * tr_correct / (int)train_labels_sample.size();
    cout << "Train Accuracy: " << fixed << setprecision(2) << train_acc << "%" << endl;

    // Save outputs
    model.save("cuda_ovr_svm_model.txt");
    { ofstream fp("cuda_predictions.csv"); for (int p : final_predictions) fp << p << "\n"; }
    { ofstream fa("cuda_accuracy.csv"); fa << fixed << setprecision(4) << train_acc << "," << acc << "\n"; }
    cout << "Predictions saved to cuda_predictions.csv" << endl;

    return 0;
}
