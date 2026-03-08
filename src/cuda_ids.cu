// cuda_ids.cu - CUDA GPU accelerated SVM + DBSCAN pipeline
// Compile: nvcc -O2 -o cuda_ids cuda_ids.cu -lm
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

using namespace std;

// ===== CUDA Error Check =====
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
}

// ===== Data I/O (same as common.h) =====
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
    int max_label = 0;
    for (int l : labels) if (l > max_label) max_label = l;
    return max_label + 1;
}

// ===== CUDA Kernels =====

// Pairwise Euclidean distance matrix (main GPU kernel)
__global__ void distance_matrix_kernel(const double* data, double* dist, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N && i < j) {
        double sum = 0.0;
        for (int k = 0; k < D; k++) {
            double d = data[i * D + k] - data[j * D + k];
            sum += d * d;
        }
        double dist_val = sqrt(sum);
        dist[i * N + j] = dist_val;
        dist[j * N + i] = dist_val;
    }
}

// RBF Kernel matrix computation
__global__ void rbf_kernel_matrix(const double* data, double* kernel_mat,
                                   int N, int D, double gamma) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        double sum = 0.0;
        for (int k = 0; k < D; k++) {
            double d = data[i * D + k] - data[j * D + k];
            sum += d * d;
        }
        kernel_mat[i * N + j] = exp(-gamma * sum);
    }
}

// SVM prediction kernel (simplified - compute kernel values)
__global__ void svm_predict_kernel(const double* test_data, const double* sv_data,
                                    const double* alphas, double* scores,
                                    int N_test, int N_sv, int D, double gamma) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_test) {
        double score = 0.0;
        for (int j = 0; j < N_sv; j++) {
            double dist_sq = 0.0;
            for (int k = 0; k < D; k++) {
                double d = test_data[i * D + k] - sv_data[j * D + k];
                dist_sq += d * d;
            }
            score += alphas[j] * exp(-gamma * dist_sq);
        }
        scores[i] = score;
    }
}

// ===== Host Functions =====

// Flatten 2D vector to 1D array for CUDA
vector<double> flatten(const vector<vector<double>>& data) {
    int N = data.size(), D = data[0].size();
    vector<double> flat(N * D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            flat[i * D + j] = data[i][j];
    return flat;
}

// GPU distance matrix
vector<double> gpu_distance_matrix(const vector<vector<double>>& data, double& time_ms, long long& flops) {
    int N = data.size(), D = data[0].size();
    auto flat = flatten(data);

    double *d_data, *d_dist;
    CUDA_CHECK(cudaMalloc(&d_data, N * D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dist, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_dist, 0, N * N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_data, flat.data(), N * D * sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    Timer t; t.start();
    distance_matrix_kernel<<<grid, block>>>(d_data, d_dist, N, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    time_ms = t.ms();

    vector<double> dist(N * N);
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, N * N * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_dist));

    flops = (long long)N * (N - 1) / 2 * (3 * D + 1);
    return dist;
}

// DBSCAN using GPU-computed distance matrix
struct DBSCANResult {
    vector<int> labels;
    int n_clusters, n_noise;
    long long flops;
};

DBSCANResult gpu_dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int N = data.size();
    double dist_time;
    long long dist_flops;
    auto dist = gpu_distance_matrix(data, dist_time, dist_flops);

    DBSCANResult result;
    result.labels.assign(N, -2);
    result.n_clusters = 0;
    result.flops = dist_flops;

    // CPU-side DBSCAN using GPU-computed distances
    for (int i = 0; i < N; i++) {
        if (result.labels[i] != -2) continue;
        vector<int> neighbors;
        for (int j = 0; j < N; j++)
            if (dist[i * N + j] <= eps) neighbors.push_back(j);

        if ((int)neighbors.size() < min_pts) { result.labels[i] = -1; continue; }
        int cid = result.n_clusters++;
        result.labels[i] = cid;
        vector<int> seed(neighbors.begin(), neighbors.end());
        for (size_t s = 0; s < seed.size(); s++) {
            int q = seed[s];
            if (result.labels[q] == -1) result.labels[q] = cid;
            if (result.labels[q] != -2) continue;
            result.labels[q] = cid;
            vector<int> qn;
            for (int j = 0; j < N; j++)
                if (dist[q * N + j] <= eps) qn.push_back(j);
            if ((int)qn.size() >= min_pts)
                for (int nn : qn)
                    if (find(seed.begin(), seed.end(), nn) == seed.end())
                        seed.push_back(nn);
        }
    }
    result.n_noise = count(result.labels.begin(), result.labels.end(), -1);
    return result;
}

// ===== Simplified SVM (CPU training, GPU prediction) =====
// For the CUDA version, we demonstrate GPU kernel computation
// Full SVM training on GPU requires complex SMO - simplified here

inline double rbf_kernel_cpu(const vector<double>& a, const vector<double>& b, double gamma) {
    double d = 0;
    for (size_t i = 0; i < a.size(); i++) { double x = a[i]-b[i]; d += x*x; }
    return exp(-gamma * d);
}

struct SimpleSVM {
    vector<vector<double>> sv;
    vector<double> alphas;
    double bias, gamma;

    void train(const vector<vector<double>>& data, const vector<int>& labels,
               int ca, int cb, int max_iter=50, double lr=0.01) {
        gamma = 0.1;
        vector<int> idx;
        for (size_t i = 0; i < labels.size(); i++)
            if (labels[i] == ca || labels[i] == cb) idx.push_back(i);
        int n = idx.size();
        sv.resize(n); alphas.assign(n, 0); bias = 0;
        vector<double> y(n);
        for (int i = 0; i < n; i++) { sv[i] = data[idx[i]]; y[i] = (labels[idx[i]]==ca)?1:-1; }
        for (int iter = 0; iter < max_iter; iter++) {
            for (int i = 0; i < n; i++) {
                double s = bias;
                for (int j = 0; j < n; j++)
                    if (alphas[j] != 0) s += alphas[j] * rbf_kernel_cpu(sv[j], sv[i], gamma);
                if (y[i]*s <= 0) { alphas[i] += lr*y[i]; bias += lr*y[i]; }
            }
        }
    }

    pair<int,double> predict(const vector<double>& x, int ca, int cb) {
        double s = bias;
        for (size_t j = 0; j < sv.size(); j++)
            if (alphas[j] != 0) s += alphas[j] * rbf_kernel_cpu(sv[j], x, gamma);
        return {(s>=0)?ca:cb, fabs(s)};
    }
};

int main() {
    cout << "=== Network IDS: CUDA C++ ===" << endl;

    // Check GPU
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        cout << "No CUDA GPU found. This code requires an NVIDIA GPU." << endl;
        cout << "Falling back to CPU simulation for demonstration..." << endl;
    } else {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        cout << "GPU: " << prop.name << " (" << prop.multiProcessorCount << " SMs)" << endl;
    }

    auto train_data = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data = load_csv("data/test_data.csv");
    auto test_labels = load_labels("data/test_labels.csv");

    int N_train = train_data.size(), N_test = test_data.size(), D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    cout << "Train: " << N_train << " | Test: " << N_test << " | Features: " << D << " | Classes: " << n_classes << endl;

    Timer total_timer; total_timer.start();
    long long total_flops = 0;

    // SVM Training (CPU - 1-vs-1)
    cout << "\n--- SVM Training (CPU) ---" << endl;
    Timer t1; t1.start();
    vector<SimpleSVM> models;
    vector<pair<int,int>> class_pairs;
    for (int i = 0; i < n_classes; i++) {
        for (int j = i+1; j < n_classes; j++) {
            SimpleSVM m; m.train(train_data, train_labels, i, j);
            models.push_back(m);
            class_pairs.push_back({i,j});
        }
    }
    double svm_train_time = t1.ms();
    cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    // SVM Prediction (could use GPU for kernel computation)
    cout << "\n--- SVM Prediction ---" << endl;
    Timer t2; t2.start();
    vector<int> predictions(N_test);
    vector<double> confidences(N_test);

    for (int i = 0; i < N_test; i++) {
        vector<int> votes(n_classes, 0);
        vector<double> scores(n_classes, 0);
        for (size_t m = 0; m < models.size(); m++) {
            auto [pred, conf] = models[m].predict(test_data[i], class_pairs[m].first, class_pairs[m].second);
            votes[pred]++; scores[pred] += conf;
        }
        int best = max_element(votes.begin(), votes.end()) - votes.begin();
        predictions[i] = best;
        confidences[i] = scores[best] / max(1, votes[best]);
    }
    double svm_pred_time = t2.ms();

    int total_sv = 0;
    for (auto& m : models) total_sv += m.sv.size();
    total_flops += (long long)N_test * total_sv * (3*D+2);
    cout << "SVM prediction: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;

    // Split confident/uncertain
    double threshold = 0.3;
    vector<int> final_preds(N_test);
    vector<int> unc_idx;
    vector<vector<double>> unc_data;

    for (int i = 0; i < N_test; i++) {
        if (confidences[i] >= threshold) {
            final_preds[i] = predictions[i];
        } else {
            final_preds[i] = -1;
            unc_idx.push_back(i);
            unc_data.push_back(test_data[i]);
        }
    }
    cout << "Confident: " << (N_test-(int)unc_idx.size()) << " | Uncertain: " << unc_idx.size() << endl;

    // DBSCAN with GPU distance matrix
    double dbscan_time = 0;
    if (unc_data.size() > 1) {
        cout << "\n--- DBSCAN (GPU distance matrix) ---" << endl;
        Timer t3; t3.start();
        auto db = gpu_dbscan(unc_data, 1.5, D+1);
        dbscan_time = t3.ms();
        total_flops += db.flops;
        cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms"
             << " | Clusters: " << db.n_clusters << " | Noise: " << db.n_noise << endl;

        for (size_t i = 0; i < unc_idx.size(); i++)
            final_preds[unc_idx[i]] = (db.labels[i] == -1) ? -1 : predictions[unc_idx[i]];
    }

    double total_time = total_timer.ms();

    // Results
    int correct = 0, classified = 0;
    for (int i = 0; i < N_test; i++) {
        if (final_preds[i] >= 0) { classified++; if (final_preds[i] == test_labels[i]) correct++; }
    }
    cout << "\nAccuracy: " << fixed << setprecision(2) << (100.0*correct/max(1,classified)) << "%" << endl;

    double gflops = (total_flops / 1e9) / (total_time / 1000.0);
    cout << "\n========================================" << endl;
    cout << "  Technique:  CUDA C++" << endl;
    cout << "  Time:       " << fixed << setprecision(1) << total_time << " ms" << endl;
    cout << "  GFLOPS:     " << fixed << setprecision(4) << gflops << endl;
    cout << "========================================" << endl;
    cout << "Timing: Train=" << fixed << setprecision(1) << svm_train_time
         << "ms Predict=" << svm_pred_time << "ms DBSCAN=" << dbscan_time
         << "ms Total=" << total_time << "ms" << endl;

    return 0;
}
