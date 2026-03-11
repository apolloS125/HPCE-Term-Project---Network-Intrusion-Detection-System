// common.h - Shared data structures, SVM & DBSCAN algorithms, I/O, timing
#ifndef COMMON_H
#define COMMON_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <cstdlib>
#include <map>

using namespace std;

// ===================== DATA I/O =====================
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

// ===================== TIMER =====================
struct Timer {
    chrono::high_resolution_clock::time_point start_time;
    void start() { start_time = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(end - start_time).count();
    }
    double elapsed_sec() { return elapsed_ms() / 1000.0; }
};

// ===================== METRICS =====================
void print_metrics(const string& technique, double time_sec,
                   long long flop_count, int n_samples, int n_features) {
    double gflops = (flop_count / 1e9) / time_sec;
    cout << "\n========================================" << endl;
    cout << "  Technique:  " << technique << endl;
    cout << "  Samples:    " << n_samples << endl;
    cout << "  Features:   " << n_features << endl;
    cout << "  Time:       " << fixed << setprecision(3) << time_sec * 1000 << " ms" << endl;
    cout << "  FLOP Count: " << flop_count << endl;
    cout << "  GFLOPS:     " << fixed << setprecision(4) << gflops << endl;
    cout << "========================================\n" << endl;
}

double accuracy(const vector<int>& pred, const vector<int>& truth) {
    int correct = 0;
    for (size_t i = 0; i < pred.size(); i++)
        if (pred[i] == truth[i]) correct++;
    return (double)correct / pred.size() * 100.0;
}

// Auto-detect number of classes from label vector
int detect_n_classes(const vector<int>& labels) {
    int max_label = 0;
    for (int l : labels) if (l > max_label) max_label = l;
    return max_label + 1;
}

void print_confusion_matrix(const vector<int>& pred, const vector<int>& truth, int n_classes) {
    vector<vector<int>> cm(n_classes, vector<int>(n_classes, 0));
    for (size_t i = 0; i < pred.size(); i++)
        if (pred[i] >= 0 && pred[i] < n_classes && truth[i] >= 0 && truth[i] < n_classes)
            cm[truth[i]][pred[i]]++;
    
    cout << "Confusion Matrix (rows=actual, cols=predicted):" << endl;
    cout << setw(10) << "";
    for (int j = 0; j < n_classes; j++) cout << setw(8) << ("C" + to_string(j));
    cout << endl;
    for (int i = 0; i < n_classes; i++) {
        cout << setw(10) << ("C" + to_string(i));
        for (int j = 0; j < n_classes; j++) cout << setw(8) << cm[i][j];
        cout << endl;
    }
}

// ===================== RBF KERNEL =====================
inline double rbf_kernel(const vector<double>& a, const vector<double>& b, double gamma) {
    double dist_sq = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        dist_sq += d * d;
    }
    return exp(-gamma * dist_sq);
}

// ===================== EUCLIDEAN DISTANCE =====================
inline double euclidean_dist(const vector<double>& a, const vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return sqrt(sum);
}

// ===================== SIMPLE SVM (1-vs-1 with RBF) =====================
// Simplified SVM using kernel perceptron (for educational purposes)
// For production use SMO, but this is sufficient for HPCE performance comparison
struct SVMModel {
    vector<vector<double>> support_vectors;
    vector<double> alphas;
    double bias;
    double gamma;
    int class_a, class_b;
    vector<int> epoch_errors;
    vector<int> val_epoch_errors;
    int n_samples = 0;
    int n_val_samples = 0;
};

struct MultiClassSVM {
    vector<SVMModel> models;
    int n_classes;
    double gamma;
    int max_iter;
    double lr;

    MultiClassSVM(int nc = 5, double g = 0.1, int mi = 100, double learning_rate = 0.01)
        : n_classes(nc), gamma(g), max_iter(mi), lr(learning_rate) {}

    // Train binary SVM using simplified SGD on hinge loss with RBF kernel
    // Anti-overfitting: 80/20 train/val split, early stopping, weight decay
    SVMModel train_binary(const vector<vector<double>>& data, const vector<int>& labels,
                          int ca, int cb) {
        SVMModel model;
        model.gamma = gamma;
        model.class_a = ca;
        model.class_b = cb;

        // Extract relevant samples
        vector<int> indices;
        for (size_t i = 0; i < labels.size(); i++)
            if (labels[i] == ca || labels[i] == cb) indices.push_back(i);

        // Subsample if too many samples (limit to 10000 per class pair)
        int max_samples = 10000;
        if ((int)indices.size() > max_samples) {
            // Deterministic shuffle using simple stride sampling
            vector<int> sampled;
            int step = indices.size() / max_samples;
            for (int i = 0; i < max_samples; i++)
                sampled.push_back(indices[i * step]);
            indices = sampled;
        }

        int n = indices.size();
        if (n == 0) return model;

        // ---- Train/Validation split (80/20) ----
        bool use_val = (n >= 10);
        int n_train = use_val ? max(2, (int)(n * 0.8)) : n;
        int n_val = n - n_train;
        model.n_samples = n_train;
        model.n_val_samples = n_val;

        // Store all as support vectors (kernel perceptron approach)
        model.support_vectors.resize(n);
        model.alphas.resize(n, 0.0);
        model.bias = 0.0;

        vector<double> y(n);
        for (int i = 0; i < n; i++) {
            model.support_vectors[i] = data[indices[i]];
            y[i] = (labels[indices[i]] == ca) ? 1.0 : -1.0;
        }

        // Pre-compute kernel matrix to avoid redundant O(D) computations
        vector<vector<double>> K(n, vector<double>(n));
        for (int i = 0; i < n; i++) {
            K[i][i] = 1.0;  // rbf_kernel(x, x) = exp(0) = 1
            for (int j = i + 1; j < n; j++) {
                double k = rbf_kernel(model.support_vectors[i], model.support_vectors[j], gamma);
                K[i][j] = k;
                K[j][i] = k;
            }
        }

        // ---- Early stopping state ----
        double decay = 0.001;    // L2 weight decay
        int patience = 15;
        int best_val_errors = n_val + 1;
        int wait = 0;
        vector<double> best_alphas = model.alphas;
        double best_bias = model.bias;

        // Training loop (kernel perceptron) - train on [0, n_train), validate on [n_train, n)
        for (int iter = 0; iter < max_iter; iter++) {
            // --- Train phase: only update on training samples ---
            int errors = 0;
            for (int i = 0; i < n_train; i++) {
                double score = model.bias;
                for (int j = 0; j < n_train; j++) {
                    if (model.alphas[j] != 0.0) {
                        score += model.alphas[j] * K[j][i];
                    }
                }
                if (y[i] * score <= 0) {  // misclassified
                    model.alphas[i] += lr * y[i];
                    model.bias += lr * y[i];
                    errors++;
                }
            }

            // --- Weight decay (L2 regularization) ---
            for (int j = 0; j < n_train; j++)
                model.alphas[j] *= (1.0 - decay);

            // --- Validation phase ---
            int val_errors = 0;
            if (use_val) {
                for (int i = n_train; i < n; i++) {
                    double score = model.bias;
                    for (int j = 0; j < n_train; j++) {
                        if (model.alphas[j] != 0.0)
                            score += model.alphas[j] * K[j][i];
                    }
                    if (y[i] * score <= 0) val_errors++;
                }
            }

            cout << "  SVM(" << ca << "v" << cb << ") Epoch " << (iter+1) << "/" << max_iter
                 << " | Train Errors: " << errors << "/" << n_train;
            if (use_val)
                cout << " | Val Errors: " << val_errors << "/" << n_val;
            cout << endl;

            model.epoch_errors.push_back(errors);
            model.val_epoch_errors.push_back(val_errors);

            // --- Early stopping ---
            if (use_val) {
                if (val_errors < best_val_errors) {
                    best_val_errors = val_errors;
                    best_alphas = vector<double>(model.alphas.begin(), model.alphas.begin() + n_train);
                    best_bias = model.bias;
                    wait = 0;
                } else {
                    wait++;
                    if (wait >= patience) {
                        cout << "  Early stopping at epoch " << (iter+1)
                             << " (best val errors=" << best_val_errors << ")" << endl;
                        // Restore best weights
                        for (int j = 0; j < n_train; j++) model.alphas[j] = best_alphas[j];
                        model.bias = best_bias;
                        break;
                    }
                }
            } else {
                if (errors == 0) break;
            }
        }

        // Trim model to only training support vectors (val alphas are 0)
        model.support_vectors.resize(n_train);
        model.alphas.resize(n_train);
        return model;
    }

    // Predict single sample with binary model + confidence
    pair<int, double> predict_binary(const SVMModel& model, const vector<double>& x) {
        double score = model.bias;
        for (size_t j = 0; j < model.support_vectors.size(); j++) {
            if (model.alphas[j] != 0.0) {
                score += model.alphas[j] * rbf_kernel(model.support_vectors[j], x, model.gamma);
            }
        }
        int pred = (score >= 0) ? model.class_a : model.class_b;
        return {pred, fabs(score)};
    }

    // Train all 1-vs-1 classifiers
    long long train(const vector<vector<double>>& data, const vector<int>& labels) {
        Timer t; t.start();
        models.clear();
        long long total_flops = 0;
        for (int i = 0; i < n_classes; i++) {
            for (int j = i + 1; j < n_classes; j++) {
                models.push_back(train_binary(data, labels, i, j));
                // Count training FLOPs: iterations * n * n * (3D + 2)
                int n_sv = models.back().support_vectors.size();
                int D = data[0].size();
                total_flops += (long long)max_iter * n_sv * n_sv * (3 * D + 2);
            }
        }
        return total_flops;
    }

    // Predict with confidence (returns label + confidence score)
    pair<int, double> predict_one(const vector<double>& x) {
        vector<int> votes(n_classes, 0);
        vector<double> scores(n_classes, 0.0);
        for (auto& model : models) {
            auto [pred, conf] = predict_binary(model, x);
            votes[pred]++;
            scores[pred] += conf;
        }
        int best = max_element(votes.begin(), votes.end()) - votes.begin();
        double confidence = scores[best] / max(1, votes[best]);
        return {best, confidence};
    }

    // Predict batch - returns predictions, confidences, and FLOP count
    struct PredResult {
        vector<int> predictions;
        vector<double> confidences;
        long long flops;
    };

    PredResult predict(const vector<vector<double>>& data) {
        PredResult result;
        result.predictions.resize(data.size());
        result.confidences.resize(data.size());
        result.flops = 0;

        int total_sv = 0;
        for (auto& m : models) total_sv += m.support_vectors.size();
        int D = data[0].size();

        for (size_t i = 0; i < data.size(); i++) {
            auto [pred, conf] = predict_one(data[i]);
            result.predictions[i] = pred;
            result.confidences[i] = conf;
        }
        // FLOPs: n_test * total_sv * (3D + 2)
        result.flops = (long long)data.size() * total_sv * (3 * D + 2);
        return result;
    }
};

// ===================== DBSCAN =====================
struct DBSCANResult {
    vector<int> cluster_labels;  // -1 = noise
    int n_clusters;
    int n_noise;
    long long flops;
};

// Compute pairwise distance matrix
vector<vector<double>> compute_distance_matrix(const vector<vector<double>>& data, long long& flops) {
    int N = data.size();
    int D = data[0].size();
    vector<vector<double>> dist(N, vector<double>(N, 0.0));

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double d = euclidean_dist(data[i], data[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    // FLOPs: N*(N-1)/2 * 3D (sub + mul + add per feature) + sqrt
    flops = (long long)N * (N - 1) / 2 * (3 * D + 1);
    return dist;
}

DBSCANResult dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int N = data.size();
    DBSCANResult result;
    result.cluster_labels.assign(N, -2);  // -2 = unvisited
    result.n_clusters = 0;

    long long dist_flops = 0;
    auto dist_matrix = compute_distance_matrix(data, dist_flops);
    result.flops = dist_flops;

    auto region_query = [&](int p) -> vector<int> {
        vector<int> neighbors;
        for (int i = 0; i < N; i++)
            if (dist_matrix[p][i] <= eps) neighbors.push_back(i);
        return neighbors;
    };

    for (int i = 0; i < N; i++) {
        if (result.cluster_labels[i] != -2) continue;

        auto neighbors = region_query(i);
        if ((int)neighbors.size() < min_pts) {
            result.cluster_labels[i] = -1;  // noise
            continue;
        }

        int cluster_id = result.n_clusters++;
        result.cluster_labels[i] = cluster_id;

        vector<int> seed_set(neighbors.begin(), neighbors.end());
        for (size_t j = 0; j < seed_set.size(); j++) {
            int q = seed_set[j];
            if (result.cluster_labels[q] == -1)
                result.cluster_labels[q] = cluster_id;
            if (result.cluster_labels[q] != -2) continue;

            result.cluster_labels[q] = cluster_id;
            auto q_neighbors = region_query(q);
            if ((int)q_neighbors.size() >= min_pts) {
                for (int nn : q_neighbors) {
                    if (find(seed_set.begin(), seed_set.end(), nn) == seed_set.end())
                        seed_set.push_back(nn);
                }
            }
        }
    }

    result.n_noise = count(result.cluster_labels.begin(), result.cluster_labels.end(), -1);
    return result;
}

// ===================== FULL PIPELINE =====================
struct PipelineResult {
    vector<int> final_predictions;
    vector<int> svm_confident;
    vector<int> uncertain_indices;
    int n_noise;
    double svm_time_ms;
    double dbscan_time_ms;
    double total_time_ms;
    long long svm_train_flops;
    long long svm_predict_flops;
    long long dbscan_flops;
    long long total_flops;
};

// ===================== MODEL SAVE / LOAD =====================
void save_svm_model(const MultiClassSVM& svm, const string& filename) {
    ofstream f(filename);
    if (!f.is_open()) { cerr << "Cannot write " << filename << endl; return; }
    f << svm.n_classes << " " << svm.gamma << " " << svm.max_iter << " " << svm.lr << "\n";
    f << svm.models.size() << "\n";
    for (auto& m : svm.models) {
        f << m.class_a << " " << m.class_b << " " << m.gamma << " " << m.bias << "\n";
        int nsv = m.support_vectors.size();
        int D = nsv > 0 ? m.support_vectors[0].size() : 0;
        f << nsv << " " << D << "\n";
        for (int i = 0; i < nsv; i++) {
            f << m.alphas[i];
            for (int j = 0; j < D; j++) f << " " << m.support_vectors[i][j];
            f << "\n";
        }
    }
    f.close();
    cout << "Model saved to " << filename << endl;
}

void save_predictions(const vector<int>& preds, const string& filename) {
    ofstream f(filename);
    if (!f.is_open()) { cerr << "Cannot write " << filename << endl; return; }
    for (int p : preds) f << p << "\n";
    f.close();
    cout << "Predictions saved to " << filename << endl;
}

void save_epoch_errors(const MultiClassSVM& svm, const string& filename) {
    ofstream f(filename);
    if (!f.is_open()) { cerr << "Cannot write " << filename << endl; return; }
    f << "class_a,class_b,epoch,train_errors,n_train,val_errors,n_val\n";
    for (auto& m : svm.models)
        for (int e = 0; e < (int)m.epoch_errors.size(); e++)
            f << m.class_a << "," << m.class_b << "," << (e+1) << ","
              << m.epoch_errors[e] << "," << m.n_samples << ","
              << (e < (int)m.val_epoch_errors.size() ? m.val_epoch_errors[e] : 0)
              << "," << m.n_val_samples << "\n";
    f.close();
    cout << "Epoch errors saved to " << filename << endl;
}

double compute_train_accuracy(MultiClassSVM& svm, const vector<vector<double>>& data,
                               const vector<int>& labels, int max_samples = 10000) {
    int n = (int)data.size();
    int step = (n > max_samples) ? (n / max_samples) : 1;
    vector<vector<double>> sample_data;
    vector<int> sample_labels;
    for (int i = 0; i < n && (int)sample_data.size() < max_samples; i += step) {
        sample_data.push_back(data[i]);
        sample_labels.push_back(labels[i]);
    }
    auto result = svm.predict(sample_data);
    int correct = 0;
    for (int i = 0; i < (int)result.predictions.size(); i++)
        if (result.predictions[i] == sample_labels[i]) correct++;
    return 100.0 * correct / (int)result.predictions.size();
}

void save_dbscan_model(const DBSCANResult& db, const vector<vector<double>>& data,
                       double eps, int min_pts, const string& filename) {
    ofstream f(filename);
    if (!f.is_open()) { cerr << "Cannot write " << filename << endl; return; }
    int N = data.size();
    int D = N > 0 ? data[0].size() : 0;
    f << eps << " " << min_pts << "\n";
    f << db.n_clusters << " " << db.n_noise << "\n";
    f << N << " " << D << "\n";
    for (int i = 0; i < N; i++) {
        f << db.cluster_labels[i];
        for (int j = 0; j < D; j++) f << " " << data[i][j];
        f << "\n";
    }
    f.close();
    cout << "DBSCAN model saved to " << filename << endl;
}

// ===================== MODEL LOAD =====================
MultiClassSVM load_svm_model(const string& filename) {
    ifstream f(filename);
    if (!f.is_open()) { cerr << "Cannot open " << filename << endl; exit(1); }
    MultiClassSVM svm;
    int n_models;
    f >> svm.n_classes >> svm.gamma >> svm.max_iter >> svm.lr;
    f >> n_models;
    svm.models.resize(n_models);
    for (int k = 0; k < n_models; k++) {
        auto& m = svm.models[k];
        int nsv, D;
        f >> m.class_a >> m.class_b >> m.gamma >> m.bias;
        f >> nsv >> D;
        m.support_vectors.resize(nsv, vector<double>(D));
        m.alphas.resize(nsv);
        for (int i = 0; i < nsv; i++) {
            f >> m.alphas[i];
            for (int j = 0; j < D; j++) f >> m.support_vectors[i][j];
        }
    }
    f.close();
    cout << "SVM model loaded from " << filename << endl;
    return svm;
}

struct DBSCANModel {
    double eps;
    int min_pts;
    int n_clusters;
    int n_noise;
    vector<int> cluster_labels;
    vector<vector<double>> core_data;
};

DBSCANModel load_dbscan_model(const string& filename) {
    ifstream f(filename);
    if (!f.is_open()) { cerr << "Cannot open " << filename << endl; exit(1); }
    DBSCANModel model;
    int N, D;
    f >> model.eps >> model.min_pts;
    f >> model.n_clusters >> model.n_noise;
    f >> N >> D;
    model.cluster_labels.resize(N);
    model.core_data.resize(N, vector<double>(D));
    for (int i = 0; i < N; i++) {
        f >> model.cluster_labels[i];
        for (int j = 0; j < D; j++) f >> model.core_data[i][j];
    }
    f.close();
    cout << "DBSCAN model loaded from " << filename << endl;
    return model;
}

#endif // COMMON_H
