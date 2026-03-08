// openmp.cpp - OpenMP parallel SVM + DBSCAN pipeline
// Compile: g++ -O2 -fopenmp -o openmp openmp.cpp -lm
#include "common.h"
#include <omp.h>

int NUM_THREADS = 8;

// ===== OpenMP parallel distance matrix =====
vector<vector<double>> omp_distance_matrix(const vector<vector<double>>& data, long long& flops) {
    int N = data.size();
    int D = data[0].size();
    vector<vector<double>> dist(N, vector<double>(N, 0.0));

    #pragma omp parallel for schedule(dynamic, 16) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < D; k++) {
                double d = data[i][k] - data[j][k];
                sum += d * d;
            }
            double d = sqrt(sum);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    flops = (long long)N * (N - 1) / 2 * (3 * D + 1);
    return dist;
}

// ===== OpenMP parallel SVM prediction =====
MultiClassSVM::PredResult omp_predict(MultiClassSVM& svm, const vector<vector<double>>& data) {
    int N = data.size();
    MultiClassSVM::PredResult result;
    result.predictions.resize(N);
    result.confidences.resize(N);

    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        auto [pred, conf] = svm.predict_one(data[i]);
        result.predictions[i] = pred;
        result.confidences[i] = conf;
    }

    int total_sv = 0;
    for (auto& m : svm.models) total_sv += m.support_vectors.size();
    int D = data[0].size();
    result.flops = (long long)N * total_sv * (3 * D + 2);
    return result;
}

// ===== OpenMP DBSCAN =====
DBSCANResult omp_dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int N = data.size();
    DBSCANResult result;
    result.cluster_labels.assign(N, -2);
    result.n_clusters = 0;

    long long dist_flops = 0;
    auto dist_matrix = omp_distance_matrix(data, dist_flops);
    result.flops = dist_flops;

    // DBSCAN clustering (sequential BFS)
    for (int i = 0; i < N; i++) {
        if (result.cluster_labels[i] != -2) continue;
        vector<int> neighbors;
        for (int j = 0; j < N; j++)
            if (dist_matrix[i][j] <= eps) neighbors.push_back(j);

        if ((int)neighbors.size() < min_pts) { result.cluster_labels[i] = -1; continue; }
        int cid = result.n_clusters++;
        result.cluster_labels[i] = cid;
        vector<int> seed(neighbors.begin(), neighbors.end());
        for (size_t s = 0; s < seed.size(); s++) {
            int q = seed[s];
            if (result.cluster_labels[q] == -1) result.cluster_labels[q] = cid;
            if (result.cluster_labels[q] != -2) continue;
            result.cluster_labels[q] = cid;
            vector<int> qn;
            for (int j = 0; j < N; j++)
                if (dist_matrix[q][j] <= eps) qn.push_back(j);
            if ((int)qn.size() >= min_pts)
                for (int nn : qn)
                    if (find(seed.begin(), seed.end(), nn) == seed.end())
                        seed.push_back(nn);
        }
    }
    result.n_noise = count(result.cluster_labels.begin(), result.cluster_labels.end(), -1);
    return result;
}

int main(int argc, char* argv[]) {
    if (argc > 1) NUM_THREADS = atoi(argv[1]);
    omp_set_num_threads(NUM_THREADS);
    cout << "=== Network IDS: OpenMP (" << NUM_THREADS << " threads) ===" << endl;

    auto train_data = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data = load_csv("data/test_data.csv");
    auto test_labels = load_labels("data/test_labels.csv");

    // Subsample to fit in memory
    const int MAX_TRAIN = 10000;
    const int MAX_TEST = 5000;
    if ((int)train_data.size() > MAX_TRAIN) {
        int step = train_data.size() / MAX_TRAIN;
        vector<vector<double>> sd; vector<int> sl;
        for (int i = 0; i < MAX_TRAIN; i++) { sd.push_back(train_data[i * step]); sl.push_back(train_labels[i * step]); }
        train_data = sd; train_labels = sl;
    }
    if ((int)test_data.size() > MAX_TEST) {
        int step = test_data.size() / MAX_TEST;
        vector<vector<double>> sd; vector<int> sl;
        for (int i = 0; i < MAX_TEST; i++) { sd.push_back(test_data[i * step]); sl.push_back(test_labels[i * step]); }
        test_data = sd; test_labels = sl;
    }

    int N_train = train_data.size(), N_test = test_data.size(), D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    cout << "Train: " << N_train << " | Test: " << N_test << " | Features: " << D << " | Classes: " << n_classes << endl;

    Timer total_timer; total_timer.start();
    long long total_flops = 0;

    // SVM Training
    MultiClassSVM svm(n_classes, 0.1, 50, 0.01);
    Timer t1; t1.start();
    long long train_flops = svm.train(train_data, train_labels);
    double svm_train_time = t1.elapsed_ms();
    total_flops += train_flops;
    cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    // SVM Prediction (OpenMP parallel)
    Timer t2; t2.start();
    auto pred_result = omp_predict(svm, test_data);
    double svm_pred_time = t2.elapsed_ms();
    total_flops += pred_result.flops;
    cout << "SVM prediction: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;

    // Split
    double threshold = 0.3;
    vector<int> final_predictions(N_test);
    vector<int> uncertain_idx;
    vector<vector<double>> uncertain_data;

    for (int i = 0; i < N_test; i++) {
        if (pred_result.confidences[i] >= threshold) {
            final_predictions[i] = pred_result.predictions[i];
        } else {
            final_predictions[i] = -1;
            uncertain_idx.push_back(i);
            uncertain_data.push_back(test_data[i]);
        }
    }
    cout << "Confident: " << (N_test - (int)uncertain_idx.size()) << " | Uncertain: " << uncertain_idx.size() << endl;

    // DBSCAN
    double dbscan_time = 0;
    if (uncertain_data.size() > 1) {
        Timer t3; t3.start();
        auto db = omp_dbscan(uncertain_data, 1.5, D + 1);
        dbscan_time = t3.elapsed_ms();
        total_flops += db.flops;
        cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms | Clusters: " << db.n_clusters << " | Noise: " << db.n_noise << endl;
        for (size_t i = 0; i < uncertain_idx.size(); i++)
            final_predictions[uncertain_idx[i]] = (db.cluster_labels[i] == -1) ? -1 : pred_result.predictions[uncertain_idx[i]];
    }

    double total_time = total_timer.elapsed_ms();

    int correct = 0, classified = 0;
    for (int i = 0; i < N_test; i++) {
        if (final_predictions[i] >= 0) { classified++; if (final_predictions[i] == test_labels[i]) correct++; }
    }
    cout << "\nAccuracy: " << fixed << setprecision(2) << (100.0 * correct / max(1, classified)) << "%" << endl;

    print_metrics("OpenMP (" + to_string(NUM_THREADS) + " threads)", total_time / 1000.0, total_flops, N_test, D);
    cout << "Timing: Train=" << fixed << setprecision(1) << svm_train_time << "ms Predict=" << svm_pred_time << "ms DBSCAN=" << dbscan_time << "ms Total=" << total_time << "ms" << endl;
    return 0;
}
