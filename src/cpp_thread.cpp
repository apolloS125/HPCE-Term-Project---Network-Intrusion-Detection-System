// cpp_thread.cpp - C++ std::thread parallel SVM + DBSCAN pipeline
// Compile: g++ -O2 -pthread -o cpp_thread cpp_thread.cpp -lm
#include "common.h"
#include <thread>
#include <mutex>
#include <barrier>
#include <functional>

int NUM_THREADS = 8;

// ===== Parallel distance matrix computation =====
void compute_distance_chunk(const vector<vector<double>>& data,
                            vector<vector<double>>& dist,
                            int start_row, int end_row) {
    int N = data.size();
    int D = data[0].size();
    for (int i = start_row; i < end_row; i++) {
        for (int j = i + 1; j < N; j++) {
            double d = euclidean_dist(data[i], data[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
}

vector<vector<double>> parallel_distance_matrix(const vector<vector<double>>& data,
                                                 long long& flops) {
    int N = data.size();
    int D = data[0].size();
    vector<vector<double>> dist(N, vector<double>(N, 0.0));

    vector<thread> threads;
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        int start = t * chunk;
        int end = min(start + chunk, N);
        if (start >= N) break;
        threads.emplace_back(compute_distance_chunk, ref(data), ref(dist), start, end);
    }
    for (auto& th : threads) th.join();

    flops = (long long)N * (N - 1) / 2 * (3 * D + 1);
    return dist;
}

// ===== Parallel SVM prediction =====
void predict_chunk(MultiClassSVM& svm, const vector<vector<double>>& data,
                   vector<int>& predictions, vector<double>& confidences,
                   int start, int end) {
    for (int i = start; i < end; i++) {
        auto [pred, conf] = svm.predict_one(data[i]);
        predictions[i] = pred;
        confidences[i] = conf;
    }
}

MultiClassSVM::PredResult parallel_predict(MultiClassSVM& svm,
                                            const vector<vector<double>>& data) {
    int N = data.size();
    MultiClassSVM::PredResult result;
    result.predictions.resize(N);
    result.confidences.resize(N);

    vector<thread> threads;
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; t++) {
        int start = t * chunk;
        int end = min(start + chunk, N);
        if (start >= N) break;
        threads.emplace_back(predict_chunk, ref(svm), ref(data),
                            ref(result.predictions), ref(result.confidences), start, end);
    }
    for (auto& th : threads) th.join();

    int total_sv = 0;
    for (auto& m : svm.models) total_sv += m.support_vectors.size();
    int D = data[0].size();
    result.flops = (long long)N * total_sv * (3 * D + 2);
    return result;
}

// ===== Parallel DBSCAN =====
DBSCANResult parallel_dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int N = data.size();
    DBSCANResult result;
    result.cluster_labels.assign(N, -2);
    result.n_clusters = 0;

    long long dist_flops = 0;
    auto dist_matrix = parallel_distance_matrix(data, dist_flops);
    result.flops = dist_flops;

    // DBSCAN clustering (sequential - hard to parallelize BFS)
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
            result.cluster_labels[i] = -1;
            continue;
        }
        int cluster_id = result.n_clusters++;
        result.cluster_labels[i] = cluster_id;
        vector<int> seed_set(neighbors.begin(), neighbors.end());
        for (size_t j = 0; j < seed_set.size(); j++) {
            int q = seed_set[j];
            if (result.cluster_labels[q] == -1) result.cluster_labels[q] = cluster_id;
            if (result.cluster_labels[q] != -2) continue;
            result.cluster_labels[q] = cluster_id;
            auto q_neighbors = region_query(q);
            if ((int)q_neighbors.size() >= min_pts)
                for (int nn : q_neighbors)
                    if (find(seed_set.begin(), seed_set.end(), nn) == seed_set.end())
                        seed_set.push_back(nn);
        }
    }
    result.n_noise = count(result.cluster_labels.begin(), result.cluster_labels.end(), -1);
    return result;
}

int main(int argc, char* argv[]) {
    if (argc > 1) NUM_THREADS = atoi(argv[1]);
    cout << "=== Network IDS: C++ Thread (" << NUM_THREADS << " threads) ===" << endl;

    auto train_data = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data = load_csv("data/test_data.csv");
    auto test_labels = load_labels("data/test_labels.csv");

    int N_train = train_data.size(), N_test = test_data.size(), D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    cout << "Train: " << N_train << " | Test: " << N_test << " | Features: " << D << " | Classes: " << n_classes << endl;

    Timer total_timer; total_timer.start();
    long long total_flops = 0;

    // SVM Training (sequential - hard to parallelize SGD)
    cout << "\n--- SVM Training ---" << endl;
    MultiClassSVM svm(n_classes, 0.1, 50, 0.01);
    Timer t1; t1.start();
    long long train_flops = svm.train(train_data, train_labels);
    double svm_train_time = t1.elapsed_ms();
    total_flops += train_flops;
    cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    // SVM Prediction (parallel)
    cout << "\n--- SVM Prediction (parallel) ---" << endl;
    Timer t2; t2.start();
    auto pred_result = parallel_predict(svm, test_data);
    double svm_pred_time = t2.elapsed_ms();
    total_flops += pred_result.flops;
    cout << "SVM prediction: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;

    // Split confident vs uncertain
    double confidence_threshold = 0.3;
    vector<int> final_predictions(N_test);
    vector<int> uncertain_indices;
    vector<vector<double>> uncertain_data;
    int confident_count = 0;

    for (int i = 0; i < N_test; i++) {
        if (pred_result.confidences[i] >= confidence_threshold) {
            final_predictions[i] = pred_result.predictions[i];
            confident_count++;
        } else {
            final_predictions[i] = -1;
            uncertain_indices.push_back(i);
            uncertain_data.push_back(test_data[i]);
        }
    }
    cout << "Confident: " << confident_count << " | Uncertain: " << uncertain_indices.size() << endl;

    // DBSCAN (parallel distance matrix)
    int n_noise = 0;
    double dbscan_time = 0;
    if (!uncertain_data.empty() && uncertain_data.size() > 1) {
        cout << "\n--- DBSCAN (parallel distance matrix) ---" << endl;
        Timer t3; t3.start();
        auto dbscan_result = parallel_dbscan(uncertain_data, 1.5, D + 1);
        dbscan_time = t3.elapsed_ms();
        total_flops += dbscan_result.flops;
        n_noise = dbscan_result.n_noise;
        cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms" << endl;
        cout << "Clusters: " << dbscan_result.n_clusters << " | Noise: " << n_noise << endl;

        for (size_t i = 0; i < uncertain_indices.size(); i++) {
            int idx = uncertain_indices[i];
            final_predictions[idx] = (dbscan_result.cluster_labels[i] == -1) ? -1 : pred_result.predictions[idx];
        }
    }

    double total_time = total_timer.elapsed_ms();

    // Results
    int correct = 0, total_classified = 0;
    for (int i = 0; i < N_test; i++) {
        if (final_predictions[i] >= 0) {
            total_classified++;
            if (final_predictions[i] == test_labels[i]) correct++;
        }
    }
    double acc = (total_classified > 0) ? (100.0 * correct / total_classified) : 0;
    cout << "\n--- Results ---" << endl;
    cout << "Accuracy: " << fixed << setprecision(2) << acc << "%" << endl;
    cout << "Unknown attacks: " << (N_test - total_classified) << endl;

    print_metrics("C++ Thread (" + to_string(NUM_THREADS) + " threads)", total_time / 1000.0, total_flops, N_test, D);

    cout << "Timing Breakdown:" << endl;
    cout << "  SVM Train:   " << fixed << setprecision(1) << svm_train_time << " ms" << endl;
    cout << "  SVM Predict: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;
    cout << "  DBSCAN:      " << fixed << setprecision(1) << dbscan_time << " ms" << endl;
    cout << "  Total:       " << fixed << setprecision(1) << total_time << " ms" << endl;

    return 0;
}
