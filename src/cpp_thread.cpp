// cpp_thread.cpp - C++ std::thread parallel SVM + DBSCAN pipeline
// Upgraded: RFF transformer + One-Vs-Rest Linear SVM (from cuda/src techniques)
// Compile: g++ -O2 -pthread -std=c++17 -o cpp_thread cpp_thread.cpp -lm
#include "common.h"
#include <thread>
#include <mutex>
#include <functional>

int NUM_THREADS = 8;

// ===== Parallel RFF transformation =====
void rff_transform_chunk(const RffTransformer& rff,
                         const vector<vector<double>>& data,
                         vector<vector<double>>& out,
                         int start, int end) {
    for (int i = start; i < end; i++)
        out[i] = rff.transform(data[i]);
}

vector<vector<double>> parallel_rff_transform(const RffTransformer& rff,
                                               const vector<vector<double>>& data) {
    int N = data.size();
    vector<vector<double>> out(N);
    vector<thread> threads;
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; t++) {
        int start = t * chunk;
        int end = min(start + chunk, N);
        if (start >= N) break;
        threads.emplace_back(rff_transform_chunk, cref(rff), cref(data), ref(out), start, end);
    }
    for (auto& th : threads) th.join();
    return out;
}

// ===== Parallel OvR predict =====
void predict_chunk_ovr(const OvrLinearSVM& svm,
                       const vector<vector<double>>& rff_data,
                       vector<int>& predictions, vector<double>& confidences,
                       int start, int end) {
    for (int i = start; i < end; i++) {
        auto [pred, conf] = svm.predict_one(rff_data[i]);
        predictions[i] = pred;
        confidences[i] = conf;
    }
}

OvrLinearSVM::PredResult parallel_predict_ovr(const OvrLinearSVM& svm,
                                               const vector<vector<double>>& rff_data) {
    int N = rff_data.size();
    OvrLinearSVM::PredResult result;
    result.predictions.resize(N);
    result.confidences.resize(N);
    result.flops = (long long)N * svm.n_classes * (svm.D_rff + 1);

    vector<thread> threads;
    int chunk = (N + NUM_THREADS - 1) / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; t++) {
        int start = t * chunk;
        int end = min(start + chunk, N);
        if (start >= N) break;
        threads.emplace_back(predict_chunk_ovr, cref(svm), cref(rff_data),
                             ref(result.predictions), ref(result.confidences), start, end);
    }
    for (auto& th : threads) th.join();
    return result;
}

// ===== Parallel distance matrix for DBSCAN =====
void compute_distance_chunk(const vector<vector<double>>& data,
                            vector<vector<double>>& dist,
                            int start_row, int end_row) {
    int N = data.size();
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
        threads.emplace_back(compute_distance_chunk, cref(data), ref(dist), start, end);
    }
    for (auto& th : threads) th.join();
    flops = (long long)N * (N - 1) / 2 * (3 * D + 1);
    return dist;
}

DBSCANResult parallel_dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int N = data.size();
    DBSCANResult result;
    result.cluster_labels.assign(N, -2);
    result.n_clusters = 0;

    long long dist_flops = 0;
    auto dist_matrix = parallel_distance_matrix(data, dist_flops);
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
        if ((int)neighbors.size() < min_pts) { result.cluster_labels[i] = -1; continue; }
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
    cout << "  Algorithm: RFF + One-Vs-Rest Linear SVM (Production technique)" << endl;

    auto train_data = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data = load_csv("data/test_data.csv");
    auto test_labels = load_labels("data/test_labels.csv");

    int N_train = train_data.size(), N_test = test_data.size(), D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    cout << "Train: " << N_train << " | Test: " << N_test
         << " | Features: " << D << " | Classes: " << n_classes << endl;

    Timer total_timer; total_timer.start();
    long long total_flops = 0;

    // ===== Step 1: RFF Transform (parallel) =====
    const int D_RFF = 1024;
    const double GAMMA = 0.1;
    cout << "\n--- RFF Transform (D_rff=" << D_RFF << ", gamma=" << GAMMA << ") ---" << endl;

    RffTransformer rff(D, D_RFF, GAMMA, /*seed=*/42);

    Timer t_rff; t_rff.start();
    auto rff_train = parallel_rff_transform(rff, train_data);
    auto rff_test  = parallel_rff_transform(rff, test_data);
    double rff_time = t_rff.elapsed_ms();
    total_flops += (long long)(N_train + N_test) * D_RFF * (D + 1);
    cout << "RFF transform: " << fixed << setprecision(1) << rff_time << " ms" << endl;

    rff.save("thread_rff_params.txt");

    // ===== Step 2: OvR Linear SVM Training (sequential, over RFF features) =====
    cout << "\n--- OvR Linear SVM Training (SGD + Cosine LR + Class Weights) ---" << endl;
    OvrLinearSVM svm(n_classes, D_RFF, /*lr=*/0.05, /*epochs=*/30, /*lambda=*/1e-4, /*conf_thresh=*/0.3);

    Timer t1; t1.start();
    long long train_flops = svm.train(rff_train, train_labels);
    double svm_train_time = t1.elapsed_ms();
    total_flops += train_flops;
    cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    // ===== Step 3: OvR SVM Prediction (parallel) =====
    cout << "\n--- OvR SVM Prediction (parallel) ---" << endl;
    Timer t2; t2.start();
    auto pred_result = parallel_predict_ovr(svm, rff_test);
    double svm_pred_time = t2.elapsed_ms();
    total_flops += pred_result.flops;
    cout << "SVM prediction: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;

    // ===== Step 4: Split confident vs uncertain =====
    double confidence_threshold = svm.conf_threshold;
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

    const int MAX_DBSCAN = 5000;
    if ((int)uncertain_data.size() > MAX_DBSCAN) {
        for (size_t i = MAX_DBSCAN; i < uncertain_indices.size(); i++)
            final_predictions[uncertain_indices[i]] = pred_result.predictions[uncertain_indices[i]];
        uncertain_indices.resize(MAX_DBSCAN);
        uncertain_data.resize(MAX_DBSCAN);
    }
    cout << "Confident: " << confident_count << " | Uncertain: " << uncertain_indices.size() << endl;

    // ===== Step 5: DBSCAN on uncertain subset (parallel distance matrix) =====
    int n_noise = 0;
    double dbscan_time = 0;
    if (uncertain_data.size() > 1) {
        cout << "\n--- DBSCAN (parallel distance matrix) ---" << endl;
        Timer t3; t3.start();
        double dbscan_eps = 2.0;
        int dbscan_min_pts = max(3, D / 10);
        auto dbscan_result = parallel_dbscan(uncertain_data, dbscan_eps, dbscan_min_pts);
        dbscan_time = t3.elapsed_ms();
        total_flops += dbscan_result.flops;
        n_noise = dbscan_result.n_noise;
        cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms" << endl;
        cout << "Clusters: " << dbscan_result.n_clusters << " | Noise: " << n_noise << endl;

        save_dbscan_model(dbscan_result, uncertain_data, dbscan_eps, dbscan_min_pts, "thread_dbscan_model.txt");

        for (size_t i = 0; i < uncertain_indices.size(); i++) {
            int idx = uncertain_indices[i];
            final_predictions[idx] = (dbscan_result.cluster_labels[i] == -1)
                                     ? -1 : pred_result.predictions[idx];
        }
    }

    double total_time = total_timer.elapsed_ms();

    // ===== Results =====
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

    print_metrics("C++ Thread (" + to_string(NUM_THREADS) + " threads, RFF+OvR)",
                  total_time / 1000.0, total_flops, N_test, D);
    cout << "Timing Breakdown:" << endl;
    cout << "  RFF Transform:" << fixed << setprecision(1) << rff_time   << " ms" << endl;
    cout << "  SVM Train:    " << svm_train_time << " ms" << endl;
    cout << "  SVM Predict:  " << svm_pred_time  << " ms" << endl;
    cout << "  DBSCAN:       " << dbscan_time     << " ms" << endl;
    cout << "  Total:        " << total_time      << " ms" << endl;

    // Save outputs
    svm.save_model("thread_ovr_svm_model.txt");
    save_epoch_losses(svm, "thread_epoch_losses.csv");
    save_predictions(final_predictions, "thread_predictions.csv");

    // Compute train accuracy on a sample of RFF-transformed train data
    int sample_n = min(N_train, 20000);
    int step = N_train / sample_n;
    vector<vector<double>> rff_train_sample;
    vector<int> train_labels_sample;
    for (int i = 0; i < N_train && (int)rff_train_sample.size() < sample_n; i += step) {
        rff_train_sample.push_back(rff_train[i]);
        train_labels_sample.push_back(train_labels[i]);
    }
    double train_acc = compute_train_accuracy_ovr(svm, rff_train_sample, train_labels_sample);
    cout << "Train Accuracy: " << fixed << setprecision(2) << train_acc << "%" << endl;
    { ofstream fa("thread_accuracy.csv"); fa << fixed << setprecision(4) << train_acc << "," << acc << "\n"; }

    return 0;
}
