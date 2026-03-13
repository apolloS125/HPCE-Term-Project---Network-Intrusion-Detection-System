// openmp.cpp - OpenMP parallel SVM + DBSCAN pipeline
// Upgraded: RFF transformer + One-Vs-Rest Linear SVM (from cuda/src techniques)
// Compile: g++ -O2 -fopenmp -std=c++17 -o openmp openmp.cpp -lm
#include "common.h"
#include <omp.h>

int NUM_THREADS = 8;

// ===== OpenMP parallel RFF transform =====
vector<vector<double>> omp_rff_transform(const RffTransformer& rff,
                                          const vector<vector<double>>& data) {
    int N = data.size();
    vector<vector<double>> out(N);
    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++)
        out[i] = rff.transform(data[i]);
    return out;
}

// ===== OpenMP parallel OvR predict =====
OvrLinearSVM::PredResult omp_predict_ovr(const OvrLinearSVM& svm,
                                          const vector<vector<double>>& rff_data) {
    int N = rff_data.size();
    OvrLinearSVM::PredResult result;
    result.predictions.resize(N);
    result.confidences.resize(N);
    result.flops = (long long)N * svm.n_classes * (svm.D_rff + 1);
    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (int i = 0; i < N; i++) {
        auto [p, c] = svm.predict_one(rff_data[i]);
        result.predictions[i] = p;
        result.confidences[i] = c;
    }
    return result;
}

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

DBSCANResult omp_dbscan(const vector<vector<double>>& data, double eps, int min_pts) {
    int N = data.size();
    DBSCANResult result;
    result.cluster_labels.assign(N, -2);
    result.n_clusters = 0;

    long long dist_flops = 0;
    auto dist_matrix = omp_distance_matrix(data, dist_flops);
    result.flops = dist_flops;

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
    cout << "  Algorithm: RFF + One-Vs-Rest Linear SVM (Production technique)" << endl;

    auto train_data   = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data    = load_csv("data/test_data.csv");
    auto test_labels  = load_labels("data/test_labels.csv");

    int N_train = train_data.size(), N_test = test_data.size(), D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    cout << "Train: " << N_train << " | Test: " << N_test
         << " | Features: " << D << " | Classes: " << n_classes << endl;

    Timer total_timer; total_timer.start();
    long long total_flops = 0;

    // ===== Step 1: RFF Transform (OpenMP parallel) =====
    const int D_RFF = 1024;
    const double GAMMA = 0.1;
    cout << "\n--- RFF Transform (D_rff=" << D_RFF << ", gamma=" << GAMMA << ") ---" << endl;

    RffTransformer rff(D, D_RFF, GAMMA, /*seed=*/42);

    Timer t_rff; t_rff.start();
    auto rff_train = omp_rff_transform(rff, train_data);
    auto rff_test  = omp_rff_transform(rff, test_data);
    double rff_time = t_rff.elapsed_ms();
    total_flops += (long long)(N_train + N_test) * D_RFF * (D + 1);
    cout << "RFF transform: " << fixed << setprecision(1) << rff_time << " ms" << endl;

    rff.save("openmp_rff_params.txt");

    // ===== Step 2: OvR Linear SVM Training =====
    cout << "\n--- OvR Linear SVM Training (SGD + Cosine LR + Class Weights) ---" << endl;
    OvrLinearSVM svm(n_classes, D_RFF, /*lr=*/0.05, /*epochs=*/30, /*lambda=*/1e-4, /*conf_thresh=*/0.3);

    Timer t1; t1.start();
    long long train_flops = svm.train(rff_train, train_labels);
    double svm_train_time = t1.elapsed_ms();
    total_flops += train_flops;
    cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    // ===== Step 3: OvR SVM Prediction (OpenMP parallel) =====
    cout << "\n--- OvR SVM Prediction (OpenMP parallel) ---" << endl;
    Timer t2; t2.start();
    auto pred_result = omp_predict_ovr(svm, rff_test);
    double svm_pred_time = t2.elapsed_ms();
    total_flops += pred_result.flops;
    cout << "SVM prediction: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;

    // ===== Step 4: Split confident vs uncertain =====
    double threshold = svm.conf_threshold;
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

    const int MAX_DBSCAN = 5000;
    if ((int)uncertain_data.size() > MAX_DBSCAN) {
        for (int i = MAX_DBSCAN; i < N_test; i++)
            if (final_predictions[i] == -1) final_predictions[i] = pred_result.predictions[i];
        uncertain_idx.resize(MAX_DBSCAN);
        uncertain_data.resize(MAX_DBSCAN);
    }
    cout << "Confident: " << (N_test - (int)uncertain_idx.size())
         << " | Uncertain: " << uncertain_idx.size() << endl;

    // ===== Step 5: DBSCAN (OpenMP distance matrix) =====
    double dbscan_time = 0;
    if (uncertain_data.size() > 1) {
        cout << "\n--- DBSCAN (OpenMP parallel distance matrix) ---" << endl;
        Timer t3; t3.start();
        double dbscan_eps = 2.0;
        int dbscan_min_pts = max(3, D / 10);
        auto db = omp_dbscan(uncertain_data, dbscan_eps, dbscan_min_pts);
        dbscan_time = t3.elapsed_ms();
        total_flops += db.flops;
        cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms"
             << " | Clusters: " << db.n_clusters << " | Noise: " << db.n_noise << endl;

        save_dbscan_model(db, uncertain_data, dbscan_eps, dbscan_min_pts, "openmp_dbscan_model.txt");

        for (size_t i = 0; i < uncertain_idx.size(); i++)
            final_predictions[uncertain_idx[i]] = (db.cluster_labels[i] == -1)
                                                  ? -1 : pred_result.predictions[uncertain_idx[i]];
    }

    double total_time = total_timer.elapsed_ms();

    // ===== Results =====
    int correct = 0, classified = 0;
    for (int i = 0; i < N_test; i++) {
        if (final_predictions[i] >= 0) { classified++; if (final_predictions[i] == test_labels[i]) correct++; }
    }
    double acc = classified > 0 ? 100.0 * correct / classified : 0;
    cout << "\nAccuracy: " << fixed << setprecision(2) << acc << "%" << endl;

    print_metrics("OpenMP (" + to_string(NUM_THREADS) + " threads, RFF+OvR)",
                  total_time / 1000.0, total_flops, N_test, D);
    cout << "Timing: RFF=" << fixed << setprecision(1) << rff_time << "ms"
         << " Train=" << svm_train_time << "ms"
         << " Predict=" << svm_pred_time << "ms"
         << " DBSCAN=" << dbscan_time << "ms"
         << " Total=" << total_time << "ms" << endl;

    svm.save_model("openmp_ovr_svm_model.txt");
    save_epoch_losses(svm, "openmp_epoch_losses.csv");
    save_predictions(final_predictions, "openmp_predictions.csv");

    int sample_n = min(N_train, 20000);
    int step = max(1, N_train / sample_n);
    vector<vector<double>> rff_train_sample;
    vector<int> train_labels_sample;
    for (int i = 0; i < N_train && (int)rff_train_sample.size() < sample_n; i += step) {
        rff_train_sample.push_back(rff_train[i]);
        train_labels_sample.push_back(train_labels[i]);
    }
    double train_acc = compute_train_accuracy_ovr(svm, rff_train_sample, train_labels_sample);
    cout << "Train Accuracy: " << fixed << setprecision(2) << train_acc << "%" << endl;
    { ofstream fa("openmp_accuracy.csv"); fa << fixed << setprecision(4) << train_acc << "," << acc << "\n"; }

    return 0;
}
