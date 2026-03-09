// sequential.cpp - Sequential baseline for SVM + DBSCAN IDS pipeline
// Compile: g++ -O2 -o sequential sequential.cpp -lm
#include "common.h"

int main(int argc, char* argv[]) {
    cout << "=== Network IDS: Sequential Baseline ===" << endl;

    // Load data
    auto train_data = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data = load_csv("data/test_data.csv");
    auto test_labels = load_labels("data/test_labels.csv");

    int N_train = train_data.size();
    int N_test = test_data.size();
    int D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    cout << "Train: " << N_train << " | Test: " << N_test << " | Features: " << D << " | Classes: " << n_classes << endl;

    Timer total_timer; total_timer.start();
    long long total_flops = 0;

    // ===== STAGE 2: SVM Classification =====
    cout << "\n--- Stage 2: SVM Training ---" << endl;
    MultiClassSVM svm(n_classes, 0.1, 200, 0.01);
    Timer svm_train_timer; svm_train_timer.start();
    long long train_flops = svm.train(train_data, train_labels);
    double svm_train_time = svm_train_timer.elapsed_ms();
    total_flops += train_flops;
    cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    cout << "\n--- Stage 2: SVM Prediction ---" << endl;
    Timer svm_pred_timer; svm_pred_timer.start();
    auto pred_result = svm.predict(test_data);
    double svm_pred_time = svm_pred_timer.elapsed_ms();
    total_flops += pred_result.flops;
    cout << "SVM prediction: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;

    // Split confident vs uncertain
    double confidence_threshold = 0.05;
    vector<int> final_predictions(N_test);
    vector<int> uncertain_indices;
    vector<vector<double>> uncertain_data;
    int confident_count = 0;

    for (int i = 0; i < N_test; i++) {
        if (pred_result.confidences[i] >= confidence_threshold) {
            final_predictions[i] = pred_result.predictions[i];
            confident_count++;
        } else {
            final_predictions[i] = -1;  // will be filled by DBSCAN
            uncertain_indices.push_back(i);
            uncertain_data.push_back(test_data[i]);
        }
    }
    cout << "Confident: " << confident_count << " (" 
         << fixed << setprecision(1) << (100.0 * confident_count / N_test) << "%)" << endl;
    cout << "Uncertain: " << uncertain_indices.size() << " -> sending to DBSCAN" << endl;

    // Cap DBSCAN input to avoid O(n^2) blowup
    const int MAX_DBSCAN = 2000;
    if ((int)uncertain_data.size() > MAX_DBSCAN) {
        for (size_t i = MAX_DBSCAN; i < uncertain_indices.size(); i++)
            final_predictions[uncertain_indices[i]] = pred_result.predictions[uncertain_indices[i]];
        uncertain_indices.resize(MAX_DBSCAN);
        uncertain_data.resize(MAX_DBSCAN);
    }

    // ===== STAGE 3: DBSCAN on uncertain data =====
    int n_noise = 0;
    double dbscan_time = 0;
    long long dbscan_flops = 0;

    if (!uncertain_data.empty() && uncertain_data.size() > 1) {
        cout << "\n--- Stage 3: DBSCAN Anomaly Detection ---" << endl;
        Timer dbscan_timer; dbscan_timer.start();

        double eps = 1.5;
        int min_pts = D + 1;
        auto dbscan_result = dbscan(uncertain_data, eps, min_pts);

        dbscan_time = dbscan_timer.elapsed_ms();
        dbscan_flops = dbscan_result.flops;
        total_flops += dbscan_flops;
        n_noise = dbscan_result.n_noise;

        cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms" << endl;
        cout << "Clusters found: " << dbscan_result.n_clusters << endl;
        cout << "Noise points (unknown attacks): " << n_noise << endl;

        // Assign DBSCAN results: noise = label -1 (unknown), clustered = nearest SVM prediction
        for (size_t i = 0; i < uncertain_indices.size(); i++) {
            int idx = uncertain_indices[i];
            if (dbscan_result.cluster_labels[i] == -1) {
                final_predictions[idx] = -1;  // unknown attack
            } else {
                final_predictions[idx] = pred_result.predictions[idx]; // use SVM's best guess
            }
        }
    }

    double total_time = total_timer.elapsed_ms();

    // ===== STAGE 4: Results =====
    cout << "\n--- Stage 4: Final Results ---" << endl;

    // Calculate accuracy (excluding unknowns)
    int correct = 0, total_classified = 0;
    for (int i = 0; i < N_test; i++) {
        if (final_predictions[i] >= 0) {
            total_classified++;
            if (final_predictions[i] == test_labels[i]) correct++;
        }
    }
    double acc = (total_classified > 0) ? (100.0 * correct / total_classified) : 0;
    cout << "Classified: " << total_classified << "/" << N_test << endl;
    cout << "Unknown attacks: " << (N_test - total_classified) << endl;
    cout << "Accuracy (classified only): " << fixed << setprecision(2) << acc << "%" << endl;

    // Print confusion matrix for classified samples
    vector<int> cm_pred, cm_truth;
    for (int i = 0; i < N_test; i++) {
        if (final_predictions[i] >= 0 && final_predictions[i] < n_classes) {
            cm_pred.push_back(final_predictions[i]);
            cm_truth.push_back(test_labels[i]);
        }
    }
    if (!cm_pred.empty()) print_confusion_matrix(cm_pred, cm_truth, n_classes);

    // Performance summary
    print_metrics("Sequential (Baseline)", total_time / 1000.0, total_flops, N_test, D);

    cout << "Timing Breakdown:" << endl;
    cout << "  SVM Train:   " << fixed << setprecision(1) << svm_train_time << " ms" << endl;
    cout << "  SVM Predict: " << fixed << setprecision(1) << svm_pred_time << " ms" << endl;
    cout << "  DBSCAN:      " << fixed << setprecision(1) << dbscan_time << " ms" << endl;
    cout << "  Total:       " << fixed << setprecision(1) << total_time << " ms" << endl;

    save_svm_model(svm, "seq_svm_model.txt");
    save_predictions(final_predictions, "seq_predictions.csv");

    return 0;
}
