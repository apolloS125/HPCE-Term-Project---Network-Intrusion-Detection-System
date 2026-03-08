// mpi_ids.cpp - MPI distributed SVM + DBSCAN pipeline
// Compile: mpicxx -O2 -o mpi_ids mpi_ids.cpp -lm
// Run:     mpirun -np 4 ./mpi_ids
#include "common.h"
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) cout << "=== Network IDS: MPI (" << size << " processes) ===" << endl;

    // All processes load data (simplified - in production use MPI I/O)
    auto train_data = load_csv("data/train_data.csv");
    auto train_labels = load_labels("data/train_labels.csv");
    auto test_data = load_csv("data/test_data.csv");
    auto test_labels = load_labels("data/test_labels.csv");

    int N_train = train_data.size(), N_test = test_data.size(), D = train_data[0].size();
    int n_classes = detect_n_classes(train_labels);
    if (rank == 0) cout << "Train: " << N_train << " | Test: " << N_test << " | Features: " << D << " | Classes: " << n_classes << endl;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();
    long long total_flops = 0;

    // ===== SVM Training (each process trains on full data - same model) =====
    double t1 = MPI_Wtime();
    MultiClassSVM svm(n_classes, 0.1, 50, 0.01);
    long long train_flops = svm.train(train_data, train_labels);
    double svm_train_time = (MPI_Wtime() - t1) * 1000;
    total_flops += train_flops;

    // ===== SVM Prediction (distributed across processes) =====
    int chunk = (N_test + size - 1) / size;
    int my_start = rank * chunk;
    int my_end = min(my_start + chunk, N_test);
    int my_count = max(0, my_end - my_start);

    double t2 = MPI_Wtime();
    vector<int> local_preds(my_count);
    vector<double> local_confs(my_count);

    for (int i = 0; i < my_count; i++) {
        auto [pred, conf] = svm.predict_one(test_data[my_start + i]);
        local_preds[i] = pred;
        local_confs[i] = conf;
    }

    // Gather predictions to root
    vector<int> all_preds(N_test);
    vector<double> all_confs(N_test);

    // Use MPI_Gatherv for variable-length chunks
    vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; i++) {
        int s = i * chunk;
        int e = min(s + chunk, N_test);
        recvcounts[i] = max(0, e - s);
        displs[i] = s;
    }
    MPI_Gatherv(local_preds.data(), my_count, MPI_INT,
                all_preds.data(), recvcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_confs.data(), my_count, MPI_DOUBLE,
                all_confs.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double svm_pred_time = (MPI_Wtime() - t2) * 1000;

    int total_sv = 0;
    for (auto& m : svm.models) total_sv += m.support_vectors.size();
    total_flops += (long long)N_test * total_sv * (3 * D + 2);

    // ===== DBSCAN on root (uncertain subset) =====
    double dbscan_time = 0;
    vector<int> final_predictions;

    if (rank == 0) {
        final_predictions.resize(N_test);
        double threshold = 0.3;
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
        cout << "Confident: " << (N_test - (int)uncertain_idx.size())
             << " | Uncertain: " << uncertain_idx.size() << endl;

        if (uncertain_data.size() > 1) {
            double t3 = MPI_Wtime();

            // Distribute DBSCAN distance computation across processes
            // For simplicity, run on root (in production, distribute)
            auto db = dbscan(uncertain_data, 1.5, D + 1);

            dbscan_time = (MPI_Wtime() - t3) * 1000;
            total_flops += db.flops;
            cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms"
                 << " | Clusters: " << db.n_clusters << " | Noise: " << db.n_noise << endl;

            for (size_t i = 0; i < uncertain_idx.size(); i++)
                final_predictions[uncertain_idx[i]] = (db.cluster_labels[i] == -1) ? -1 : all_preds[uncertain_idx[i]];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = (MPI_Wtime() - total_start) * 1000;

    if (rank == 0) {
        int correct = 0, classified = 0;
        for (int i = 0; i < N_test; i++) {
            if (final_predictions[i] >= 0) { classified++; if (final_predictions[i] == test_labels[i]) correct++; }
        }
        cout << "\nAccuracy: " << fixed << setprecision(2) << (100.0 * correct / max(1, classified)) << "%" << endl;
        print_metrics("MPI (" + to_string(size) + " processes)", total_time / 1000.0, total_flops, N_test, D);
        cout << "Timing: Train=" << fixed << setprecision(1) << svm_train_time << "ms Predict=" << svm_pred_time << "ms DBSCAN=" << dbscan_time << "ms Total=" << total_time << "ms" << endl;
    }

    MPI_Finalize();
    return 0;
}
