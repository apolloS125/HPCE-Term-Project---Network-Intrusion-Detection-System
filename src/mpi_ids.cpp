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

    // Rank 0 loads data, then broadcasts dimensions and flat arrays
    vector<vector<double>> train_data, test_data;
    vector<int> train_labels, test_labels;
    int N_train = 0, N_test = 0, D = 0, n_classes = 0;

    // Subsample limit for training data to fit in memory
    const int MAX_TRAIN = 50000;
    const int MAX_TEST = 20000;

    if (rank == 0) {
        train_data = load_csv("data/train_data.csv");
        train_labels = load_labels("data/train_labels.csv");
        test_data = load_csv("data/test_data.csv");
        test_labels = load_labels("data/test_labels.csv");

        D = train_data[0].size();
        n_classes = detect_n_classes(train_labels);

        // Subsample training data if too large
        if ((int)train_data.size() > MAX_TRAIN) {
            int step = train_data.size() / MAX_TRAIN;
            vector<vector<double>> sampled_data;
            vector<int> sampled_labels;
            for (int i = 0; i < MAX_TRAIN; i++) {
                sampled_data.push_back(train_data[i * step]);
                sampled_labels.push_back(train_labels[i * step]);
            }
            train_data = sampled_data;
            train_labels = sampled_labels;
        }
        // Subsample test data if too large
        if ((int)test_data.size() > MAX_TEST) {
            int step = test_data.size() / MAX_TEST;
            vector<vector<double>> sampled_data;
            vector<int> sampled_labels;
            for (int i = 0; i < MAX_TEST; i++) {
                sampled_data.push_back(test_data[i * step]);
                sampled_labels.push_back(test_labels[i * step]);
            }
            test_data = sampled_data;
            test_labels = sampled_labels;
        }

        N_train = train_data.size();
        N_test = test_data.size();
        cout << "Train: " << N_train << " | Test: " << N_test << " | Features: " << D << " | Classes: " << n_classes << endl;
    }

    // Broadcast dimensions
    MPI_Bcast(&N_train, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_test, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Flatten and broadcast train data
    vector<double> flat_train(N_train * D);
    if (rank == 0)
        for (int i = 0; i < N_train; i++)
            for (int j = 0; j < D; j++)
                flat_train[i * D + j] = train_data[i][j];
    MPI_Bcast(flat_train.data(), N_train * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast train labels
    if (rank != 0) train_labels.resize(N_train);
    MPI_Bcast(train_labels.data(), N_train, MPI_INT, 0, MPI_COMM_WORLD);

    // Flatten and broadcast test data
    vector<double> flat_test(N_test * D);
    if (rank == 0)
        for (int i = 0; i < N_test; i++)
            for (int j = 0; j < D; j++)
                flat_test[i * D + j] = test_data[i][j];
    MPI_Bcast(flat_test.data(), N_test * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast test labels
    if (rank != 0) test_labels.resize(N_test);
    MPI_Bcast(test_labels.data(), N_test, MPI_INT, 0, MPI_COMM_WORLD);

    // Reconstruct 2D vectors on non-root ranks
    if (rank != 0) {
        train_data.resize(N_train, vector<double>(D));
        for (int i = 0; i < N_train; i++)
            for (int j = 0; j < D; j++)
                train_data[i][j] = flat_train[i * D + j];
        test_data.resize(N_test, vector<double>(D));
        for (int i = 0; i < N_test; i++)
            for (int j = 0; j < D; j++)
                test_data[i][j] = flat_test[i * D + j];
    }
    // Free flat arrays
    flat_train.clear(); flat_train.shrink_to_fit();
    flat_test.clear(); flat_test.shrink_to_fit();

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
        double threshold = 0.05;
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

        // Cap DBSCAN input to avoid O(n^2) blowup
        const int MAX_DBSCAN = 2000;
        if ((int)uncertain_data.size() > MAX_DBSCAN) {
            uncertain_idx.resize(MAX_DBSCAN);
            uncertain_data.resize(MAX_DBSCAN);
            // Mark the rest as SVM prediction (fallback)
            for (int i = MAX_DBSCAN; i < N_test; i++)
                if (final_predictions[i] == -1) final_predictions[i] = all_preds[i];
        }
        cout << "Confident: " << (N_test - (int)uncertain_idx.size())
             << " | Uncertain: " << uncertain_idx.size() << endl;

        if (uncertain_data.size() > 1) {
            double t3 = MPI_Wtime();

            // Distribute DBSCAN distance computation across processes
            // For simplicity, run on root (in production, distribute)
            double dbscan_eps = 1.5;
            int dbscan_min_pts = D + 1;
            auto db = dbscan(uncertain_data, dbscan_eps, dbscan_min_pts);

            dbscan_time = (MPI_Wtime() - t3) * 1000;
            total_flops += db.flops;
            cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms"
                 << " | Clusters: " << db.n_clusters << " | Noise: " << db.n_noise << endl;

            // Save DBSCAN model
            save_dbscan_model(db, uncertain_data, dbscan_eps, dbscan_min_pts, "mpi_dbscan_model.txt");

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

        // Save model and predictions
        save_svm_model(svm, "mpi_svm_model.txt");
        save_predictions(final_predictions, "mpi_predictions.csv");
    }

    MPI_Finalize();
    return 0;
}
