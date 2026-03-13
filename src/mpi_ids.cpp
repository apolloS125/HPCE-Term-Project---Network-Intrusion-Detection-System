// mpi_ids.cpp - MPI distributed SVM + DBSCAN pipeline
// Upgraded: RFF transformer + One-Vs-Rest Linear SVM (from cuda/src techniques)
// Compile: mpicxx -O2 -std=c++17 -o mpi_ids mpi_ids.cpp -lm
// Run:     mpirun -np 4 ./mpi_ids
#include "common.h"
#include <mpi.h>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "=== Network IDS: MPI (" << size << " processes) ===" << endl;
        cout << "  Algorithm: RFF + One-Vs-Rest Linear SVM (Production technique)" << endl;
    }

    // ===== Load and distribute data (Rank 0) =====
    vector<vector<double>> train_data, test_data;
    vector<int> train_labels, test_labels;
    int N_train = 0, N_test = 0, D = 0, n_classes = 0;

    const int MAX_TRAIN = 200000;
    const int MAX_TEST  = 100000;

    if (rank == 0) {
        train_data   = load_csv("data/train_data.csv");
        train_labels = load_labels("data/train_labels.csv");
        test_data    = load_csv("data/test_data.csv");
        test_labels  = load_labels("data/test_labels.csv");

        D = train_data[0].size();
        n_classes = detect_n_classes(train_labels);

        auto subsample = [](auto& data, auto& labels, int max_n) {
            if ((int)data.size() > max_n) {
                int step = data.size() / max_n;
                decltype(data) sd; decltype(labels) sl;
                for (int i = 0; i < max_n; i++) { sd.push_back(data[i * step]); sl.push_back(labels[i * step]); }
                data = sd; labels = sl;
            }
        };
        subsample(train_data, train_labels, MAX_TRAIN);
        subsample(test_data, test_labels, MAX_TEST);

        N_train = train_data.size();
        N_test  = test_data.size();
        cout << "Train: " << N_train << " | Test: " << N_test
             << " | Features: " << D << " | Classes: " << n_classes << endl;
    }

    MPI_Bcast(&N_train,   1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N_test,    1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D,         1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ===== Broadcast RFF parameters (generated on Rank 0) =====
    const int D_RFF = 1024;
    const double GAMMA = 0.1;

    // Broadcast Omega (D_RFF * D) and phi (D_RFF)
    vector<double> omega_flat(D_RFF * D), phi_vec(D_RFF);
    if (rank == 0) {
        RffTransformer rff_gen(D, D_RFF, GAMMA, /*seed=*/42);
        omega_flat = rff_gen.Omega;
        phi_vec    = rff_gen.phi;
    }
    MPI_Bcast(omega_flat.data(), D_RFF * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(phi_vec.data(),    D_RFF,     MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct RFF on all ranks
    RffTransformer rff;
    rff.D_in  = D;
    rff.D_rff = D_RFF;
    rff.gamma = GAMMA;
    rff.Omega = omega_flat;
    rff.phi   = phi_vec;

    // ===== Broadcast flat train & test arrays =====
    vector<double> flat_train(N_train * D), flat_test(N_test * D);
    if (rank == 0) {
        for (int i = 0; i < N_train; i++) for (int j = 0; j < D; j++) flat_train[i*D+j] = train_data[i][j];
        for (int i = 0; i < N_test;  i++) for (int j = 0; j < D; j++) flat_test[i*D+j]  = test_data[i][j];
    }
    MPI_Bcast(flat_train.data(), N_train * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_test.data(),  N_test  * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        train_labels.resize(N_train); test_labels.resize(N_test);
        train_data.resize(N_train, vector<double>(D));
        test_data.resize(N_test,   vector<double>(D));
        for (int i = 0; i < N_train; i++) for (int j = 0; j < D; j++) train_data[i][j] = flat_train[i*D+j];
        for (int i = 0; i < N_test;  i++) for (int j = 0; j < D; j++) test_data[i][j]  = flat_test[i*D+j];
    }
    MPI_Bcast(train_labels.data(), N_train, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_labels.data(),  N_test,  MPI_INT, 0, MPI_COMM_WORLD);
    flat_train.clear(); flat_test.clear();

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();
    long long total_flops = 0;

    // ===== Step 1: RFF Transform (each process transforms its own slice) =====
    // Each rank transforms FULL train (for training) — same model on all ranks
    // and a CHUNK of test data for distributed prediction
    double t_rff_start = MPI_Wtime();
    auto rff_train = rff.transform_batch(train_data); // all ranks transform full train
    double rff_train_time = (MPI_Wtime() - t_rff_start) * 1000;
    total_flops += (long long)N_train * D_RFF * (D + 1);

    // ===== Step 2: OvR Linear SVM Training (same on every rank) =====
    double t1 = MPI_Wtime();
    OvrLinearSVM svm(n_classes, D_RFF, /*lr=*/0.05, /*epochs=*/30, /*lambda=*/1e-4, /*conf_thresh=*/0.3);
    long long train_flops = svm.train(rff_train, train_labels);
    double svm_train_time = (MPI_Wtime() - t1) * 1000;
    total_flops += train_flops;
    if (rank == 0)
        cout << "SVM training: " << fixed << setprecision(1) << svm_train_time << " ms" << endl;

    // ===== Step 3: Distributed RFF Transform + OvR Predict on test data =====
    int chunk = (N_test + size - 1) / size;
    int my_start = rank * chunk;
    int my_end   = min(my_start + chunk, N_test);
    int my_count = max(0, my_end - my_start);

    double t2 = MPI_Wtime();
    vector<int>    local_preds(my_count);
    vector<double> local_confs(my_count);
    for (int i = 0; i < my_count; i++) {
        auto z = rff.transform(test_data[my_start + i]);
        auto [pred, conf] = svm.predict_one(z);
        local_preds[i] = pred;
        local_confs[i] = conf;
    }
    total_flops += (long long)my_count * n_classes * (D_RFF + 1);

    // Gather predictions to Rank 0
    vector<int>    all_preds(N_test);
    vector<double> all_confs(N_test);
    vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; i++) {
        int s = i * chunk, e = min(s + chunk, N_test);
        recvcounts[i] = max(0, e - s);
        displs[i] = s;
    }
    MPI_Gatherv(local_preds.data(), my_count, MPI_INT,    all_preds.data(), recvcounts.data(), displs.data(), MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Gatherv(local_confs.data(), my_count, MPI_DOUBLE, all_confs.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double svm_pred_time = (MPI_Wtime() - t2) * 1000;

    // ===== Step 4: DBSCAN on Rank 0 (uncertain subset) =====
    double dbscan_time = 0;
    vector<int> final_predictions;

    if (rank == 0) {
        final_predictions.resize(N_test);
        double threshold = svm.conf_threshold;
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

        const int MAX_DBSCAN = 5000;
        if ((int)uncertain_data.size() > MAX_DBSCAN) {
            for (int i = MAX_DBSCAN; i < N_test; i++)
                if (final_predictions[i] == -1) final_predictions[i] = all_preds[i];
            uncertain_idx.resize(MAX_DBSCAN);
            uncertain_data.resize(MAX_DBSCAN);
        }
        cout << "Confident: " << (N_test - (int)uncertain_idx.size())
             << " | Uncertain: " << uncertain_idx.size() << endl;

        if (uncertain_data.size() > 1) {
            double t3 = MPI_Wtime();
            double dbscan_eps = 2.0;
            int dbscan_min_pts = max(3, D / 10);
            auto db = dbscan(uncertain_data, dbscan_eps, dbscan_min_pts);
            dbscan_time = (MPI_Wtime() - t3) * 1000;
            total_flops += db.flops;
            cout << "DBSCAN: " << fixed << setprecision(1) << dbscan_time << " ms"
                 << " | Clusters: " << db.n_clusters << " | Noise: " << db.n_noise << endl;

            save_dbscan_model(db, uncertain_data, dbscan_eps, dbscan_min_pts, "mpi_dbscan_model.txt");

            for (size_t i = 0; i < uncertain_idx.size(); i++)
                final_predictions[uncertain_idx[i]] = (db.cluster_labels[i] == -1)
                                                      ? -1 : all_preds[uncertain_idx[i]];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double total_time = (MPI_Wtime() - total_start) * 1000;

    if (rank == 0) {
        int correct = 0, classified = 0;
        for (int i = 0; i < N_test; i++)
            if (final_predictions[i] >= 0) { classified++; if (final_predictions[i] == test_labels[i]) correct++; }

        double acc = 100.0 * correct / max(1, classified);
        cout << "\nAccuracy: " << fixed << setprecision(2) << acc << "%" << endl;
        print_metrics("MPI (" + to_string(size) + " processes, RFF+OvR)",
                      total_time / 1000.0, total_flops, N_test, D);
        cout << "Timing: RFF_train=" << fixed << setprecision(1) << rff_train_time << "ms"
             << " Train=" << svm_train_time << "ms"
             << " Predict=" << svm_pred_time << "ms"
             << " DBSCAN=" << dbscan_time << "ms"
             << " Total=" << total_time << "ms" << endl;

        svm.save_model("mpi_ovr_svm_model.txt");
        rff.save("mpi_rff_params.txt");
        save_epoch_losses(svm, "mpi_epoch_losses.csv");
        save_predictions(final_predictions, "mpi_predictions.csv");

        // Train accuracy on sample
        int sample_n = min(N_train, 20000);
        int step = max(1, N_train / sample_n);
        vector<vector<double>> rff_sample; vector<int> label_sample;
        for (int i = 0; i < N_train && (int)rff_sample.size() < sample_n; i += step) {
            rff_sample.push_back(rff_train[i]); label_sample.push_back(train_labels[i]);
        }
        double train_acc = compute_train_accuracy_ovr(svm, rff_sample, label_sample);
        cout << "Train Accuracy: " << fixed << setprecision(2) << train_acc << "%" << endl;
        { ofstream fa("mpi_accuracy.csv"); fa << fixed << setprecision(4) << train_acc << "," << acc << "\n"; }
    }

    MPI_Finalize();
    return 0;
}
