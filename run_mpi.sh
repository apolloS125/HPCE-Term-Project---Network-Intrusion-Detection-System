#!/bin/bash
#SBATCH --job-name=mpi_ids
#SBATCH --output=mpi_ids_%j.out
#SBATCH --error=mpi_ids_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --partition=nvidia-rtx2080

module load mpi 2>/dev/null || true

cd $SLURM_SUBMIT_DIR

make mpi
mpirun -np 4 ./bin/mpi_ids


          ┌─────────────────────────────────────────────────┐
          │  Training set  (holdout REMOVED)                │
          │  DDoS + DoS + Normal + PortScan only            │
          └──────────────┬──────────────────────────────────┘
                         │ mpi_train.cpp
                         ▼
                    [ model.bin ]

          ┌─────────────────────────────────────────────────┐
          │  Test set  (FULL — including holdout)           │
          └──────────────┬──────────────────────────────────┘
                         │ mpi_infer.cpp: SVM predicts all
                         ▼
          ┌──────────────────────────────────────────────────┐
          │  Partition by TRUE label + confidence            │
          ├─────────────────┬────────────────────────────────┤
          │ is_holdout=true │ → DBSCAN pool (anchors)        │
          ├─────────────────┼────────────────────────────────┤
          │ conf >= 0.5     │ → SVM final prediction         │
          ├─────────────────┼────────────────────────────────┤
          │ conf < 0.5      │                                │
          │  pred=Normal   │ → Normal directly (skip DBSCAN)│
          │  pred=attack   │ → DBSCAN (uncertain attack pool)│
          └─────────────────┴────────────────────────────────┘
                         │
                         ▼
                  [ hybrid_predictions.csv ]