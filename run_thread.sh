#!/bin/bash
#SBATCH --job-name=thread_ids
#SBATCH --output=thread_ids_%j.out
#SBATCH --error=thread_ids_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=nvidia-rtx2080

cd $SLURM_SUBMIT_DIR

make thread
./bin/cpp_thread 8
