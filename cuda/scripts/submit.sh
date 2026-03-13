#!/bin/bash
#SBATCH --job-name=cuda_svm
#SBATCH --partition=nvidia-rtx2080
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd "$SLURM_SUBMIT_DIR"

# Compile
nvcc -O3 -arch=sm_75 -std=c++17 -lcublas -o cuda_svm cuda_svm.cu

# Run
./cuda_svm
