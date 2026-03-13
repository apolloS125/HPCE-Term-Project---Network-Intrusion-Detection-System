#!/bin/bash
#SBATCH --job-name=cuda_hybrid
#SBATCH --partition=nvidia-rtx2080
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd "$SLURM_SUBMIT_DIR"

# Compile RFF version
nvcc -O3 -arch=sm_75 -std=c++17 -lcublas -o cuda_hybrid cuda_hybrid.cu

# Run
./cuda_hybrid
