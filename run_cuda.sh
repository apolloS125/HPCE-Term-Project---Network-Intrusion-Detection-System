#!/bin/bash
#SBATCH --job-name=cuda_ids
#SBATCH --output=cuda_ids_%j.out
#SBATCH --error=cuda_ids_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nvidia-rtx2080
#SBATCH --time=01:00:00

echo "=== CUDA IDS Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Date: $(date)"
echo ""

# Build
make cuda

# Run
./bin/cuda_ids
