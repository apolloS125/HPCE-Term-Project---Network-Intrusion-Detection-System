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
