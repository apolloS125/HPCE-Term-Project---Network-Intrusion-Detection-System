#!/bin/bash
#SBATCH --job-name=mpi_ids
#SBATCH --output=mpi_ids_%j.out
#SBATCH --error=mpi_ids_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:10:00
#SBATCH --partition=nvidia-rtx2080

module load mpi 2>/dev/null || true

cd $SLURM_SUBMIT_DIR

make mpi
mpirun -np 2 ./bin/mpi_ids
