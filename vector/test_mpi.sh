#!/bin/bash
#SBATCH --job-name=test_mpi       # Descriptive job name
#SBATCH --partition=a40           # Partition for A40 GPUs
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH --qos=normal              # Use normal QOS for faster scheduling
#SBATCH --time=1:00:00            # Limit to 1 hour
#SBATCH --cpus-per-task=4         # Request 4 CPUs per task
#SBATCH --mem=32G                 # Request 32 GB of memory
#SBATCH --ntasks=2                # Number of MPI tasks
#SBATCH --output=slurm-%j.out     # Save standard output
#SBATCH --error=slurm-%j.err      # Save standard error

# Load MPI module
module purge
module load mpich-3.3.2

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate /h/h3trinh/condaenvs/vit_env

# Run the MPI job with srun
srun --mpi=pmi2 python test_mpi.py

