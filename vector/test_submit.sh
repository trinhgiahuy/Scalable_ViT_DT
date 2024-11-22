#!/bin/bash
#SBATCH --job-name=test_code     # Descriptive job name
#SBATCH --partition=a40          # Use the a40 partition for testing
#SBATCH --gres=gpu:2             # Request 2 GPUs
#SBATCH --qos=normal             # Use normal QOS for faster scheduling
#SBATCH --time=1:00:00           # Limit the test run to 1 hour
#SBATCH -c 16                    # Request 16 CPU cores
#SBATCH --mem=32G                # Request 32 GB of memory
#SBATCH --output=slurm-%j.out    # Save standard output
#SBATCH --error=slurm-%j.err     # Save standard error

# Load required modules
# THESE module does not work
# module load python/3.8.2
# module load py-torch/2.3.0-py3.10-CUDA12.1.1-7qgk
#module load modules-u22/1.0 
#module load module load py-torch/2.4.1-CUDA12.4.0-f3vy 

source /h/h3trinh/ece750/bin/activate 

module load pytorch2.1-cuda11.8-python3.9

# Run the test script
python3 test_torch_gpu.py

nvidia-smi

