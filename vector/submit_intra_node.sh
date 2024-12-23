#!/bin/bash
#SBATCH --job-name=deepspeed_test
#SBATCH --partition=t4v2     	# Replace with your partition
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --ntasks-per-node=7     # Number of tasks (ranks) per node
#SBATCH --gres=gpu:7             # Request 2 GPUs
##SBATCH --qos=normal            # Use normal QOS for faster scheduling
#SBATCH --cpus-per-task=4        # Number of CPUs per task
#SBATCH --mem=128G                # Memory allocation
#SBATCH --time=5:00:00           # Max runtime
#SBATCH --output=distributed_%j.out
#SBATCH --error=distributed_%j.err

# Activate environment
source /h/h3trinh/.bashrc
conda activate /h/h3trinh/condaenvs/vit_env

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export OMP_NUM_THREADS=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Run DeepSpeed with torchrun
torchrun --nproc_per_node=7 \
		 --nnodes=1 \
		 --rdzv_id=${SLURM_JOB_ID} \
		 --rdzv_backend=c10d \
		 --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
		 train_intra_node.py \
		 --deepspeed \
		 --deepspeed_config "deepspeed_config.json"
