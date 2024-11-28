#!/bin/bash

#SBATCH --job-name=deepspeed_test
#SBATCH --partition=t4v2
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH --output=distributed_%j.out
#SBATCH --error=distributed_%j.err

# Activate environment
source /h/h3trinh/.bashrc
conda activate /h/h3trinh/condaenvs/vit_env

# Retrieve node information
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node IP: $head_node_ip"

# Set environment variables for rendezvous
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
export OMP_NUM_THREADS=2
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

# Have to use srun with torchrun
srun torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    test_train.py \
    --deepspeed \
    --deepspeed_config "deepspeed_config.json"
