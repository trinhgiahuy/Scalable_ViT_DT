#!/bin/bash
#SBATCH --job-name=deepspeed_test
#SBATCH --partition=a40
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=inter_mode_%j.out
#SBATCH --error=inter_node_%j.err

# do not use interactive terminal
# Load environment and activate conda
source /h/h3trinh/.bashrc
conda activate /h/h3trinh/condaenvs/vit_env

# Set NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0
# # export NCCL_IB_DISABLE=0
# # export NCCL_LAUNCH_MODE=PARALLEL
# export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=29500
export OMP_NUM_THREADS=4

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

echo "nodes_array[@]"
echo ${nodes_array[@]}

# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# head_node_ip=$(hostname --ip-address)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


echo Node IP: $head_node_ip
export LOGLEVEL=INFO


# Run DeepSpeed with torchrun
# srun torchrun \
# torchrun \
# --nnodes=4 \
# --nproc_per_node 1 \
# --rdzv_id $RANDOM \
# --rdzv_backend c10d \
# --rdzv_endpoint $head_node_ip:29500 \
# work_dist_train_inter_node.py \
# --deepspeed \
# --deepspeed_config "deepspeed_config.json"

# srun torchrun
srun torchrun \
		--nproc_per_node 1 \
		--nnodes 2 \
		--rdzv_id $RANDOM \
		--rdzv_backend c10d \
		--rdzv_endpoint $head_node_ip:29500 \
		train_inter_node.py \
		--deepspeed \
		--deepspeed_config "deepspeed_config.json"

# --deepspeed \
# --deepspeed_config "deepspeed_config.json"

# srun torchrun \
# --nnodes 2 \
# --nproc_per_node 1 \
# --rdzv_id $RANDOM \
# --rdzv_backend c10d \
# --rdzv_endpoint $head_node_ip:29500 \
# multinode.py