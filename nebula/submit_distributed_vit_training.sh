#!/bin/bash
#SBATCH --job-name=final_project
#SBATCH --partition=dualcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=12:00:00
#SBATCH --output=project_%j.out

#If you are using your own custom venv, replace mine with yours. Otherwise, stick to this default. It has torch, transformers, accelerate and a bunch of others. I'm happy to add more common libraries
# source /local/transformers/bin/activate
source /mnt/slurm_nfs/$USER/vit_env/bin/activate

#Trust. If you're using anything from huggingface, leave these lines it. These don't affect your job at all anyway, so really...just leave it in.
#export TRANSFORMERS_CACHE=/local/cache
export HF_HOME=/local/cache
# export HUGGINGFACE_TOKEN='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'
#export SENTENCE_TRANSFORMERS_HOME=/local/cache

# Set NCCL environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker0

# Explicitly set this,
#TODO: On SHARCNET, try to automatically compute it somehow?
NPROC_PER_NODE=1


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
# export GPUS_PER_NODE=2

# Redirect Triton cache directory to a writable location
# Otherwise, this will violate with permission of your home directory
# However, got this warning
# Warning: The cache directory for DeepSpeed Triton autotune, /mnt/slurm_nfs/h3trinh/.triton_cache, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.Warning: The cache directory for DeepSpeed Triton autotune, /mnt/slurm_nfs/h3trinh/.triton_cache, appears to be on an NFS system. While this is generally acceptable, if you experience slowdowns or hanging when DeepSpeed exits, it is recommended to set the TRITON_CACHE_DIR environment variable to a non-NFS path.
export TRITON_CACHE_DIR="/mnt/slurm_nfs/$USER/.triton_cache"
mkdir -p $TRITON_CACHE_DIR

# (Optional) Set XDG_CACHE_HOME to redirect other cache files
export XDG_CACHE_HOME="/mnt/slurm_nfs/$USER/.cache"

# Print nodes allocated and variables for debugging
echo "Nodes allocated:"
scontrol show hostnames $SLURM_JOB_NODELIST

echo "SLURM_NTASKS: ${SLURM_NTASKS}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES}"
echo "Using NPROC_PER_NODE: ${NPROC_PER_NODE}"

echo "Running torchrun with the following command:"
echo "torchrun --nnodes=\"${SLURM_JOB_NUM_NODES}\" --nproc_per_node=\"${NPROC_PER_NODE}\" \
--rdzv_id=\"${SLURM_JOB_ID}\" --rdzv_backend=\"c10d\" --rdzv_endpoint=\"${MASTER_ADDR}:${MASTER_PORT}\" \
distributed_vit_training.py --deepspeed --deepspeed_config \"deepspeed_config.json\" \
--batch_size \"32\" --epochs \"5\" --dataset_percentage \"0.01\" --data_path \"./data\""

# Run the training script with torchrun
torchrun --nnodes="${SLURM_JOB_NUM_NODES}" \
         --nproc_per_node="${NPROC_PER_NODE}" \
         --rdzv_id="${SLURM_JOB_ID}" \
         --rdzv_backend="c10d" \
         --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
         distributed_vit_training.py \
         --deepspeed \
         --deepspeed_config "deepspeed_config.json" \
         --batch_size "32" \
         --epochs "100" \
         --dataset_percentage "1" \
         --data_path "./data"

# Deactivate the virtual environment
deactivate