import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import psutil
import subprocess
import time

# Initialize the distributed process group
def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = '129.97.92.168'  # Replace with actual master node IP
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def log_metrics(rank, local_rank, logfile='metrics.log'):
    with open(logfile, 'a') as f:
        # Record CPU and memory usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        # Record GPU utilization and memory for the specific local GPU
        gpu_usage = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader', '--id={}'.format(local_rank)]
        ).decode().strip()
        gpu_memory = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader', '--id={}'.format(local_rank)]
        ).decode().strip()

        # Log the metrics with timestamp and rank
        log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Rank {rank} | Local GPU {local_rank} | CPU: {cpu_usage}%, Mem: {memory_usage}%, GPU: {gpu_usage}%, GPU Mem: {gpu_memory} MB\n"
        f.write(log_entry)
        print(log_entry)

# Perform matrix multiplication across GPUs
def matrix_multiply(rank, size):
    torch.manual_seed(1234)

    # Map global rank to the local GPU device (using modulus operator)
    local_rank = rank % torch.cuda.device_count()

    # Define matrix size for testing
    N = 1000

    # Initialize random matrices on the assigned local GPU
    A = torch.randn(N, N).cuda(local_rank)
    B = torch.randn(N, N).cuda(local_rank)

    # Perform matrix multiplication
    result = torch.matmul(A, B)

    # Log system metrics for this rank
    log_metrics(rank, local_rank)

    # Ensure synchronization across all processes
    if rank == 0:
        print(f"Rank {rank}: Matrix multiplication complete")
    dist.barrier()

    # Clean up the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    # Size is the number of processes you want to run
    size = 5

    # Set up distributed processing
    mp.spawn(init_process, args=(size, matrix_multiply), nprocs=size, join=True)
