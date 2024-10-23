import torch
import torch.distributed as dist
import os
import psutil
import subprocess
import time
import threading

# Initialize the distributed process group
def init_process(rank, size, fn, epochs=10, backend='nccl'):
    os.environ['MASTER_ADDR'] = '129.97.92.168'  # ecetesla0 master node IP
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, epochs)

# Function to log system metrics during computation
def log_metrics_during_computation(rank, local_rank, logfile='metrics.log', stop_event=None):
    with open(logfile, 'a') as f:
        while not stop_event.is_set():
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            gpu_usage = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader', '--id={}'.format(local_rank)]
            ).decode().strip()
            gpu_memory = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader', '--id={}'.format(local_rank)]
            ).decode().strip()

            log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Rank {rank} | Local GPU {local_rank} | CPU: {cpu_usage}%, Mem: {memory_usage}%, GPU: {gpu_usage}%, GPU Mem: {gpu_memory} MB\n"
            f.write(log_entry)
            print(log_entry)
            time.sleep(1)

# Perform matrix multiplication across GPUs
def matrix_multiply(rank, size, epochs):
    torch.manual_seed(1234)

    if not torch.cuda.is_available():
        print(f"Rank {rank}: CUDA is not available!")
        return

    # Map global rank to the local GPU device
    local_rank = rank % torch.cuda.device_count()

    # Define matrix size for testing
    N = 5000

    # Initialize random matrices on the assigned local GPU
    A = torch.randn(N, N).cuda(local_rank)
    B = torch.randn(N, N).cuda(local_rank)

    for epoch in range(epochs):
        print(f"Rank {rank} | Epoch {epoch + 1}/{epochs}: Starting matrix multiplication")

        stop_event = threading.Event()
        logging_thread = threading.Thread(target=log_metrics_during_computation, args=(rank, local_rank, 'metrics.log', stop_event))
        logging_thread.start()

        start_time = time.time()
        result = torch.matmul(A, B)
        end_time = time.time()

        stop_event.set()
        logging_thread.join()

        print(f"Rank {rank} | Epoch {epoch + 1}/{epochs} | Computation Time: {end_time - start_time:.4f} seconds")

        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    init_process(rank, size, matrix_multiply, epochs=10)
