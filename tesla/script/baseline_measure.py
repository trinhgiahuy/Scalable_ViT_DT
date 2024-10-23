import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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

# def log_metrics(rank, local_rank, logfile='metrics.txt'):
#     with open(logfile, 'a') as f:
#         # Record CPU and memory usage
#         cpu_usage = psutil.cpu_percent(interval=1)
#         memory_usage = psutil.virtual_memory().percent

#         # Record GPU utilization and memory for the specific local GPU
#         gpu_usage = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader', '--id={}'.format(local_rank)]
#         ).decode().strip()
#         gpu_memory = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader', '--id={}'.format(local_rank)]
#         ).decode().strip()

#         # Log the metrics with timestamp and rank
#         log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Rank {rank} | Local GPU {local_rank} | CPU: {cpu_usage}%, Mem: {memory_usage}%, GPU: {gpu_usage}%, GPU Mem: {gpu_memory} MB\n"
#         f.write(log_entry)
#         print(log_entry)

#TODO: Probably find some alternative ways to this function, NVIDIA open soure benchmarking library?
# For example, memory-profiler, ...
def log_metrics_during_computation(rank, local_rank, logfile='metrics.txt', stop_event=None):
    with open(logfile, 'a') as f:
        while not stop_event.is_set():
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

            # Sleep for 1 second before logging again
            time.sleep(1)

# Perform matrix multiplication across GPUs
def matrix_multiply(rank, size, epochs):
    torch.manual_seed(1234)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print(f"Rank {rank}: CUDA is not available!")
        return

    # Map global rank to the local GPU device
    local_rank = rank % torch.cuda.device_count()

    # Define matrix size for testing (increase N for a longer operation)
    N = 5000

    # Initialize random matrices on the assigned local GPU
    A = torch.randn(N, N).cuda(local_rank)
    B = torch.randn(N, N).cuda(local_rank)

    for epoch in range(epochs):
        start_time = time.time()

        # Create an event to stop the logging thread once matrix multiplication finishes
        stop_event = threading.Event()

        # Start logging metrics in a separate thread
        logging_thread = threading.Thread(target=log_metrics_during_computation, args=(rank, local_rank, 'metrics.txt', stop_event))
        logging_thread.start()

        print(f"Rank {rank}: Starting Epoch {epoch + 1}/{epochs}")

        # Perform matrix multiplication in a loop to ensure sustained computation
        result = torch.matmul(A, B)
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Rank {rank}: Epoch {epoch + 1}/{epochs} complete in {computation_time:.4f} seconds")

        # Log computation time in metrics
        with open('metrics.txt', 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Rank {rank} | Epoch {epoch + 1} | Computation Time: {computation_time:.4f} seconds\n")
        
        # Stop logging metrics after the computation finishes
        stop_event.set()

        # Ensure synchronization across all processes
        dist.barrier()

        # if rank == 0:
            # print(f"Rank {rank}: Matrix multiplication complete")

        # Wait for the logging thread to finish
        logging_thread.join()

    # Clean up the process group
    dist.destroy_process_group()

if __name__ == "__main__":
    # Size is the number of processes you want to run
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    init_process(rank, size, matrix_multiply, epochs=10)

    # [NOT USE] Set up distributed processing
    # mp.spawn(init_process, args=(size, matrix_multiply), nprocs=size, join=True)