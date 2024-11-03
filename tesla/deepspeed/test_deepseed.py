import torch
import deepspeed
import os
import time
import threading
from deepspeed import init_distributed

# Initialize the distributed process group using DeepSpeed
def init_dist_process_group(rank, size, fn, epochs=10, backend='nccl'):
    os.environ['MASTER_ADDR'] = '129.97.92.168'  # ecetesla0 master node IP
    os.environ['MASTER_PORT'] = '29500'

    # Initialize DeepSpeed's distributed environment as Nebula code
    init_distributed()  
    fn(rank, size, epochs)

def log_metrics_during_computation(rank, local_rank, logfile='metrics.txt', stop_event=None):
    while not stop_event.is_set():
        gpu_util = torch.cuda.utilization(local_rank)
        gpu_mem = torch.cuda.memory_allocated(local_rank) / 1024 / 1024  # Convert to MB
        log_entry = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Rank {rank} | GPU Util: {gpu_util}%, GPU Mem: {gpu_mem:.2f} MB\n"
        
        with open(logfile, 'a') as f:
            f.write(log_entry)
        time.sleep(1)

def matrix_multiply(rank, size, epochs):
    torch.manual_seed(1234)
    local_rank = rank % torch.cuda.device_count()
    
    N = 5000
    A = torch.randn(N, N).cuda(local_rank)
    B = torch.randn(N, N).cuda(local_rank)

    for epoch in range(epochs):
        start_time = time.time()
        stop_event = threading.Event()
        logging_thread = threading.Thread(target=log_metrics_during_computation, args=(rank, local_rank, 'metrics.txt', stop_event))
        logging_thread.start()

        result = torch.matmul(A, B)
        end_time = time.time()
        
        computation_time = end_time - start_time
        print(f"Rank {rank}: Epoch {epoch + 1}/{epochs} complete in {computation_time:.4f} seconds")

        with open('metrics.txt', 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Rank {rank} | Epoch {epoch + 1} | Computation Time: {computation_time:.4f} seconds\n")
        
        stop_event.set()
        logging_thread.join()
        
        torch.distributed.barrier()
    
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    init_dist_process_group(rank, size, matrix_multiply, epochs=10)