import os
import time
import torch
import GPUtil
import threading
from deepspeed import init_distributed

# Initialize the distributed process group using DeepSpeed
def init_dist_process_group(rank, size, fn, epochs=10, backend='nccl'):
    os.environ['MASTER_ADDR'] = '129.97.92.168'  # ecetesla0 master node IP
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize DeepSpeed's distributed environment
    init_distributed()
    fn(rank, size, epochs)

# Log GPU metrics once per epoch
def log_metrics_once_per_epoch(rank, local_rank, epoch, computation_time, logfile='metrics.csv'):
    # Define the header and check if the file already exists
    header = [
        "Timestamp", "Rank", "Epoch", "Computation_Time", "GPU_ID", "GPU_Name",
        "GPU_Load", "GPU_Free_Mem_MB", "GPU_Used_Mem_MB", "GPU_Total_Mem_MB", "GPU_Temp_C"
    ]
    
    # Write the header if the file doesn't exist
    file_exists = os.path.isfile(logfile)
    if not file_exists:
        with open(logfile, 'w') as f:
            f.write(",".join(header) + "\n")
    
    # Get GPU metrics
    gpus = GPUtil.getGPUs()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Log metrics for each GPU
    for gpu in gpus:
        gpu_info = [
            timestamp, rank, epoch, computation_time, gpu.id, gpu.name, 
            f"{gpu.load*100:.1f}%", f"{gpu.memoryFree:.1f}", f"{gpu.memoryUsed:.1f}",
            f"{gpu.memoryTotal:.1f}", f"{gpu.temperature}"
        ]
        
        # Write data row to the CSV file
        with open(logfile, 'a') as f:
            f.write(",".join(map(str, gpu_info)) + "\n")

# Define computation function
def matrix_multiply(rank, size, epochs):
    torch.manual_seed(1234)
    local_rank = rank % torch.cuda.device_count()
    
    N = 5000
    A = torch.randn(N, N).cuda(local_rank)
    B = torch.randn(N, N).cuda(local_rank)

    for epoch in range(epochs):
        start_time = time.time()

        # Perform matrix multiplication
        result = torch.matmul(A, B)
        
        end_time = time.time()
        computation_time = end_time - start_time
        print(f"Rank {rank}: Epoch {epoch + 1}/{epochs} complete in {computation_time:.4f} seconds")

        # Log metrics at the end of each epoch
        log_metrics_once_per_epoch(rank, local_rank, epoch + 1, computation_time, 'metrics.csv')
        
        torch.distributed.barrier()
    
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    init_dist_process_group(rank, size, matrix_multiply, epochs=10)
