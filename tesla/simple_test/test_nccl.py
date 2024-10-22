import os
import torch
import torch.distributed as dist

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '129.97.92.168'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def simple_nccl_test(rank, size):
    """ A simple NCCL test """
    available_gpus = torch.cuda.device_count()
    
    # Assign GPU based on rank mod available GPUs
    gpu_id = rank % available_gpus
    
    tensor = torch.ones(1).cuda(gpu_id)
    print(f'[BEFORE ALL REDUCE] Rank {rank} using GPU {gpu_id} has data {tensor[0]}')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f'[AFTER ALL REDUCE] Rank {rank} using GPU {gpu_id} has data {tensor[0]}')

if __name__ == "__main__":
    
    # Get the total number of processes
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))  
    # Use MPI environment variable for rank
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])  
    init_process(rank, size, simple_nccl_test)
