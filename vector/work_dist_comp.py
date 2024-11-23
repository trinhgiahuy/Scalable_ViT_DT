import os
import torch
import deepspeed
import torch.distributed as dist
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Test")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="DeepSpeed config file")
    return parser.parse_args()


def setup_deepspeed():
    """Initialize DeepSpeed distributed environment."""
    deepspeed.init_distributed()

    # Get environment variables for rank, world size, and local rank
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set CUDA device for the local rank
    torch.cuda.set_device(local_rank)

    print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}.")
    return rank, world_size, local_rank


def distributed_computation(rank, world_size):
    """Perform basic distributed operations."""
    device = torch.device(f"cuda:{rank}")
    tensor = torch.tensor([rank], device=device, dtype=torch.float32)

    # Broadcast from rank 0
    if rank == 0:
        print(f"Rank {rank}: Broadcasting tensor {tensor.item()}")
    dist.broadcast(tensor, src=0)
    print(f"Rank {rank}: After broadcast, tensor = {tensor.item()}")

    # Reduce to rank 0 (sum across ranks)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"Rank {rank}: After reduce, tensor = {tensor.item()}")

    # All-reduce (sum across ranks, all receive the result)
    tensor = torch.tensor([rank], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: After all-reduce, tensor = {tensor.item()}")


def main():
    args = parse_args()

    # Setup distributed environment
    rank, world_size, local_rank = setup_deepspeed()

    # Perform distributed computation
    distributed_computation(rank, world_size)

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
