import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import logging
import argparse
import deepspeed
import GPUtil
from tqdm import tqdm
import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Test")
    # parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    # parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="DeepSpeed config file")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--dataset_percentage', type=float, default=1, help='Percentage of CIFAR-10 dataset to use')
    parser.add_argument('--data_path', type=str, help='Path to the dataset')   
    deepspeed.add_config_arguments(parser)
   
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

    # Explicitly set device_ids for barrier to get rid of warning 
    # [rank0]:[W1126 21:29:24.178484654 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 to perform barrier as devices used by this process are currently unknown. 
    # This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier() to force use of a particular device,or call init_process_group() with a device_id.
    device_id = torch.cuda.current_device()
    dist.barrier(device_ids=[device_id])

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


# Log GPU metrics once per epoch
def log_metrics_once_per_epoch(rank, local_rank, epoch, computation_time, logfile):
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


#TODO: This function require more detail of benchmarking metrics record(communication, overhead, dataload, imbalance, ...?)
#TODO: This function setup the rank is not corrects,
# Find a way to configur the rank is not corresponding to
def setup_logging(save_dir,rank):
    """
    rank here corresponding to MPI processes' rank (not GPU rank)
    """
    logger = logging.getLogger(f'Rank_{rank}')
    logger.setLevel(logging.INFO)

    # Create file handler which logs messages
    fh = logging.FileHandler(f"{save_dir}/vit_training_rank_{rank}.log")
    fh.setLevel(logging.INFO)

    # Format log messages to include rank
    formatter = logging.Formatter('%(asctime)s - Rank %(rank)d - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger if not already added
    if not logger.handlers:
        logger.addHandler(fh)

    # Wrap the logger with LoggerAdapter to pass rank info
    logger = logging.LoggerAdapter(logger, {'rank': rank})

    return logger

def create_model():
    model = models.vit_b_16(weights=None)

    return model


def get_data_loader(args):

    # Preprocessing follows https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download dataset on rank 0 and synchronize other as Nebula code
    if args.rank == 0:
        dataset = datasets.CIFAR100(root=args.data_path, train=True, download=False, transform=transform)
        dist.barrier()
    else:
        dist.barrier()
        dataset = datasets.CIFAR100(root=args.data_path, train=True, download=False, transform=transform)

    # Subset of dataset to experiment with dataset_percentage
    subset_size = int(len(dataset) * args.dataset_percentage)
    indices = list(range(subset_size))
    subset_dataset = torch.utils.data.Subset(dataset, indices)

    # Distributed sampling ensures each GPU sees a unique subset
    #TODO: Research here on imbalance (in case of non-homogenous)
    sampler = DistributedSampler(subset_dataset, shuffle=True)

    data_loader = DataLoader(subset_dataset, batch_size=args.micro_batch_size_per_gpu, sampler=sampler, num_workers=2, pin_memory=True)
    
    return data_loader


def calculate_accuracy(outputs, labels):
    """Calculates accuracy for each class."""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).float()
    accuracy = correct.sum() / len(correct)

    return accuracy.item()


def train(save_dir, args, model_engine, data_loader, criterion, logger, rank, local_rank):

    # print(rank)
    # print(local_rank)
    # print(args.world_size)

    #TODO: Create directories based on configuration
    metrics_save_file=f"metrics.csv"
    metrics_save_file=os.path.join(save_dir ,metrics_save_file)

    model_engine.train()
    total_steps = len(data_loader)
    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)  # Shuffle data for each epoch
        epoch_start_time = time.time()
        running_loss = 0.0
        running_accuracy = 0.0

        with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, labels) in enumerate(data_loader):

                # Half precision in case of homogeneous training ony (enable this fp16 training)
                # inputs = inputs.to(args.local_rank, non_blocking=True).half()

                # This is for all 5 GPUs training
                inputs = inputs.to(args.local_rank, non_blocking=True)
                labels = labels.to(args.local_rank, non_blocking=True)

                outputs = model_engine(inputs)
                loss = criterion(outputs, labels)

                model_engine.backward(loss)
                model_engine.step()

                running_loss += loss.item()
                running_accuracy += calculate_accuracy(outputs, labels)

                # if batch_idx % 10 == 0:
                #     logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{total_steps}], Loss: {loss.item():.4f}")

                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "accuracy": running_accuracy / (batch_idx + 1)})

        epoch_loss = running_loss / total_steps
        epoch_accuracy = running_accuracy / total_steps
        epoch_time = time.time() - epoch_start_time

        # Log the metrics
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_time:.2f} seconds")

        log_metrics_once_per_epoch(rank, local_rank, epoch + 1, epoch_time, metrics_save_file)

        torch.cuda.empty_cache()


def main():
    
    args = parse_args()
    # Setup distributed environment
    rank, world_size, local_rank = setup_deepspeed()
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size
    args.data_path = '/h/h3trinh/cifar100'
    model = create_model()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # model_engine, optimizer, _, _ = deepspeed.initialize(
    #     args=args,
    #     model=model,
    #     optmizer=optimizer,
    #     model_parameters=model.parameters()
    # )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters()
    )


    args.micro_batch_size_per_gpu = model_engine.train_micro_batch_size_per_gpu()
    print(f"model_engine.train_micro_batch_size_per_gpu(): {model_engine.train_micro_batch_size_per_gpu()}")

    if world_size == 1:
        save_dir = f"{os.getcwd()}/log/cifar100/t4v2/{world_size}_GPU"
    else:
        save_dir = f"{os.getcwd()}/log/cifar100/t4v2/{world_size}_GPUs"

    save_dir=f"{save_dir}/p{args.dataset_percentage}_b{args.micro_batch_size_per_gpu}_e{args.epochs}"
    print(save_dir)

    if rank == 0:
        if not os.path.exists(save_dir):
            print("Path {save_dir} not exist. Creating..")
            os.makedirs(save_dir, exist_ok=True)
    
    dist.barrier()

    logger = setup_logging(save_dir=save_dir,rank=rank)

    # Set up DataLoader for distributed training
    data_loader = get_data_loader(args)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Start training
    train(save_dir, args, model_engine, data_loader, criterion, logger, rank, local_rank)

    # logger.info(f"Training completed on Rank: {rank}")

    # Clean up the distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
