import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torchvision import models, datasets, transforms
import logging
import argparse
import deepspeed
import GPUtil
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Training Verification")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--dataset_percentage', type=float, default=0.01, help='Percentage of CIFAR-10 dataset to use')
    deepspeed.add_config_arguments(parser)  # Add DeepSpeed-specific arguments

    return parser.parse_args()


def setup_distributed():
    # Initialize the distributed process group
    deepspeed.init_distributed()

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


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
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)
        dist.barrier()
    else:
        dist.barrier()
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)

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
    model_engine.train()
    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        
        total_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(args.local_rank).float()
            labels = labels.to(args.local_rank)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")

        # All-reduce to compute the average loss across all ranks
        avg_loss_tensor = torch.tensor(total_loss, device=args.local_rank)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / args.world_size

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_time:.2f} seconds. Average Loss: {avg_loss:.4f}")


def main():
    args = parse_args()
    # args.data_path = '/h/h3trinh/cifar10/cifar-10-batches-py'  # Update with your dataset path
    args.data_path = '/datasets/cifar10' 
    local_rank, rank, world_size = setup_distributed()
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size

    # logger = setup_logging(rank)
    # logger.info(f"Starting distributed training: Rank {rank}, Local Rank {local_rank}, World Size {world_size}")

    # Model and optimizer
    model = create_model()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters()
    )

    args.micro_batch_size_per_gpu = model_engine.train_micro_batch_size_per_gpu()
    print(f"model_engine.train_micro_batch_size_per_gpu(): {model_engine.train_micro_batch_size_per_gpu()}")

    if world_size == 1:
        save_dir = f"{os.getcwd()}/log/{world_size}_GPU"
    else:
        save_dir = f"{os.getcwd()}/log/{world_size}_GPUs"
        save_dir=f"{save_dir}/p{args.dataset_percentage}_b{args.micro_batch_size_per_gpu}_e{args.epochs}"
    
    print(save_dir)
    
    if rank == 0:
        if not os.path.exists(save_dir):
            print("Path {save_dir} not exist. Creating..")
            os.makedirs(save_dir, exist_ok=True)

    dist.barrier()

    logger = setup_logging(save_dir=save_dir,rank=rank)

    # Data loader
    data_loader = get_data_loader(args)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Train the model
    train(save_dir, args, model_engine, data_loader, criterion, logger, rank, local_rank)

    logger.info(f"Training completed on Rank {rank}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
