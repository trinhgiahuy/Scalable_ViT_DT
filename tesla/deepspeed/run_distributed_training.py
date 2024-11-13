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

#TODO: The script should parse the argument and create a new directory into log directory,
#directory contain the configuration
#
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed ViT Training with DeepSpeed")
    
    # Training parameters
    #TODO: Add more later on
    #Add these arguments in entry_cn.sh file
    #?: Not sure but DeepSpeed does not use batch_size but use train_batch_size its own
    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size per GPU')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--dataset_percentage', type=float, default=0.1, help='Percentage of CIFAR-10 dataset to use')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')

    #TODO: Read docs again about DeepSpeed offered params to avoid conflict with use defined vars
    # DeepSpeed parameters
    deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    return args

#TODO: Check this why in each local rank log has 5 lines (expecte 1 line for each? Check Nebula log file again)
def setup_distributed():
    # Initialize the distributed process group
    deepspeed.init_distributed()

    # Retrieve local rank and other environment variables
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    return local_rank, rank, world_size

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

#TODO: This function require more detail of benchmarking metrics record(communication, overhead, dataload, imbalance, ...?)
#TODO: This function setup the rank is not corrects, 
# Find a way to configur the rank is not corresponding to 
def setup_logging(rank):
    """
    rank here corresponding to MPI processes' rank (not GPU rank)
    """
    logger = logging.getLogger(f'Rank_{rank}')
    logger.setLevel(logging.INFO)

    # Create file handler which logs messages
    fh = logging.FileHandler(f'vit_training_rank_{rank}.log')
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
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
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

    data_loader = DataLoader(subset_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return data_loader

def calculate_accuracy(outputs, labels):
    """Calculates accuracy for each class."""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()

def train(args, model_engine, data_loader, criterion, logger, rank, local_rank):
    model_engine.train()
    total_steps = len(data_loader)
    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)  # Shuffle data for each epoch
        epoch_start_time = time.time()
        running_loss = 0.0
        running_accuracy = 0.0

        with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, labels) in enumerate(data_loader):

                inputs = inputs.to(args.local_rank, non_blocking=True).half()
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
    
        log_metrics_once_per_epoch(rank, local_rank, epoch + 1, epoch_time, 'metrics.csv')

        torch.cuda.empty_cache()

def main():
    args = parse_args()
    local_rank, rank, world_size = setup_distributed()
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size

    logger = setup_logging(rank)
    # logger.info(f"Starting training on Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")

    model = create_model()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize DeepSpeed with the specified configuration file
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters()
    )

    # Set up DataLoader for distributed training
    data_loader = get_data_loader(args)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Start training
    train(args, model_engine, data_loader, criterion, logger, rank, local_rank)

    # logger.info(f"Training completed on Rank: {rank}")

    # Clean up the distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()