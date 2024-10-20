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

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed ViT Training with DeepSpeed")
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size per GPU')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--dataset_percentage', type=float, default=0.01, help='Percentage of dataset to use')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')

    # DeepSpeed parameters
    deepspeed.add_config_arguments(parser)  # Do not reassign parser here

    args = parser.parse_args()
    return args

def setup_distributed():
    # Initialize the distributed process group
    deepspeed.init_distributed()

    # Retrieve local rank and other environment variables
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    return local_rank, rank, world_size

def setup_logging(rank):
    logger = logging.getLogger(f'Rank_{rank}')
    logger.setLevel(logging.INFO)

    # Create file handler which logs messages
    fh = logging.FileHandler(f'vit_training_rank_{rank}.log')
    fh.setLevel(logging.INFO)

    # Use 'rank' with LoggerAdapter
    formatter = logging.Formatter('%(asctime)s - Rank %(rank)d - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)

    # Add the handler to the logger if not already added
    if not logger.handlers:
        logger.addHandler(fh)

    # Wrap the logger with LoggerAdapter
    logger = logging.LoggerAdapter(logger, {'rank': rank})

    return logger

def create_model():
    # Create a Vision Transformer model
    model = models.vit_b_16(weights=None)  # Initialize with random weights
    return model

def get_real_data_loader(args):
    # Use CIFAR-10 dataset as an example
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Only download the dataset on rank 0
    if args.rank == 0:
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        dist.barrier()  # Synchronize after downloading
    else:
        dist.barrier()  # Wait for rank 0 to download
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)

    # Create a subset of the dataset
    subset_size = int(len(dataset) * args.dataset_percentage)
    indices = list(range(subset_size))
    subset_dataset = torch.utils.data.Subset(dataset, indices)

    # Now create the DistributedSampler on the subset dataset
    sampler = DistributedSampler(subset_dataset, shuffle=True)

    data_loader = DataLoader(subset_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return data_loader


def train(args, model_engine, data_loader, criterion, logger):
    model_engine.train()
    total_steps = len(data_loader)
    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(args.local_rank, non_blocking=True).half()
            labels = labels.to(args.local_rank, non_blocking=True)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            if batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{total_steps}], Loss: {loss.item():.4f}")

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] completed in {epoch_time:.2f} seconds")

def main():
    args = parse_args()
    local_rank, rank, world_size = setup_distributed()
    args.local_rank = local_rank
    args.rank = rank
    args.world_size = world_size

    logger = setup_logging(rank)
    logger.info(f"Starting training on Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")

    # Create the model
    model = create_model()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters()
    )

    # Choose the data loader
    data_loader = get_real_data_loader(args)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Start training
    train(args, model_engine, data_loader, criterion, logger)

    logger.info(f"Training completed on Rank: {rank}")

    # Clean up the distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
