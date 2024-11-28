import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import argparse
import deepspeed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--dataset_percentage", type=float, default=0.01, help="Percentage of CIFAR-10 dataset to use")
    parser.add_argument("--data_path", type=str, default="/path/to/data", help="Path to the dataset")
    deepspeed.add_config_arguments(parser)
    return parser.parse_args()


import os
import torch
import torch.distributed as dist
import deepspeed

def setup_deepspeed_environment():
    """
    Initialize the DeepSpeed distributed environment.
    """
    deepspeed.init_distributed()  # Initialize distributed backend

    # Get rank information
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Rank within the current node
    global_rank = int(os.environ.get("RANK", 0))       # Global rank across all nodes
    world_size = int(os.environ.get("WORLD_SIZE", 1))  # Total number of processes

    # Set the device for the current process
    torch.cuda.set_device(local_rank)

    print(f"[Rank {global_rank}/{world_size}] Initialized on GPU {local_rank}.")
    return local_rank, global_rank, world_size



def create_model():
    """Create the model (e.g., ViT)."""
    return models.vit_b_16(weights=None)


def get_data_loader(args):
    """Prepare the dataset and DataLoader."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if int(os.environ["RANK"]) == 0:
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        dist.barrier()  # Synchronize dataset download across nodes
    else:
        dist.barrier()
        dataset = datasets.CIFAR10(root=args.data_path, train=True, download=False, transform=transform)

    subset_size = int(len(dataset) * args.dataset_percentage)
    indices = list(range(subset_size))
    subset_dataset = torch.utils.data.Subset(dataset, indices)

    sampler = DistributedSampler(subset_dataset, shuffle=True)
    data_loader = DataLoader(subset_dataset, batch_size=args.micro_batch_size_per_gpu, sampler=sampler, num_workers=4, pin_memory=True)
    return data_loader


def train(save_dir, args, model_engine, data_loader, criterion):
    """Training loop."""
    model_engine.train()
    total_steps = len(data_loader)

    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        running_loss = 0.0

        with tqdm(total=total_steps, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as pbar:
            for inputs, labels in data_loader:
                inputs = inputs.to(args.local_rank, non_blocking=True)
                labels = labels.to(args.local_rank, non_blocking=True)

                outputs = model_engine(inputs)
                loss = criterion(outputs, labels)

                model_engine.backward(loss)
                model_engine.step()

                running_loss += loss.item()
                pbar.update(1)

        epoch_loss = running_loss / total_steps
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} completed in {epoch_time:.2f}s with loss {epoch_loss:.4f}")

import torch
import deepspeed
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Training")
    # parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="Path to DeepSpeed config file")
    parser.add_argument('--data_path', type=str, help='Path to the dataset')   
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--dataset_percentage", type=float, default=0.01, help="Fraction of dataset to use for training")
    deepspeed.add_config_arguments(parser)
    return parser.parse_args()

def create_model():
    """Create the model (ViT) for training."""
    model = models.vit_b_16(weights=None)  # Vision Transformer, no pre-trained weights
    return model

def prepare_data_loader(args, local_rank):
    """Prepare the CIFAR-10 dataset and DataLoader."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download dataset on rank 0
    # if local_rank == 0:
    #     datasets.CIFAR10(root=args.data_path, train=True, download=False)
    # dist.barrier()  # Synchronize dataset across ranks

    dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=transform)
    subset_size = int(len(dataset) * args.dataset_percentage)
    dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))

    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size_per_gpu,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    return dataloader

def train_one_epoch(epoch, model_engine, data_loader, criterion, local_rank):
    """Train for one epoch."""
    model_engine.train()
    total_loss = 0
    data_loader.sampler.set_epoch(epoch)

    with tqdm(data_loader, desc=f"Epoch {epoch+1}", unit="batch", disable=(local_rank != 0)) as pbar:
        for inputs, labels in pbar:
            inputs = inputs.to(local_rank)
            labels = labels.to(local_rank)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(data_loader)

def main():
    args = parse_args()
    local_rank, global_rank, world_size = setup_deepspeed_environment()
    args.data_path = '/h/h3trinh/cifar10'

    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters()
    )

    args.micro_batch_size_per_gpu = model_engine.train_micro_batch_size_per_gpu()
    data_loader = prepare_data_loader(args, local_rank)
    criterion = CrossEntropyLoss().to(local_rank)

    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(epoch, model_engine, data_loader, criterion, local_rank)
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    main()
