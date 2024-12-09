import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import deepspeed
import torch.distributed as dist
import time

# from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import logging
import os
import torch.nn as nn
import GPUtil
from tqdm import tqdm
import datetime


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def setup_deepspeed():
    
    deepspeed.init_distributed()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    # device_id = torch.cuda.current_device()
    # dist.barrier(device_ids=[device_id])

    print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}.")

    return rank, world_size, local_rank

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


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)



class MyTrainDataset(Dataset):
    def __init__(self, num_samples=2048):
        super(MyTrainDataset, self).__init__()
        self.len=num_samples

    def __getitem__(self, index):
        x=torch.randn((20))
        y=torch.randn((1))
        return x, y

    def __len__(self):
        return self.len


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def create_model():
    model = models.vit_b_16(weights=None)

    return model

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def calculate_accuracy(outputs, labels):
    """Calculates accuracy for each class."""
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).float()
    accuracy = correct.sum() / len(correct)

    return accuracy.item()


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



# def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
def main():

    args = parse_args()
    print('initilaizing deepspeed')
    # ddp_setup()
    rank, world_size, local_rank = setup_deepspeed()
    print('deepspeed initilaized')
    # dataset, model, optimizer = load_train_objs()
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

    # train_data = prepare_dataloader(dataset, batch_size)
    # trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    # trainer.train(total_epochs)
    # destroy_process_group()

    # Set up DataLoader for distributed training
    data_loader = get_data_loader(args)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Start training
    train(save_dir, args, model_engine, data_loader, criterion, logger, rank, local_rank)

    # logger.info(f"Training completed on Rank: {rank}")

    # Clean up the distributed process group
    dist.destroy_process_group()

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Test")
    # parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    # parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="DeepSpeed config file")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--dataset_percentage', type=float, default=1, help='Percentage of CIFAR-10 dataset to use')
    parser.add_argument('--data_path', type=str, help='Path to the dataset')   
    deepspeed.add_config_arguments(parser)
   
    return parser.parse_args()

if __name__ == "__main__":
    main()