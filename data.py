import torch
import torchvision
from torch.utils.data import random_split, Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from dataset import *

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)


def get_dataloaders(
    dataset_name: str = "mnist",
    batch_size: int = 32,
    train_transforms=None,
    test_transforms=None,
):
    # usually, we want to split data into training, validation, and test sets
    # for simplicity, we will only use training and test sets
    if train_transforms is None:
        train_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    if test_transforms is None:
        test_transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    if dataset_name == "mnist":
        # Do Not Change This Code
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transforms
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transforms
        )

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif dataset_name == "cifar100":
        # TODO: Load CIFAR100 dataset
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transforms)
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=test_transforms)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    elif dataset_name == "products":
        paths = [
            "clean_products125k-yolo",
            "products125k_generated_copy_paste",
            "pure_background",
            "products125k_generated_bg_single",
            "products125k_hands_bg",
            "SKU-110K-r-yolo",
            "single_product_hands",
            # "nondiff_multi",
            "nondiff_multi_dense",
        ]

        # train_size = 200
        # test_size = 100

        # # Ensure the dataset is large enough
        # if len(dataset) < train_size + test_size:
        #     raise ValueError(f"Dataset size {len(dataset)} is smaller than the required train size ({train_size}) + test size ({test_size}).")

        # # Split the dataset into training and test sets
        # train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, len(dataset) - train_size - test_size])

        
        paths = [Path(f"/home/biometrics/zhantaoy/datasets/{path}") for path in paths]
        # Remove DistributedSampler from dataset initialization
        dataset = ImageLabelDataset(root_dirs=paths, transform=train_transforms)
        
        print(f"Size of the dataset {len(dataset)}")

        train_split_ratio = 0.7
        train_size = int(train_split_ratio * len(dataset))
        test_size = len(dataset) - train_size

        torch.manual_seed(42)  # Or any other fixed seed value
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        print(f"Train set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")

        # Use DistributedSampler in DataLoader if needed for distributed training
        train_sampler = DistributedSampler(train_dataset)  # Add this if using distributed training
        # test_sampler = DistributedSampler(test_dataset)    # Add this if needed for distributed testing

        # Create the DataLoader for both training and test sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, sampler=train_sampler, worker_init_fn=seed_worker)
        # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=False, sampler=test_sampler, worker_init_fn=seed_worker)
        test_loader = None

    elif dataset_name == "imagenet-tiny":        
        top_path = Path("/home/biometrics/ritarka/vision/data/tiny-imagenet-200")
        train_dataset = TinyImageNet(top_path / "train", transform=train_transforms)
        test_dataset = TinyImageNet(top_path / "val", transform=test_transforms, validation=True)
        
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=20, pin_memory=True,     sampler=train_sampler, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False, sampler=test_sampler, worker_init_fn=seed_worker)

        

    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return train_loader, test_loader
