# data_utils.py
import os
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(img_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, test_tf


def get_dataset(dataset_name: str,
                data_dir: str,
                img_size: int = 224) -> Tuple[torch.utils.data.Dataset,
                                              torch.utils.data.Dataset,
                                              int]:
    dataset_name = dataset_name.lower()
    train_tf, test_tf = get_transforms(img_size)

    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root=data_dir, train=True,
                                    download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(root=data_dir, train=False,
                                   download=True, transform=test_tf)
        num_classes = 10

    elif dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(root=data_dir, train=True,
                                     download=True, transform=train_tf)
        test_ds = datasets.CIFAR100(root=data_dir, train=False,
                                    download=True, transform=test_tf)
        num_classes = 100

    elif dataset_name == "svhn":
        train_ds = datasets.SVHN(root=data_dir, split="train",
                                 download=True, transform=train_tf)
        test_ds = datasets.SVHN(root=data_dir, split="test",
                                download=True, transform=test_tf)
        num_classes = 10

    elif dataset_name == "flowers102":
        # Requires torchvision>=0.15
        train_ds = datasets.Flowers102(root=data_dir, split="train",
                                       download=True, transform=train_tf)
        test_ds = datasets.Flowers102(root=data_dir, split="test",
                                      download=True, transform=test_tf)
        num_classes = 102

    elif dataset_name == "tinyimagenet":
        # Expect folder: data_dir/tinyimagenet/{train,val}/class_x/xxx.png
        train_root = os.path.join(data_dir, "tinyimagenet", "train")
        val_root = os.path.join(data_dir, "tinyimagenet", "val")
        train_ds = datasets.ImageFolder(root=train_root, transform=train_tf)
        test_ds = datasets.ImageFolder(root=val_root, transform=test_tf)
        num_classes = len(train_ds.classes)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_ds, test_ds, num_classes


def split_dataset_iid(train_ds, num_clients: int) -> List[Subset]:
    """Simple IID split: equal shards of indices."""
    n = len(train_ds)
    indices = torch.randperm(n).tolist()
    shard_size = n // num_clients
    client_subsets = []

    for i in range(num_clients):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_clients - 1 else n
        client_indices = indices[start:end]
        client_subsets.append(Subset(train_ds, client_indices))

    return client_subsets


def make_dataloaders_for_clients(client_subsets,
                                 batch_size: int,
                                 num_workers: int = 4):
    loaders = []
    for subset in client_subsets:
        loaders.append(
            DataLoader(subset, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers,
                       pin_memory=True)
        )
    return loaders


def make_test_loader(test_ds,
                     batch_size: int,
                     num_workers: int = 4):
    return DataLoader(test_ds, batch_size=batch_size,
                      shuffle=False, num_workers=num_workers,
                      pin_memory=True)
