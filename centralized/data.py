from typing import Tuple

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


def get_cifar10_loaders(data_root: str,
                        batch_size: int,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10, "cifar10"


def get_cifar100_loaders(data_root: str,
                         batch_size: int,
                         num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 100, "cifar100"


def get_svhn_loaders(data_root: str,
                     batch_size: int,
                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = datasets.SVHN(root=data_root, split="train", download=True, transform=transform)
    test_ds = datasets.SVHN(root=data_root, split="test", download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10, "svhn"


def get_tinyimagenet_loaders(data_root: str,
                             batch_size: int,
                             num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    """
    Expects:
      <data_root>/tiny-imagenet-200/train/...
      <data_root>/tiny-imagenet-200/val/...
    """
    train_dir = f"{data_root}/tiny-imagenet-200/train"
    val_dir = f"{data_root}/tiny-imagenet-200/val"

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    test_ds = datasets.ImageFolder(val_dir, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # TinyImageNet has 200 classes
    return train_loader, test_loader, 200, "tinyimagenet"


def get_flowers102_loaders(data_root: str,
                           batch_size: int,
                           num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int, str]:
    """
    Torchvision provides Flowers102 with split: "train", "val", "test".
    We'll combine train+val for training; test for evaluation.
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = datasets.Flowers102(root=data_root, split="train", download=True, transform=transform_train)
    val_ds = datasets.Flowers102(root=data_root, split="val", download=True, transform=transform_train)
    test_ds = datasets.Flowers102(root=data_root, split="test", download=True, transform=transform_test)

    train_concat = ConcatDataset([train_ds, val_ds])

    train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 102, "flowers102"


def get_loaders(dataset: str,
                data_root: str,
                batch_size: int,
                num_workers: int = 4):
    dataset = dataset.lower()
    if dataset == "cifar10":
        return get_cifar10_loaders(data_root, batch_size, num_workers)
    if dataset == "cifar100":
        return get_cifar100_loaders(data_root, batch_size, num_workers)
    if dataset == "svhn":
        return get_svhn_loaders(data_root, batch_size, num_workers)
    if dataset == "tinyimagenet":
        return get_tinyimagenet_loaders(data_root, batch_size, num_workers)
    if dataset == "flowers102":
        return get_flowers102_loaders(data_root, batch_size, num_workers)

    raise ValueError(f"Unsupported dataset: {dataset}")
