# data.py
"""
Data loaders for MedMNIST datasets used in your dropout-MIA pipeline.

Supported:
- breastmnist (binary)
- pneumoniamnist (binary)
- octmnist (4-class)

Notes
- We set as_rgb=True so that images become 3-channel and can be fed to
  torchvision ResNet/MobileNet without modifying their first conv layer.
- We resize to 224x224 to match the default receptive field expectations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as T
import os



@dataclass
class LoaderBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    dataset_name: str


class _SqueezeLabel(torch.utils.data.Dataset):
    """MedMNIST labels are often shape (1,) numpy arrays; squeeze to scalar int."""
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        # y can be numpy array, list, int, or tensor
        if isinstance(y, torch.Tensor):
            y = y.view(-1)[0].long()
        else:
            try:
                # numpy array / list-like
                y = int(y[0])  # type: ignore[index]
            except Exception:
                y = int(y)
            y = torch.tensor(y, dtype=torch.long)
        return x, y


def _medmnist_transform(img_size: int = 224) -> T.Compose:
    # ImageNet normalization works fine for 3-channel resized images.
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def get_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, int, str]:
    """
    Returns:
        train_loader, test_loader, num_classes, dataset_name
    """
    name = dataset.lower().strip()
    os.makedirs(data_root, exist_ok=True)


    try:
        import medmnist  # noqa: F401
        from medmnist import INFO
    except Exception as e:
        raise ImportError(
            "medmnist is required. Install with: pip install medmnist"
        ) from e

    if name not in {"breastmnist", "pneumoniamnist", "octmnist"}:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Use: breastmnist | pneumoniamnist | octmnist"
        )

    info = INFO[name]
    DataClass = getattr(__import__("medmnist", fromlist=[info["python_class"]]), info["python_class"])

    transform = _medmnist_transform(img_size=img_size)

    # Download and build splits
    train_ds = DataClass(split="train", root=data_root, download=True, transform=transform, as_rgb=True)
    test_ds = DataClass(split="test", root=data_root, download=True, transform=transform, as_rgb=True)

    train_ds = _SqueezeLabel(train_ds)
    test_ds = _SqueezeLabel(test_ds)

    # Labels are single-label classification for these datasets
    num_classes = len(info["label"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader, num_classes, name
