#!/usr/bin/env python3
"""
data_utils.py

Minimal dataset factory used by abilation.py.

Supports MedMNIST datasets:
- breastmnist
- pneumoniamnist
- octmnist

Returns (train_dataset, test_dataset, num_classes).

Notes
- We set `as_rgb=True` so the images become 3-channel (needed for ResNet/MobileNet).
- Labels from MedMNIST come as numpy arrays (shape [1]); we convert them to int.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _medmnist_get(dataset_name: str, root: str, img_size: int) -> Tuple[Dataset, Dataset, int]:
    try:
        import medmnist
        from medmnist import INFO
    except Exception as e:
        raise ImportError(
            "medmnist is required for medical datasets. Install with: pip install medmnist"
        ) from e

    dataset_name = dataset_name.lower().strip()
    if dataset_name not in INFO:
        raise ValueError(
            f"Unknown MedMNIST dataset '{dataset_name}'. Available keys include: "
            f"{', '.join(sorted([k for k in INFO.keys() if k.endswith('mnist')])[:30])} ..."
        )

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    num_classes = len(info["label"])

    # Basic transform: resize -> tensor -> normalize
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Images are uint8 in [0,255] => ToTensor puts in [0,1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    def target_transform(y):
        # y is typically a numpy array like [0] or [3]
        if isinstance(y, (list, tuple)) and len(y) == 1:
            return int(y[0])
        try:
            import numpy as np
            if isinstance(y, np.ndarray):
                return int(y.reshape(-1)[0])
        except Exception:
            pass
        return int(y)

    os.makedirs(root, exist_ok=True)

    train_ds = DataClass(
        split="train",
        root=root,
        download=True,
        transform=tfm,
        target_transform=target_transform,
        as_rgb=True,
    )
    test_ds = DataClass(
        split="test",
        root=root,
        download=True,
        transform=tfm,
        target_transform=target_transform,
        as_rgb=True,
    )
    return train_ds, test_ds, num_classes


def get_dataset(dataset: str, data_dir: str, img_size: int = 224):
    """
    Main entry used by abilation.py.

    Args:
        dataset: dataset key
        data_dir: directory where data should be stored/downloaded
        img_size: resize images to img_size x img_size
    """
    dataset = dataset.lower().strip()

    if dataset in {"breastmnist", "pneumoniamnist", "octmnist"}:
        return _medmnist_get(dataset, data_dir, img_size)

    raise ValueError(
        f"Dataset '{dataset}' not supported by this data_utils.py. "
        f"Supported: breastmnist, pneumoniamnist, octmnist"
    )
