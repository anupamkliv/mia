# train.py
"""
Train a single classifier (ResNet18/34 or MobileNetV3 Small/Large) on a MedMNIST dataset
and save weights to: dropout_results/<dataset>/<model>.pth

Training-time dropout is controlled via --train-dropout-p (default 0.0).
This matches your pipeline where dropout perturbation is applied during inference, not training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn

from data import get_loaders
from models import get_model, set_mc_dropout_p
from utils import set_seed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train on MedMNIST and save checkpoint as dropout_results/<dataset>/<model>.pth"
    )
    p.add_argument("--dataset", type=str, default="breastmnist",
                   help="breastmnist | pneumoniamnist | octmnist")
    p.add_argument("--data-root", type=str, default="./data")

    p.add_argument("--model", type=str, default="resnet18",
                   help="resnet18 | resnet34 | mobilenetv3_small | mobilenetv3_large")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # IMPORTANT: training dropout (you usually want 0.0)
    p.add_argument("--train-dropout-p", type=float, default=0.0,
                   help="Dropout p during training (0.0 reproduces your setting)")

    p.add_argument("--seed", type=int, default=42)

    # Device controls
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--gpu", type=int, default=0, help="GPU id to use if available")

    # Output
    p.add_argument("--output-dir", type=str, default="./dropout_results",
                   help="Base output folder (default: ./dropout_results)")
    return p.parse_args()


def get_device(args: argparse.Namespace) -> torch.device:
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(args.gpu)
    return torch.device(f"cuda:{args.gpu}")


def train_one_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def eval_acc(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args)
    print(f"[Device] {device}")

    train_loader, test_loader, num_classes, dataset_name = get_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    print(f"[Data] {dataset_name} | num_classes={num_classes}")

    model, mc_layers = get_model(args.model, num_classes=num_classes, dropout_p=args.train_dropout_p)
    # Ensure training-time dropout probability is what you requested (typically 0.0)
    set_mc_dropout_p(mc_layers, args.train_dropout_p)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # A light scheduler that behaves well across these datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        te_acc = eval_acc(model, test_loader, device)
        scheduler.step()

        if te_acc > best_test_acc:
            best_test_acc = te_acc

        print(f"[Train] Epoch {epoch:03d}/{args.epochs} | loss={tr_loss:.4f} | acc={tr_acc:.4f} | test_acc={te_acc:.4f}")

    # ===============================
    # Save: dropout_results/<dataset>/<model>.pth
    # ===============================
    save_dir = Path(args.output_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{args.model.lower()}.pth"

    # Save pure state_dict to avoid "unexpected key(s)" issues in downstream scripts
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Saved] {ckpt_path}")

    # Free memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
