# train.py
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

from data import get_loaders
from models import get_model, set_mc_dropout_p
from utils import set_seed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a single model and save checkpoint as dropout_results/<dataset>/<model>.pth"
    )
    p.add_argument("--dataset", type=str, default="cifar10",
                   help="cifar10 | cifar100 | svhn | tinyimagenet | flowers102")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--model", type=str, default="resnet18",
                   help="resnet18 | resnet34 | mobilenetv3_small | mobilenetv3_large")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-5)

    # IMPORTANT: training dropout (you wanted 0.0 usually)
    p.add_argument("--train-dropout-p", type=float, default=0.0,
                   help="Dropout p during training (0.0 reproduces your setting)")

    p.add_argument("--seed", type=int, default=42)

    # Device controls
    p.add_argument("--cpu", action="store_true", help="Force CPU")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU id to use (e.g. 0,1,2). Ignored if --cpu is set.")

    # Output
    p.add_argument("--output-dir", type=str, default="./dropout_results",
                   help="Base output folder (default: ./dropout_results)")
    return p.parse_args()


def get_device(args) -> torch.device:
    if args.cpu:
        return torch.device("cpu")

    # Make only the selected GPU visible
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.is_available():
        return torch.device("cuda:1")
    return torch.device("cpu")


def train_one_model(model: nn.Module,
                    train_loader,
                    device: torch.device,
                    epochs: int,
                    lr: float,
                    weight_decay: float,
                    train_dropout_p: float,
                    mc_layers):
    # Set dropout p used during training
    set_mc_dropout_p(mc_layers, train_dropout_p)

    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        print(f"[Train] Epoch {epoch+1:03d}/{epochs} | loss={epoch_loss:.4f} | acc={epoch_acc:.4f}")

    return model


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


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args)
    if device.type == "cuda":
        print(f"[Device] cuda (selected GPU id={args.gpu}) | visible name={torch.cuda.get_device_name(0)}")
    else:
        print("[Device] cpu")

    train_loader, test_loader, num_classes, dataset_name = get_loaders(
        args.dataset, args.data_root, args.batch_size, args.num_workers
    )

    # Build model (MC-dropout-ready) with the training dropout p
    model, mc_layers = get_model(args.model, num_classes=num_classes, dropout_p=args.train_dropout_p)

    model = train_one_model(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_dropout_p=args.train_dropout_p,
        mc_layers=mc_layers,
    )

    test_acc = eval_acc(model, test_loader, device)
    print(f"[Eval] test_acc={test_acc:.4f}")

    # ===============================
    # Simple save: dataset/model only
    # dropout_results/<dataset>/<model>.pth
    # ===============================
    save_dir = Path(args.output_dir) / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_dir / f"{args.model}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": args.model,
            "dataset": dataset_name,
            "num_classes": num_classes,
            "train_dropout_p": args.train_dropout_p,
            "test_acc": float(test_acc),
        },
        ckpt_path,
    )

    print(f"[Saved] {ckpt_path}")

    # Free memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
