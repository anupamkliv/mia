import argparse
import os
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from tqdm import tqdm
import json
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


# ===========================
# Utils
# ===========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MonteCarloDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


def add_head_dropout_for_resnet(base_model: nn.Module,
                                num_classes: int,
                                init_p: float = 0.0):
    in_features = base_model.fc.in_features
    base_model.fc = nn.Identity()

    class ResNetWithDropout(nn.Module):
        def __init__(self, backbone, in_features, num_classes, p):
            super().__init__()
            self.backbone = backbone
            self.dropout = MonteCarloDropout(p=p)
            self.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            feats = self.backbone(x)
            feats = self.dropout(feats)
            logits = self.fc(feats)
            return logits

    return ResNetWithDropout(base_model, in_features, num_classes, init_p)


def add_head_dropout_for_mobilenet(mnet: nn.Module,
                                   num_classes: int,
                                   init_p: float = 0.0):
    last_channel = mnet.classifier[0].in_features

    class MobileNetWithDropout(nn.Module):
        def __init__(self, backbone, last_channel, num_classes, p):
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(last_channel, 1024)
            self.act = nn.Hardswish()
            self.dropout = MonteCarloDropout(p=p)
            self.fc2 = nn.Linear(1024, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc1(self.act(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    return MobileNetWithDropout(mnet, last_channel, num_classes, init_p)


def build_model(model_name: str,
                num_classes: int,
                pretrained: bool = False) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model = add_head_dropout_for_resnet(base, num_classes)
    elif model_name == "resnet34":
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model = add_head_dropout_for_resnet(base, num_classes)
    elif model_name in ["mobilenetv3_small", "mobilenetv3-small"]:
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model = add_head_dropout_for_mobilenet(base, num_classes)
    elif model_name in ["mobilenetv3_large", "mobilenetv3-large"]:
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model = add_head_dropout_for_mobilenet(base, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# ===========================
# Dataset loaders (dataset objects + info)
# ===========================

def get_datasets(dataset: str, data_root: str):
    dataset = dataset.lower()

    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])

        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
        num_classes = 10
        name = "CIFAR10"

    elif dataset == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ])

        train_ds = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_test)
        num_classes = 100
        name = "CIFAR100"

    elif dataset == "svhn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970)),
        ])
        train_ds = datasets.SVHN(root=data_root, split="train", download=True, transform=transform)
        test_ds = datasets.SVHN(root=data_root, split="test", download=True, transform=transform)
        num_classes = 10
        name = "SVHN"

    elif dataset in ["tinyimagenet", "tiny-imagenet"]:
        root = os.path.join(data_root, "tiny-imagenet-200")
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=transform)
        test_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=transform)
        num_classes = len(train_ds.classes)
        name = "TinyImageNet"

    elif dataset in ["flowers102", "flower102"]:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])
        train_ds = datasets.Flowers102(root=data_root, split="train", download=True, transform=transform)
        test_ds = datasets.Flowers102(root=data_root, split="test", download=True, transform=transform)
        num_classes = 102
        name = "Flowers102"
    else:
        raise ValueError("Unsupported dataset")

    return train_ds, test_ds, num_classes, name


# ===========================
# Training helpers
# ===========================

def train_classifier(model: nn.Module,
                     train_loader: DataLoader,
                     device: torch.device,
                     epochs: int,
                     lr: float,
                     weight_decay: float):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Shadow train epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"[Shadow] Epoch {epoch+1}/{epochs} - Loss {epoch_loss:.4f} - Acc {epoch_acc:.4f}")
    return model


@torch.no_grad()
def collect_attack_features(model: nn.Module,
                            loader: DataLoader,
                            device: torch.device,
                            num_classes: int,
                            member_flag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect features (softmax probs + label one-hot) and membership labels (1/0)
    for all samples in loader.
    """
    model.eval()
    feats = []
    memb = []

    loop = tqdm(loader, desc=f"Collecting attack features (member={member_flag})", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)  # (B, C)
        labels_oh = F.one_hot(labels, num_classes=num_classes).float()  # (B, C)

        f = torch.cat([probs, labels_oh], dim=1)  # (B, 2C)
        feats.append(f.cpu().numpy())

        m = np.full((labels.size(0),), member_flag, dtype=np.int64)
        memb.append(m)

    feats = np.concatenate(feats, axis=0)
    memb = np.concatenate(memb, axis=0)
    return feats, memb


# ===========================
# Attack model (MLP)
# ===========================

class AttackMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_attack_model(features: np.ndarray,
                       labels: np.ndarray,
                       input_dim: int,
                       device: torch.device,
                       epochs: int,
                       batch_size: int,
                       lr: float) -> AttackMLP:
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AttackMLP(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(loader, desc=f"Attack train epoch {epoch+1}/{epochs}", leave=False)
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(dataset)
        print(f"[Attack] Epoch {epoch+1}/{epochs} - Loss {epoch_loss:.4f}")

    return model


# ===========================
# Metrics
# ===========================

def compute_mia_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    scores: higher => more likely member
    labels: 1 for member, 0 for non-member
    """
    # Threshold at 0.5 for hard labels (after sigmoid)
    probs = 1 / (1 + np.exp(-scores))
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="binary", zero_division=0
    )

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float("nan")

    fpr, tpr, _ = roc_curve(labels, scores)

    def tpr_at(fpr_limit: float) -> float:
        mask = fpr <= fpr_limit
        if np.any(mask):
            return float(tpr[mask].max())
        return 0.0

    tpr_1 = tpr_at(0.01)
    tpr_01 = tpr_at(0.001)

    # Membership advantage
    p_member = probs[labels == 1]
    p_nonmember = probs[labels == 0]
    advantage = float(abs((p_member >= 0.5).mean() - (p_nonmember >= 0.5).mean()))

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "tpr_at_1pct_fpr": tpr_1,
        "tpr_at_0_1pct_fpr": tpr_01,
        "advantage": advantage,
    }


# ===========================
# Main Shokri-style pipeline
# ===========================

def run_experiment(args):
    device = torch.device("cuda:3" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)
    set_seed(args.seed)

    # 1) Load datasets
    train_ds, test_ds, num_classes, ds_name = get_datasets(args.dataset, args.data_root)
    print(f"Dataset: {ds_name} | Train size: {len(train_ds)} | Test size: {len(test_ds)} | Classes: {num_classes}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2) Build and load target model
    target_model = build_model(args.model, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.target_checkpoint, map_location=device)
    target_model.load_state_dict(ckpt)
    target_model.to(device)
    target_model.eval()
    print(f"Loaded target checkpoint from {args.target_checkpoint}")

    # 3) TRAIN SHADOW MODELS + BUILD ATTACK TRAINING DATA
    shadow_features = []
    shadow_membership = []

    train_indices = np.arange(len(train_ds))
    test_indices = np.arange(len(test_ds))

    for shadow_idx in range(args.num_shadow_models):
        print(f"\n=== Shadow model {shadow_idx+1}/{args.num_shadow_models} ===")

        # sample indices with replacement to mimic unknown data distribution
        shadow_train_idx = np.random.choice(train_indices, size=min(args.shadow_train_size, len(train_ds)), replace=True)
        shadow_test_idx = np.random.choice(test_indices, size=min(args.shadow_test_size, len(test_ds)), replace=True)

        shadow_train_ds = Subset(train_ds, shadow_train_idx)
        shadow_test_ds = Subset(test_ds, shadow_test_idx)

        shadow_train_loader = DataLoader(shadow_train_ds, batch_size=args.shadow_batch_size, shuffle=True,
                                         num_workers=args.num_workers)
        shadow_test_loader = DataLoader(shadow_test_ds, batch_size=args.shadow_batch_size, shuffle=False,
                                        num_workers=args.num_workers)

        # build and train shadow model
        shadow_model = build_model(args.model, num_classes=num_classes, pretrained=args.shadow_pretrained)
        shadow_model = train_classifier(
            shadow_model,
            shadow_train_loader,
            device,
            epochs=args.shadow_epochs,
            lr=args.shadow_lr,
            weight_decay=args.shadow_weight_decay,
        )

        # collect attack features from shadow model
        feat_train, memb_train = collect_attack_features(
            shadow_model, shadow_train_loader, device, num_classes, member_flag=1
        )
        feat_test, memb_test = collect_attack_features(
            shadow_model, shadow_test_loader, device, num_classes, member_flag=0
        )

        shadow_features.append(np.vstack([feat_train, feat_test]))
        shadow_membership.append(np.concatenate([memb_train, memb_test]))

    shadow_features = np.concatenate(shadow_features, axis=0)
    shadow_membership = np.concatenate(shadow_membership, axis=0)

    print(f"\nAttack training data size: {shadow_features.shape[0]} samples, feature dim {shadow_features.shape[1]}")

    # 4) Train attack model
    attack_model = train_attack_model(
        shadow_features,
        shadow_membership,
        input_dim=shadow_features.shape[1],
        device=device,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        lr=args.attack_lr,
    )

    # 5) Evaluate attack on TARGET model (train = members, test = non-members)
    target_train_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False,
                                     num_workers=args.num_workers)
    target_test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    @torch.no_grad()
    def collect_target_scores(model_target, model_attack, loader, member_flag):
        model_target.eval()
        model_attack.eval()
        feats_all = []
        loop = tqdm(loader, desc=f"Target features (member={member_flag})", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            logits = model_target(images)
            probs = torch.softmax(logits, dim=1)
            labels_oh = F.one_hot(labels, num_classes=num_classes).float()

            feats = torch.cat([probs, labels_oh], dim=1)
            with torch.no_grad():
                scores = model_attack(feats.to(device))

            feats_all.append(scores.cpu().numpy())
        scores_all = np.concatenate(feats_all, axis=0)
        labels_all = np.full((scores_all.shape[0],), member_flag, dtype=np.int64)
        return scores_all, labels_all

    print("\nCollecting attack scores on target model...")
    scores_train, labels_train = collect_target_scores(target_model, attack_model, target_train_loader, member_flag=1)
    scores_test, labels_test = collect_target_scores(target_model, attack_model, target_test_loader, member_flag=0)

    scores = np.concatenate([scores_train, scores_test], axis=0)
    labels_mia = np.concatenate([labels_train, labels_test], axis=0)

    metrics = compute_mia_metrics(scores, labels_mia)

    print("\n=== Shokri Shadow Model Attack Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # save
    out_path = os.path.join(
        args.output_dir,
        f"{ds_name}_{args.model}_shokri_shadow_mia.json"
    )
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics to:", out_path)


# ===========================
# Argparse
# ===========================

def parse_args():
    p = argparse.ArgumentParser(description="Shokri et al. Shadow-Model Membership Inference Attack")

    # data / target
    p.add_argument("--dataset", type=str, required=True,
                   help="cifar10 | cifar100 | svhn | tinyimagenet | flowers102")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--model", type=str, required=True,
                   help="resnet18 | resnet34 | mobilenetv3_small | mobilenetv3_large")
    p.add_argument("--target-checkpoint", type=str, required=True,
                   help="Checkpoint of the trained TARGET model (.pth)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    # shadow model config
    p.add_argument("--num-shadow-models", type=int, default=4)
    p.add_argument("--shadow-train-size", type=int, default=20000,
                   help="Number of train samples per shadow model (sampled with replacement)")
    p.add_argument("--shadow-test-size", type=int, default=20000,
                   help="Number of test samples per shadow model (sampled with replacement)")
    p.add_argument("--shadow-batch-size", type=int, default=128)
    p.add_argument("--shadow-epochs", type=int, default=5)
    p.add_argument("--shadow-lr", type=float, default=0.1)
    p.add_argument("--shadow-weight-decay", type=float, default=5e-4)
    p.add_argument("--shadow-pretrained", action="store_true",
                   help="Use pretrained backbone for shadow models (optional, usually False)")

    # attack model config
    p.add_argument("--attack-epochs", type=int, default=10)
    p.add_argument("--attack-batch-size", type=int, default=256)
    p.add_argument("--attack-lr", type=float, default=1e-3)

    # evaluation on target
    p.add_argument("--eval-batch-size", type=int, default=256)

    # output
    p.add_argument("--output-dir", type=str, default="./mia_results")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
