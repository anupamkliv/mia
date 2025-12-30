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


# ===========================
# Utils
# ===========================

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

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
    """
    Wrap torchvision ResNet so that checkpoints saved as `features.*` can be loaded.

    We convert the ResNet stem+layers+avgpool into a single nn.Sequential called `features`.
    This matches checkpoints that were saved from wrappers using `self.features = nn.Sequential(...)`.
    """
    in_features = base_model.fc.in_features
    # keep base_model parameters but drop the classifier head
    base_model.fc = nn.Identity()

    # Convert to a single sequential module so state_dict keys look like `features.0.*`, `features.4.0.*`, ...
    features = nn.Sequential(*list(base_model.children())[:-1])  # up to avgpool

    class ResNetWithDropout(nn.Module):
        def __init__(self, features, in_features, num_classes, p):
            super().__init__()
            self.features = features
            self.dropout = MonteCarloDropout(p=p)
            self.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            feats = self.features(x)          # [B, C, 1, 1]
            feats = torch.flatten(feats, 1)   # [B, C]
            feats = self.dropout(feats)
            logits = self.fc(feats)
            return logits

    return ResNetWithDropout(features, in_features, num_classes, init_p)



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
            x = self.fc1(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    return MobileNetWithDropout(mnet, last_channel, num_classes, init_p)



# ----------------------------
# Checkpoint loading utilities
# ----------------------------
def _strip_prefix_if_present(state_dict: dict, prefix: str) -> dict:
    """Remove a prefix from all keys if any key starts with it."""
    if not state_dict:
        return state_dict
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _maybe_remap_prefix(state_dict: dict, src_prefix: str, dst_prefix: str) -> dict:
    """If keys are under src_prefix, move them to dst_prefix."""
    if not state_dict:
        return state_dict
    if any(k.startswith(src_prefix) for k in state_dict.keys()):
        return { (dst_prefix + k[len(src_prefix):]) if k.startswith(src_prefix) else k: v
                 for k, v in state_dict.items() }
    return state_dict


def extract_state_dict(ckpt_obj) -> tuple[dict, dict]:
    """
    Supports:
      1) raw state_dict
      2) dict with 'state_dict' / 'model_state_dict' / 'net' / 'model' entries
    Returns: (state_dict, metadata)
    """
    metadata = {}
    if isinstance(ckpt_obj, dict):
        # common formats
        for key in ["state_dict", "model_state_dict", "net", "model", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                metadata = {k: v for k, v in ckpt_obj.items() if k != key}
                return ckpt_obj[key], metadata
        # might already be a state dict (tensor values)
        if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj, {}
    raise ValueError(
        "Unsupported checkpoint format. Expected a state_dict or a dict containing "
        "one of: state_dict/model_state_dict/net/model/weights."
    )


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> dict:
    """
    Loads a checkpoint robustly and returns metadata (if any).
    Handles:
      - checkpoint dict with metadata + nested state_dict
      - DataParallel 'module.' prefix
      - wrapper prefix mismatches like 'base.' vs 'backbone.'
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict, metadata = extract_state_dict(ckpt)

    # normalize common prefixes
    state_dict = _strip_prefix_if_present(state_dict, "module.")

    # handle common wrapper differences
    # If training code saved as base.* but this code defines backbone.* (or vice versa)
    state_dict_try = state_dict
    # try base -> backbone
    state_dict_try = _maybe_remap_prefix(state_dict_try, "base.", "backbone.")
    # try backbone -> base (in case this script uses base.*)
    state_dict_try2 = _maybe_remap_prefix(state_dict, "backbone.", "base.")

    # try features <-> backbone (older wrappers sometimes saved as `features.*`)
    state_dict_try = _maybe_remap_prefix(state_dict_try, "backbone.", "features.")
    state_dict_try2 = _maybe_remap_prefix(state_dict_try2, "features.", "backbone.")

    # attempt load
    missing, unexpected = [], []
    try:
        res = model.load_state_dict(state_dict_try, strict=False)
        missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    except RuntimeError:
        res = model.load_state_dict(state_dict_try2, strict=False)
        missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)

    if unexpected:
        print("[WARN] Unexpected keys while loading:", unexpected[:20], "..." if len(unexpected) > 20 else "")
    if missing:
        print("[WARN] Missing keys while loading:", missing[:20], "..." if len(missing) > 20 else "")

    return metadata

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
# Dataset helpers
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
            transforms.Normalize(
                (0.4377, 0.4438, 0.4728),
                (0.1980, 0.2010, 0.1970),
            ),
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
# Training classifier (shadow models)
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

        loop = tqdm(train_loader, desc=f"Shadow epoch {epoch+1}/{epochs}", leave=False)
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


# ===========================
# White-box feature extraction (Nasr-style)
# ===========================

def get_last_layer(model: nn.Module) -> nn.Linear:
    """
    Return the last linear layer for gradient-based features.
    - ResNetWithDropout has model.fc
    - MobileNetWithDropout has model.fc2
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "fc2") and isinstance(model.fc2, nn.Linear):
        return model.fc2
    raise RuntimeError("Could not locate final linear layer (fc/fc2) in model.")


def single_sample_whitebox_features(model: nn.Module,
                                    logits: torch.Tensor,
                                    label: torch.Tensor,
                                    num_classes: int) -> np.ndarray:
    """
    Compute gradient-based white-box features for a single sample.
    Features:
      [loss, logits(C), probs(C), onehot(C), grad_w_norm, grad_b_norm]
    """
    # logits: (C,), label: scalar
    loss = F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0), reduction="mean")
    model.zero_grad()
    loss.backward(retain_graph=True)

    last_layer = get_last_layer(model)
    # grads shapes:
    # weight: (C, D), bias: (C,)
    g_w = last_layer.weight.grad  # (C, D)
    g_b = last_layer.bias.grad    # (C,)
    grad_w_norm = g_w.norm().item()
    grad_b_norm = g_b.norm().item()

    probs = torch.softmax(logits, dim=0)  # (C,)
    onehot = F.one_hot(label, num_classes=num_classes).float()  # (C,)

    feat = torch.cat([
        loss.detach().view(1),
        logits.detach(),
        probs.detach(),
        onehot.detach(),
        torch.tensor([grad_w_norm, grad_b_norm], device=logits.device)
    ])
    return feat.detach().cpu().numpy()


def estimate_feature_dim(model: nn.Module, num_classes: int, device: torch.device) -> int:
    # This function needs gradients (used by white-box features)
    torch.set_grad_enabled(True)
    """
    Quick dummy forward/backward on fake data to infer feature dimension.
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)  # size doesn't matter much, we just need logits
    logits = model(x)[0]  # (C,)
    label = torch.tensor(0, device=device)
    feat = single_sample_whitebox_features(model, logits, label, num_classes)
    return feat.shape[0]


def collect_whitebox_features(model: nn.Module,
                              loader: DataLoader,
                              device: torch.device,
                              num_classes: int,
                              member_flag: int,
                              max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract gradient-based Nasr-style white-box features for each sample:
      [loss, logits, probs, label_onehot, grad_w_norm, grad_b_norm]
    """
    model.to(device)
    model.eval()  # use running BN stats; gradients still enabled

    features = []
    labels_mem = []

    count = 0
    loop = tqdm(loader, desc=f"Whitebox feats (member={member_flag})", leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        logits_batch = model(images)  # (B, C)

        B = labels.size(0)
        for i in range(B):
            if max_samples is not None and count >= max_samples:
                break

            logits_i = logits_batch[i]   # (C,)
            label_i = labels[i]          # scalar

            # Need fresh graph for per-sample backward: recompute forward for this sample
            model.zero_grad()
            logits_single = model(images[i].unsqueeze(0))[0]
            feat_i = single_sample_whitebox_features(model, logits_single, label_i, num_classes)
            features.append(feat_i)
            labels_mem.append(member_flag)
            count += 1

        if max_samples is not None and count >= max_samples:
            break

    features = np.vstack(features) if len(features) > 0 else np.zeros((0, 1), dtype=np.float32)
    labels_mem = np.array(labels_mem, dtype=np.int64)
    return features, labels_mem


# ===========================
# Attack model
# ===========================

class AttackMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
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
    scores: attack logits, higher => more likely member
    labels: 1 for member, 0 for non-member
    """
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
# Main Nasr-style pipeline
# ===========================

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)
    set_seed(args.seed)

    # 1) Load datasets
    train_ds, test_ds, num_classes, ds_name = get_datasets(args.dataset, args.data_root)
    print(f"Dataset: {ds_name} | Train size: {len(train_ds)} | Test size: {len(test_ds)} | Classes: {num_classes}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 2) Build and load target model
    target_model = build_model(args.model, num_classes=num_classes, pretrained=False)
    metadata = load_model_checkpoint(target_model, args.target_checkpoint, device)
    target_model.to(device)
    target_model.eval()  # avoid BN batch-size=1 issue; gradients still enabled
    print(f"Loaded target checkpoint from {args.target_checkpoint}")

    # Quick feature dimension sanity (optional)
    feat_dim = estimate_feature_dim(target_model, num_classes, device)
    print(f"Estimated white-box feature dimension: {feat_dim}")

    # 3) Shadow models: build attack training set
    shadow_features_all = []
    shadow_labels_all = []

    train_indices = np.arange(len(train_ds))
    test_indices = np.arange(len(test_ds))

    for shadow_idx in range(args.num_shadow_models):
        print(f"\n=== Shadow model {shadow_idx+1}/{args.num_shadow_models} ===")

        # Sample shadow train/test sets
        shadow_train_idx = np.random.choice(train_indices,
                                            size=min(args.shadow_train_size, len(train_ds)),
                                            replace=True)
        shadow_test_idx = np.random.choice(test_indices,
                                           size=min(args.shadow_test_size, len(test_ds)),
                                           replace=True)

        shadow_train_ds = Subset(train_ds, shadow_train_idx)
        shadow_test_ds = Subset(test_ds, shadow_test_idx)

        shadow_train_loader = DataLoader(shadow_train_ds, batch_size=args.shadow_batch_size,
                                         shuffle=True, num_workers=args.num_workers)
        shadow_test_loader = DataLoader(shadow_test_ds, batch_size=args.shadow_batch_size,
                                        shuffle=False, num_workers=args.num_workers)

        # Build and train shadow model
        shadow_model = build_model(args.model, num_classes=num_classes, pretrained=args.shadow_pretrained)
        shadow_model = train_classifier(
            shadow_model,
            shadow_train_loader,
            device,
            epochs=args.shadow_epochs,
            lr=args.shadow_lr,
            weight_decay=args.shadow_weight_decay,
        )

        # Collect white-box features from shadow model
        f_train, m_train = collect_whitebox_features(
            shadow_model, shadow_train_loader, device, num_classes,
            member_flag=1,
            max_samples=args.max_shadow_member_samples
        )
        f_test, m_test = collect_whitebox_features(
            shadow_model, shadow_test_loader, device, num_classes,
            member_flag=0,
            max_samples=args.max_shadow_nonmember_samples
        )

        shadow_features_all.append(np.vstack([f_train, f_test]))
        shadow_labels_all.append(np.concatenate([m_train, m_test]))

    shadow_features_all = np.concatenate(shadow_features_all, axis=0)
    shadow_labels_all = np.concatenate(shadow_labels_all, axis=0)
    print(f"\nAttack training data: {shadow_features_all.shape[0]} samples, dim {shadow_features_all.shape[1]}")

    # 4) Train attack model
    attack_model = train_attack_model(
        shadow_features_all,
        shadow_labels_all,
        input_dim=shadow_features_all.shape[1],
        device=device,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        lr=args.attack_lr,
    )

    # 5) Evaluate on TARGET model (train = members, test = non-members)
    target_train_loader = DataLoader(train_ds, batch_size=args.eval_batch_size, shuffle=False,
                                     num_workers=args.num_workers)
    target_test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    def collect_target_scores(model_target: nn.Module,
                              model_attack: nn.Module,
                              loader: DataLoader,
                              member_flag: int,
                              max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        model_target.to(device)
        model_target.eval()   # avoid BN batch-size=1 issue; gradients still enabled
        model_attack.to(device)
        model_attack.eval()

        scores = []
        labels_m = []
        count = 0

        loop = tqdm(loader, desc=f"Target whitebox (member={member_flag})", leave=False)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            B = labels.size(0)
            for i in range(B):
                if max_samples is not None and count >= max_samples:
                    break

                model_target.zero_grad()
                logits_single = model_target(images[i].unsqueeze(0))[0]
                feat_i = single_sample_whitebox_features(model_target, logits_single, labels[i], num_classes)
                x_i = torch.from_numpy(feat_i).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logit_attack = model_attack(x_i)  # scalar
                scores.append(logit_attack.item())
                labels_m.append(member_flag)
                count += 1

            if max_samples is not None and count >= max_samples:
                break

        scores = np.array(scores, dtype=np.float32)
        labels_m = np.array(labels_m, dtype=np.int64)
        return scores, labels_m

    print("\nCollecting attack scores on target train (members)…")
    scores_train, labels_train = collect_target_scores(
        target_model, attack_model, target_train_loader,
        member_flag=1,
        max_samples=args.max_target_member_samples
    )

    print("Collecting attack scores on target test (non-members)…")
    scores_test, labels_test = collect_target_scores(
        target_model, attack_model, target_test_loader,
        member_flag=0,
        max_samples=args.max_target_nonmember_samples
    )

    scores_all = np.concatenate([scores_train, scores_test], axis=0)
    labels_all = np.concatenate([labels_train, labels_test], axis=0)

    metrics = compute_mia_metrics(scores_all, labels_all)

    print("\n=== Nasr-style White-Box Attack Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    out_path = os.path.join(
        args.output_dir,
        f"{ds_name}_{args.model}_nasr_whitebox_mia.json"
    )
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics to:", out_path)


# ===========================
# Argparse
# ===========================

def parse_args():
    p = argparse.ArgumentParser(description="Nasr-style white-box gradient membership inference attack")

    # Data / target
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

    # Shadow models
    p.add_argument("--num-shadow-models", type=int, default=5,
                   help="Number of shadow models (more => stronger but slower)")
    p.add_argument("--shadow-train-size", type=int, default=5000)
    p.add_argument("--shadow-test-size", type=int, default=5000)
    p.add_argument("--shadow-batch-size", type=int, default=64)
    p.add_argument("--shadow-epochs", type=int, default=5)
    p.add_argument("--shadow-lr", type=float, default=0.1)
    p.add_argument("--shadow-weight-decay", type=float, default=5e-4)
    p.add_argument("--shadow-pretrained", action="store_true")

    p.add_argument("--max-shadow-member-samples", type=int, default=5000,
                   help="Cap on member samples per shadow for attack training")
    p.add_argument("--max-shadow-nonmember-samples", type=int, default=5000,
                   help="Cap on non-member samples per shadow for attack training")

    # Attack model
    p.add_argument("--attack-epochs", type=int, default=10)
    p.add_argument("--attack-batch-size", type=int, default=256)
    p.add_argument("--attack-lr", type=float, default=1e-3)

    # Evaluation on target
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-target-member-samples", type=int, default=5000)
    p.add_argument("--max-target-nonmember-samples", type=int, default=5000)

    # Output
    p.add_argument("--output-dir", type=str, default="./mia_results")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
