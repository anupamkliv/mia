# fed_train.py
import argparse
import copy
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import create_model
from data_utils import (
    get_dataset,
    split_dataset_iid,
    make_dataloaders_for_clients,
    make_test_loader,
)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


def get_args():
    parser = argparse.ArgumentParser(description="Federated training with dropout-free training models")

    # Data / model
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "svhn", "tinyimagenet", "flowers102"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34",
                                 "mobilenetv3_small", "mobilenetv3_large"])
    parser.add_argument("--img-size", type=int, default=224)

    # Federated
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--fed-algo", type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "fedavgm"])
    parser.add_argument("--fedprox-mu", type=float, default=0.001)
    parser.add_argument("--server-lr", type=float, default=1.0)  # for fedavgm

    # Optimization
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    # Misc
    parser.add_argument("--device", type=str, default="cuda:3",
                        help="cuda or cpu")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)

    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total if total > 0 else 0.0


def local_train(model: nn.Module,
                global_state: Dict[str, torch.Tensor],
                loader: DataLoader,
                device: torch.device,
                args) -> nn.Module:
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for epoch in range(args.local_epochs):
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            if args.fed_algo == "fedprox":
                prox_term = 0.0
                with torch.no_grad():
                    global_params = {k: v.to(device) for k, v in global_state.items()}
                for (name, param) in model.named_parameters():
                    if name in global_params:
                        prox_term = prox_term + ((param - global_params[name]) ** 2).sum()
                loss = loss + (args.fedprox_mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

    return model


def average_weights(client_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg_state = copy.deepcopy(client_states[0])
    num_clients = len(client_states)

    for k in avg_state.keys():
        # sum over all clients
        for i in range(1, num_clients):
            avg_state[k] += client_states[i][k]

        # handle floats vs integers separately
        if avg_state[k].is_floating_point():
            avg_state[k] /= float(num_clients)
        else:
            # for counters / indices, integer average (or you could just keep client 0)
            avg_state[k] //= num_clients

    return avg_state


def fedavgm_update(global_state: Dict[str, torch.Tensor],
                   avg_state: Dict[str, torch.Tensor],
                   momentum_buffer: Dict[str, torch.Tensor],
                   server_lr: float,
                   beta: float = 0.9):
    """FedAvgM: server momentum over parameter deltas."""
    if momentum_buffer is None:
        momentum_buffer = {}
        for k in global_state.keys():
            if global_state[k].is_floating_point():
                momentum_buffer[k] = torch.zeros_like(global_state[k])

    new_global_state = {}
    for k in global_state.keys():
        # only apply momentum / server lr to floating-point params
        if global_state[k].is_floating_point():
            delta = avg_state[k] - global_state[k]
            momentum_buffer[k] = beta * momentum_buffer[k] + delta
            new_global_state[k] = global_state[k] + server_lr * momentum_buffer[k]
        else:
            # e.g. num_batches_tracked or other integer buffers: just copy
            new_global_state[k] = avg_state[k]

    return new_global_state, momentum_buffer



def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset
    train_ds, test_ds, num_classes = get_dataset(args.dataset, args.data_dir, img_size=args.img_size)
    client_subsets = split_dataset_iid(train_ds, args.num_clients)
    client_loaders = make_dataloaders_for_clients(client_subsets, batch_size=args.batch_size)
    test_loader = make_test_loader(test_ds, batch_size=args.test_batch_size)

    # 2. Global model, trained with dropout_p=0 (as requested)
    global_model = create_model(args.model, num_classes=num_classes, dropout_p=0.0).to(device)
    global_state = global_model.state_dict()

    momentum_buffer = None  # for fedavgm

    # 3. Federated rounds
    for rnd in range(1, args.rounds + 1):
        print(f"\n=== Round {rnd}/{args.rounds} ===")

        client_states = []

        for cid, loader in enumerate(client_loaders):
            print(f" Client {cid + 1}/{args.num_clients}")

            # fresh client model from current global
            client_model = create_model(args.model, num_classes=num_classes, dropout_p=0.0)
            client_model.load_state_dict(global_state)

            client_model = local_train(client_model, global_state, loader, device, args)
            client_states.append(copy.deepcopy(client_model.cpu().state_dict()))

            # free GPU
            del client_model
            torch.cuda.empty_cache()

        # 4. Aggregate
        avg_state = average_weights(client_states)

        if args.fed_algo == "fedavg":
            global_state = avg_state
        elif args.fed_algo == "fedprox":
            global_state = avg_state
        elif args.fed_algo == "fedavgm":
            global_state, momentum_buffer = fedavgm_update(
                global_state, avg_state, momentum_buffer, server_lr=args.server_lr
            )
        else:
            raise ValueError(f"Unknown fed_algo: {args.fed_algo}")

        global_model.load_state_dict(global_state)
        acc = evaluate(global_model.to(device), test_loader, device)
        print(f" Global test ACC after round {rnd}: {acc:.4f}")

        global_model.cpu()
        torch.cuda.empty_cache()

    # 5. Save final global model
    ckpt_name = f"{args.dataset}_{args.model}_{args.fed_algo}_global.pth"
    ckpt_path = os.path.join(args.save_dir, ckpt_name)
    torch.save({
        "state_dict": global_state,
        "dataset": args.dataset,
        "model": args.model,
        "num_classes": num_classes,
        "fed_algo": args.fed_algo,
        "rounds": args.rounds,
    }, ckpt_path)
    print(f"\nSaved global model to: {ckpt_path}")


if __name__ == "__main__":
    main()
