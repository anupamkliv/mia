# dropout_inference.py
import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_model, set_dropout_p
from data_utils import get_dataset, make_test_loader, split_dataset_iid, make_dataloaders_for_clients

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


# ---- compatibility helper for resnet34 checkpoints ----
def load_checkpoint_into_model(model, state_dict, model_name: str):
    """
    Load a checkpoint state_dict into the given model.

    Special case:
    - older ResNet-34 FL checkpoints saved with keys starting with
      'features.' and 'fc.' are mapped to the naming expected by
      ResNetWithDropout ('base.*' and 'base.fc.1.*').
    """
    # Only touch the resnet34 + old-style checkpoints
    if model_name == "resnet34" and any(k.startswith("features.") for k in state_dict.keys()):
        prefix_map = {
            "features.0.": "base.conv1.",
            "features.1.": "base.bn1.",
            "features.4.": "base.layer1.",
            "features.5.": "base.layer2.",
            "features.6.": "base.layer3.",
            "features.7.": "base.layer4.",
        }

        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("features."):
                # Map backbone blocks
                mapped = False
                for old_prefix, new_prefix in prefix_map.items():
                    if k.startswith(old_prefix):
                        new_k = new_prefix + k[len(old_prefix):]
                        new_state[new_k] = v
                        mapped = True
                        break
                # If some "features.*" key doesn't match our prefixes,
                # it will simply be skipped – those would correspond
                # to modules without trainable params (ReLU, MaxPool, etc.).
                if not mapped:
                    # No parameters to carry over; safe to ignore
                    pass

            elif k == "fc.weight":
                # Final classifier weight -> base.fc.1.weight
                new_state["base.fc.1.weight"] = v
            elif k == "fc.bias":
                # Final classifier bias -> base.fc.1.bias
                new_state["base.fc.1.bias"] = v
            else:
                # Any other keys (e.g. buffers, running stats under 'base.*'
                # in newer checkpoints) are kept as-is.
                new_state[k] = v

        state_dict = new_state

    # For all other models / newer checkpoints we just load directly
    model.load_state_dict(state_dict)



def get_args():
    parser = argparse.ArgumentParser(description="Dropout stability experiment on federated global model")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "svhn", "tinyimagenet", "flowers102"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet18", "resnet34",
                                 "mobilenetv3_small", "mobilenetv3_large"])
    parser.add_argument("--img-size", type=int, default=224)

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved global model from fed_train.py")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-mc", type=int, default=5,
                        help="Number of stochastic passes per sample")
    parser.add_argument("--dropout-start", type=float, default=0.01)
    parser.add_argument("--dropout-end", type=float, default=0.10)
    parser.add_argument("--dropout-steps", type=int, default=10)

    parser.add_argument("--batch-size", type=int, default=1,
                        help="We use batch_size=1 to strictly follow spec, but can be >1.")
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="0 means use full train set, else limit.")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="0 means use full test set, else limit.")

    parser.add_argument("--results-csv", type=str, default="./cifar100_mobilenetv3_small_dropout_results_fedavg.csv")
    parser.add_argument("--plots-dir", type=str, default="./plots")

    return parser.parse_args()


def build_dropout_list(start: float, end: float, steps: int) -> List[float]:
    return [round(x, 4) for x in np.linspace(start, end, steps).tolist()]


def per_sample_stats(model: torch.nn.Module,
                     loader: DataLoader,
                     device: torch.device,
                     num_mc: int,
                     max_samples: int = 0) -> Dict[str, List[float]]:
    """
    For each sample:
      - run model num_mc times with dropout active
      - compute ACC per sample (mean of 0/1 over MC runs)
      - compute AUC-proxy per sample (mean probability of true class)
    Then return std over samples for both metrics.
    """
    acc_per_sample = []
    aucproxy_per_sample = []

    model.to(device)
    model.train()  # keep dropout ON

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if max_samples > 0 and idx >= max_samples:
                break

            x, y = x.to(device), y.to(device)

            acc_runs = []
            prob_runs = []

            for _ in range(num_mc):
                logits = model(x)
                probs = F.softmax(logits, dim=1)

                preds = probs.argmax(dim=1)
                correct = (preds == y).float().item()
                acc_runs.append(correct)

                # "AUC" proxy: predicted probability of the true class
                true_prob = probs[0, y.item()].item()
                prob_runs.append(true_prob)

            acc_per_sample.append(float(np.mean(acc_runs)))
            aucproxy_per_sample.append(float(np.mean(prob_runs)))

    # standard deviation over samples
    acc_std = float(np.std(acc_per_sample, ddof=1)) if len(acc_per_sample) > 1 else 0.0
    auc_std = float(np.std(aucproxy_per_sample, ddof=1)) if len(aucproxy_per_sample) > 1 else 0.0

    return {
        "acc_std": acc_std,
        "auc_std": auc_std,
    }


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load dataset
    train_ds, test_ds, num_classes = get_dataset(args.dataset, args.data_dir, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Load checkpoint and global model
    # ckpt = torch.load(args.checkpoint, map_location="cpu")
    # state_dict = ckpt["state_dict"]

    # model = create_model(args.model, num_classes=num_classes, dropout_p=0.0)
    # model.load_state_dict(state_dict)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model = create_model(args.model, num_classes, dropout_p=0.0)

    # use compatibility loader so resnet34 FL checkpoints also work
    load_checkpoint_into_model(model, state_dict, args.model)


    # We'll change dropout_p on the fly
    dropout_values = build_dropout_list(args.dropout_start, args.dropout_end, args.dropout_steps)

    # For table
    rows = []

    train_acc_std_list = []
    train_auc_std_list = []
    test_acc_std_list = []
    test_auc_std_list = []

    for p in dropout_values:
        print(f"\n=== Dropout p = {p:.3f} ===")
        set_dropout_p(model, p)

        train_stats = per_sample_stats(
            model, train_loader, device, args.num_mc, max_samples=args.max_train_samples
        )
        test_stats = per_sample_stats(
            model, test_loader, device, args.num_mc, max_samples=args.max_test_samples
        )

        print(f" Train ACC-std: {train_stats['acc_std']:.6f}, AUCproxy-std: {train_stats['auc_std']:.6f}")
        print(f" Test  ACC-std: {test_stats['acc_std']:.6f}, AUCproxy-std: {test_stats['auc_std']:.6f}")

        train_acc_std_list.append(train_stats["acc_std"])
        train_auc_std_list.append(train_stats["auc_std"])
        test_acc_std_list.append(test_stats["acc_std"])
        test_auc_std_list.append(test_stats["auc_std"])

    # Build table as you specified:
    # columns: [dataset, model, split, metric, p1, ..., p10]
    col_names = ["dataset", "model", "split", "metric"] + [f"p={p:.2f}" for p in dropout_values]

    def make_row(split, metric, values):
        return [args.dataset, args.model, split, metric] + list(values)

    rows.append(make_row("train", "ACC", train_acc_std_list))
    rows.append(make_row("test", "ACC", test_acc_std_list))
    rows.append(make_row("train", "AUCproxy", train_auc_std_list))
    rows.append(make_row("test", "AUCproxy", test_auc_std_list))

    df = pd.DataFrame(rows, columns=col_names)
    df.to_csv(args.results_csv, index=False)
    print(f"\nSaved results table to: {args.results_csv}")

    # Plotting
    import matplotlib.pyplot as plt

    # ACC plot
    plt.figure()
    plt.plot(dropout_values, train_acc_std_list, marker="o", label="Train")
    plt.plot(dropout_values, test_acc_std_list, marker="s", label="Test")
    plt.xlabel("Dropout probability")
    plt.ylabel("Std of ACC per sample")
    plt.title(f"{args.dataset} - {args.model} - ACC std vs dropout")
    plt.legend()
    plt.grid(True)
    acc_plot_path = os.path.join(args.plots_dir, f"{args.dataset}_{args.model}_acc_std.png")
    plt.savefig(acc_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ACC std plot to: {acc_plot_path}")

    # AUC-proxy plot
    plt.figure()
    plt.plot(dropout_values, train_auc_std_list, marker="o", label="Train")
    plt.plot(dropout_values, test_auc_std_list, marker="s", label="Test")
    plt.xlabel("Dropout probability")
    plt.ylabel("Std of AUC-proxy per sample")
    plt.title(f"{args.dataset} - {args.model} - AUC-proxy std vs dropout")
    plt.legend()
    plt.grid(True)
    auc_plot_path = os.path.join(args.plots_dir, f"{args.dataset}_{args.model}_aucproxy_std.png")
    plt.savefig(auc_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved AUC-proxy std plot to: {auc_plot_path}")

    # Free GPU
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
