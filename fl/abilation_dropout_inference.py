# dropout_inference.py
import argparse
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import create_model, set_dropout_p
from data_utils import get_dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


# ----------------------------
# Args
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Dropout stability experiment on a trained global model")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cifar10", "cifar100", "svhn", "tinyimagenet", "flowers102"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet18", "resnet34", "mobilenetv3_small", "mobilenetv3_large"])
    parser.add_argument("--img-size", type=int, default=224)

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved global model from fed_train.py")

    parser.add_argument("--device", type=str, default="cuda")

    # Ablation A: sweep dropout p
    parser.add_argument("--dropout-start", type=float, default=0.01)
    parser.add_argument("--dropout-end", type=float, default=0.10)
    parser.add_argument("--dropout-steps", type=int, default=10)

    # MC settings
    parser.add_argument("--num-mc", type=int, default=5,
                        help="Default number of stochastic passes per sample (used in Ablation A).")
    parser.add_argument("--mc-list", type=str, default="1,2,5,10,20",
                        help="Comma-separated list for Ablation B, e.g. '1,2,5,10,20'.")

    # Ablation controls
    parser.add_argument("--ablation-mode", type=str, default="both",
                        choices=["A", "B", "both"],
                        help="A: sweep dropout (fixed num-mc). B: sweep num-mc. both: run both.")
    parser.add_argument("--dropout-for-mc", type=float, default=0.05,
                        help="Dropout p used for Ablation B when --ablationB-sweep-all-dropouts is not set.")
    parser.add_argument("--ablationB-sweep-all-dropouts", action="store_true",
                        help="If set, run Ablation B for every dropout p in the A grid (can be slow).")

    # Data sampling
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size. batch_size=1 follows the strict per-sample spec; can be >1.")
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="0 means use full train set, else limit.")
    parser.add_argument("--max-test-samples", type=int, default=0,
                        help="0 means use full test set, else limit.")

    # Outputs
    parser.add_argument("--results-dir", type=str, default="./dropout_results")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag appended to output filenames for easy tracking (e.g. exp name).")

    return parser.parse_args()


# ----------------------------
# Helpers
# ----------------------------
def build_dropout_list(start: float, end: float, steps: int) -> List[float]:
    return [round(x, 4) for x in np.linspace(start, end, steps).tolist()]


def parse_int_list(s: str) -> List[int]:
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    # keep unique, sorted
    items = sorted(list(dict.fromkeys(items)))
    return items


def _std(x: List[float]) -> float:
    # unbiased std if possible
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def per_sample_mc_stats(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_mc: int,
    max_samples: int = 0,
) -> Dict[str, float]:
    """
    For each sample x:
      - run model num_mc times with dropout active
      - compute:
          * mean(ACC over runs) for that sample
          * std(ACC over runs)  for that sample
          * mean(p_true over runs) for that sample
          * std(p_true over runs)  for that sample
    Aggregate over samples:
      - std over samples of per-sample means  (across-sample variability)
      - mean over samples of per-sample stds (within-sample stochasticity)

    Returns:
      {
        "acc_std_of_means", "auc_std_of_means",
        "acc_mean_of_stds", "auc_mean_of_stds",
      }
    """
    acc_mean_per_sample: List[float] = []
    auc_mean_per_sample: List[float] = []
    acc_std_per_sample: List[float] = []
    auc_std_per_sample: List[float] = []

    model.to(device)
    model.train()  # keep dropout ON

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if max_samples > 0 and idx >= max_samples:
                break

            x, y = x.to(device), y.to(device)

            acc_runs: List[float] = []
            prob_runs: List[float] = []

            for _ in range(num_mc):
                logits = model(x)
                probs = F.softmax(logits, dim=1)

                preds = probs.argmax(dim=1)
                correct = (preds == y).float().item()
                acc_runs.append(float(correct))

                # "AUC proxy": predicted probability of the true class
                true_prob = probs[0, y.item()].item()
                prob_runs.append(float(true_prob))

            acc_mean_per_sample.append(float(np.mean(acc_runs)))
            auc_mean_per_sample.append(float(np.mean(prob_runs)))

            acc_std_per_sample.append(_std(acc_runs))
            auc_std_per_sample.append(_std(prob_runs))

    # Across-sample variability of per-sample means
    acc_std_of_means = _std(acc_mean_per_sample)
    auc_std_of_means = _std(auc_mean_per_sample)

    # Within-sample stochasticity (mean std across samples)
    acc_mean_of_stds = float(np.mean(acc_std_per_sample)) if len(acc_std_per_sample) > 0 else 0.0
    auc_mean_of_stds = float(np.mean(auc_std_per_sample)) if len(auc_std_per_sample) > 0 else 0.0

    return {
        "acc_std_of_means": acc_std_of_means,
        "auc_std_of_means": auc_std_of_means,
        "acc_mean_of_stds": acc_mean_of_stds,
        "auc_mean_of_stds": auc_mean_of_stds,
    }


def load_model_and_data(args) -> Tuple[torch.nn.Module, DataLoader, DataLoader, int]:
    train_ds, test_ds, num_classes = get_dataset(args.dataset, args.data_dir, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model = create_model(args.model, num_classes=num_classes, dropout_p=0.0)
    model.load_state_dict(state_dict, strict=True)

    return model, train_loader, test_loader, num_classes


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def out_name(base: str, tag: str) -> str:
    if tag:
        return f"{base}_{tag}"
    return base


# ----------------------------
# Ablation A: sweep dropout p, fixed num_mc
# ----------------------------
def run_ablation_A(args, model, train_loader, test_loader, device, dropout_values: List[float]) -> str:
    rows = []
    # We record both summaries:
    #  (1) std over samples of per-sample means  -> "std_of_means"
    #  (2) mean over samples of per-sample stds -> "mean_of_stds"  (often the more intuitive "stochasticity" metric)
    metrics = [
        ("ACC_std_of_means", "acc_std_of_means"),
        ("AUCproxy_std_of_means", "auc_std_of_means"),
        ("ACC_mean_of_stds", "acc_mean_of_stds"),
        ("AUCproxy_mean_of_stds", "auc_mean_of_stds"),
    ]

    # Collect per metric per split across dropout
    bucket = {("train", m[0]): [] for m in metrics}
    bucket.update({("test", m[0]): [] for m in metrics})

    for p in dropout_values:
        print(f"\n[Ablation A] Dropout p = {p:.3f} | num_mc={args.num_mc}")
        set_dropout_p(model, p)

        train_stats = per_sample_mc_stats(model, train_loader, device, args.num_mc, max_samples=args.max_train_samples)
        test_stats = per_sample_mc_stats(model, test_loader, device, args.num_mc, max_samples=args.max_test_samples)

        for pretty, key in metrics:
            bucket[("train", pretty)].append(train_stats[key])
            bucket[("test", pretty)].append(test_stats[key])

        print(
            f"  Train: ACC(mean_of_stds)={train_stats['acc_mean_of_stds']:.6f}, "
            f"AUC(mean_of_stds)={train_stats['auc_mean_of_stds']:.6f}"
        )
        print(
            f"  Test : ACC(mean_of_stds)={test_stats['acc_mean_of_stds']:.6f}, "
            f"AUC(mean_of_stds)={test_stats['auc_mean_of_stds']:.6f}"
        )

    col_names = ["dataset", "model", "split", "metric"] + [f"p={p:.2f}" for p in dropout_values]

    def make_row(split: str, metric_name: str, values: List[float]):
        return [args.dataset, args.model, split, metric_name] + list(values)

    for (split, metric_name), values in bucket.items():
        rows.append(make_row(split, metric_name, values))

    df = pd.DataFrame(rows, columns=col_names)

    ensure_dir(args.results_dir)
    csv_path = os.path.join(args.results_dir, out_name(f"{args.dataset}_{args.model}_ablationA", args.tag) + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Ablation A table to: {csv_path}")

    # Plots (focus on mean_of_stds, which matches your description: std over MC passes per sample, then average)
    import matplotlib.pyplot as plt

    for metric_pretty, key in [("ACC_mean_of_stds", "acc_mean_of_stds"), ("AUCproxy_mean_of_stds", "auc_mean_of_stds")]:
        y_train = bucket[("train", metric_pretty)]
        y_test = bucket[("test", metric_pretty)]

        plt.figure()
        plt.plot(dropout_values, y_train, marker="o", label="Train")
        plt.plot(dropout_values, y_test, marker="s", label="Test")
        plt.xlabel("Dropout probability p")
        plt.ylabel(metric_pretty)
        plt.title(f"{args.dataset} - {args.model} - {metric_pretty} vs dropout (num_mc={args.num_mc})")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(args.results_dir, out_name(f"{args.dataset}_{args.model}_A_{metric_pretty}", args.tag) + ".png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        print(f"Saved Ablation A plot to: {plot_path}")

    return csv_path


# ----------------------------
# Ablation B: sweep num_mc
# ----------------------------
def run_ablation_B(args, model, train_loader, test_loader, device, dropout_values: List[float]) -> str:
    mc_list = parse_int_list(args.mc_list)
    if len(mc_list) == 0:
        raise ValueError("mc-list parsed to an empty list. Provide e.g. --mc-list '1,2,5,10,20'.")

    if args.ablationB_sweep_all_dropouts:
        ps = dropout_values
    else:
        ps = [float(args.dropout_for_mc)]

    rows = []
    for p in ps:
        set_dropout_p(model, p)
        for T in mc_list:
            print(f"\n[Ablation B] Dropout p={p:.3f} | num_mc={T}")
            train_stats = per_sample_mc_stats(model, train_loader, device, T, max_samples=args.max_train_samples)
            test_stats = per_sample_mc_stats(model, test_loader, device, T, max_samples=args.max_test_samples)

            # Store the "std over MC passes per sample then average" (mean_of_stds) as primary
            rows.append([args.dataset, args.model, "train", p, T, "ACC_mean_of_stds", train_stats["acc_mean_of_stds"]])
            rows.append([args.dataset, args.model, "test", p, T, "ACC_mean_of_stds", test_stats["acc_mean_of_stds"]])
            rows.append([args.dataset, args.model, "train", p, T, "AUCproxy_mean_of_stds", train_stats["auc_mean_of_stds"]])
            rows.append([args.dataset, args.model, "test", p, T, "AUCproxy_mean_of_stds", test_stats["auc_mean_of_stds"]])

            # Optionally also store std_of_means (handy for debugging / paper appendix)
            rows.append([args.dataset, args.model, "train", p, T, "ACC_std_of_means", train_stats["acc_std_of_means"]])
            rows.append([args.dataset, args.model, "test", p, T, "ACC_std_of_means", test_stats["acc_std_of_means"]])
            rows.append([args.dataset, args.model, "train", p, T, "AUCproxy_std_of_means", train_stats["auc_std_of_means"]])
            rows.append([args.dataset, args.model, "test", p, T, "AUCproxy_std_of_means", test_stats["auc_std_of_means"]])

    df = pd.DataFrame(rows, columns=["dataset", "model", "split", "dropout_p", "num_mc", "metric", "value"])

    ensure_dir(args.results_dir)
    csv_path = os.path.join(args.results_dir, out_name(f"{args.dataset}_{args.model}_ablationB", args.tag) + ".csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved Ablation B table to: {csv_path}")

    # Plot only if single dropout p (clean curves)
    if len(ps) == 1:
        import matplotlib.pyplot as plt

        p = ps[0]
        for metric in ["ACC_mean_of_stds", "AUCproxy_mean_of_stds"]:
            sub_train = df[(df["split"] == "train") & (df["metric"] == metric) & (df["dropout_p"] == p)].sort_values("num_mc")
            sub_test = df[(df["split"] == "test") & (df["metric"] == metric) & (df["dropout_p"] == p)].sort_values("num_mc")

            plt.figure()
            plt.plot(sub_train["num_mc"].to_numpy(), sub_train["value"].to_numpy(), marker="o", label="Train")
            plt.plot(sub_test["num_mc"].to_numpy(), sub_test["value"].to_numpy(), marker="s", label="Test")
            plt.xlabel("Number of stochastic passes per sample (T)")
            plt.ylabel(metric)
            plt.title(f"{args.dataset} - {args.model} - {metric} vs T (dropout p={p:.2f})")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(args.results_dir, out_name(f"{args.dataset}_{args.model}_B_{metric}_p{p:.2f}", args.tag) + ".png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved Ablation B plot to: {plot_path}")
    else:
        print("Ablation B plots skipped because multiple dropout_p values were used (enable your own plotting or set a single p).")

    return csv_path


# ----------------------------
# Main
# ----------------------------
def main():
    args = get_args()
    ensure_dir(args.results_dir)

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    model, train_loader, test_loader, _ = load_model_and_data(args)

    dropout_values = build_dropout_list(args.dropout_start, args.dropout_end, args.dropout_steps)

    a_csv = None
    b_csv = None

    if args.ablation_mode in ("A", "both"):
        a_csv = run_ablation_A(args, model, train_loader, test_loader, device, dropout_values)

    if args.ablation_mode in ("B", "both"):
        b_csv = run_ablation_B(args, model, train_loader, test_loader, device, dropout_values)

    # Free GPU
    del model
    torch.cuda.empty_cache()

    if a_csv:
        print(f"\nAblation A done: {a_csv}")
    if b_csv:
        print(f"Ablation B done: {b_csv}")


if __name__ == "__main__":
    main()
