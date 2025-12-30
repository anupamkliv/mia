import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import get_loaders
from models import get_model, set_mc_dropout_p
from utils import get_device, load_checkpoint, ensure_dir, to_serializable, save_json


@torch.no_grad()
def mc_dropout_scores(model,
                      data_loader,
                      device: torch.device,
                      mc_layers,
                      dropout_p: float,
                      num_mc_passes: int) -> Dict[str, np.ndarray]:
    """
    For each sample, perform `num_mc_passes` stochastic passes and compute:
      - mean max-probability (confidence)
      - variance of max-probability across passes (stability proxy)
      - predicted label majority vote (optional)
    Returns arrays aligned with dataset order.
    """
    set_mc_dropout_p(mc_layers, dropout_p)

    model.to(device)
    model.eval()
    # IMPORTANT: keep MC dropout active
    for m in mc_layers:
        m.train()

    maxprob_means = []
    maxprob_vars = []

    for images, _labels in data_loader:
        images = images.to(device, non_blocking=True)

        # (T, B, C)
        probs_T = []
        for _ in range(num_mc_passes):
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            maxprob = probs.max(dim=1).values  # (B,)
            probs_T.append(maxprob.detach().cpu().numpy())

        probs_T = np.stack(probs_T, axis=0)  # (T, B)
        maxprob_means.append(probs_T.mean(axis=0))
        maxprob_vars.append(probs_T.var(axis=0))

    return {
        "maxprob_mean": np.concatenate(maxprob_means, axis=0),
        "maxprob_var": np.concatenate(maxprob_vars, axis=0),
    }


def parse_args():
    p = argparse.ArgumentParser(description="MC-dropout inference on a trained checkpoint")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override dataset (otherwise read from checkpoint meta if available)")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dropout-values", nargs="+", type=float,
                   default=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    p.add_argument("--num-mc-passes", type=int, default=5)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory. Default: alongside checkpoint in 'mc_inference'")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.cpu)
    ckpt_path = Path(args.checkpoint)

    ckpt = load_checkpoint(ckpt_path, map_location=device)

    arch = ckpt.get("arch", None)
    dataset_name = ckpt.get("dataset", None)
    num_classes = ckpt.get("num_classes", None)

    if args.dataset is not None:
        dataset_name = args.dataset

    if arch is None or dataset_name is None or num_classes is None:
        raise ValueError("Checkpoint is missing required metadata (arch/dataset/num_classes). "
                         "Re-train using train.py or add these fields.")

    # loaders
    train_loader, test_loader, _nc, dataset_name2 = get_loaders(
        dataset_name, args.data_root, args.batch_size, args.num_workers
    )
    if _nc != int(num_classes):
        raise ValueError(f"num_classes mismatch: checkpoint={num_classes} loaders={_nc}")

    # model
    model, mc_layers = get_model(arch, num_classes=int(num_classes), dropout_p=0.0)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # output dir
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "mc_inference"
    ensure_dir(out_dir)

    rows = []
    for p in args.dropout_values:
        print(f"[MC] dropout_p={p} passes={args.num_mc_passes}")

        train_scores = mc_dropout_scores(model, train_loader, device, mc_layers, dropout_p=p, num_mc_passes=args.num_mc_passes)
        test_scores = mc_dropout_scores(model, test_loader, device, mc_layers, dropout_p=p, num_mc_passes=args.num_mc_passes)

        # summary stats you can use as attack signals
        rows.append({
            "dropout_p": p,
            "train_maxprob_mean_mean": float(train_scores["maxprob_mean"].mean()),
            "train_maxprob_var_mean": float(train_scores["maxprob_var"].mean()),
            "test_maxprob_mean_mean": float(test_scores["maxprob_mean"].mean()),
            "test_maxprob_var_mean": float(test_scores["maxprob_var"].mean()),
        })

        # save per-sample arrays (optional but useful for ROC later)
        np.save(out_dir / f"train_scores_p{p:.3f}.npy", train_scores)
        np.save(out_dir / f"test_scores_p{p:.3f}.npy", test_scores)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "mc_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    # simple plots
    plt.figure()
    plt.plot(df["dropout_p"], df["train_maxprob_var_mean"], marker="o", label="train")
    plt.plot(df["dropout_p"], df["test_maxprob_var_mean"], marker="o", label="test")
    plt.xlabel("dropout p (inference)")
    plt.ylabel("mean var of max-prob across passes")
    plt.title(f"MC-dropout stability: {dataset_name} / {arch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "stability_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["dropout_p"], df["train_maxprob_mean_mean"], marker="o", label="train")
    plt.plot(df["dropout_p"], df["test_maxprob_mean_mean"], marker="o", label="test")
    plt.xlabel("dropout p (inference)")
    plt.ylabel("mean max-prob across passes")
    plt.title(f"Confidence: {dataset_name} / {arch}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "confidence_curve.png", dpi=200)
    plt.close()

    # save run args
    save_json(to_serializable(vars(args)), out_dir / "run_args.json")

    # free memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
