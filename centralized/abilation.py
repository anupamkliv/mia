#!/usr/bin/env python3
"""
abilation.py (repo-adaptive + tuple-unwrapping + progress)

Adds:
- tqdm progress bars for:
  - sweeping p / T
  - iterating over samples in train/test

Outputs:
- CSVs + plots (one per ablation metric)
"""

import argparse
import os
import inspect
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

import models
from data_utils import get_dataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


# ----------------------------
# Checkpoint helpers
# ----------------------------
def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError(
        "Could not extract a state_dict from checkpoint. "
        "Expected a state_dict or dict with keys like state_dict/model_state_dict."
    )


# ----------------------------
# Unwrap model from tuple/list/dict wrappers
# ----------------------------
def unwrap_model(obj: Any) -> nn.Module:
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            try:
                return unwrap_model(item)
            except Exception:
                pass
    if isinstance(obj, dict):
        for v in obj.values():
            try:
                return unwrap_model(v)
            except Exception:
                pass
    raise TypeError(
        f"Model factory returned type={type(obj)} but no nn.Module could be found inside."
    )


# ----------------------------
# Repo-adaptive model builder
# ----------------------------
def _try_call_factory(factory, model_name: str, num_classes: int, dropout_p: float):
    try:
        sig = inspect.signature(factory)
    except Exception:
        sig = None

    kw_variants = [
        {"model_name": model_name, "num_classes": num_classes, "dropout_p": dropout_p},
        {"model": model_name, "num_classes": num_classes, "dropout_p": dropout_p},
        {"arch": model_name, "num_classes": num_classes, "dropout_p": dropout_p},
        {"name": model_name, "num_classes": num_classes, "dropout_p": dropout_p},
        {"model_name": model_name, "num_classes": num_classes},
        {"model": model_name, "num_classes": num_classes},
        {"arch": model_name, "num_classes": num_classes},
        {"name": model_name, "num_classes": num_classes},
        {"model_name": model_name},
        {"model": model_name},
        {"arch": model_name},
        {"name": model_name},
        {"num_classes": num_classes, "dropout_p": dropout_p},
        {"num_classes": num_classes},
    ]
    for kwargs in kw_variants:
        try:
            if sig is None:
                return factory(**kwargs)
            accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
            if accepted:
                return factory(**accepted)
        except TypeError:
            continue
        except Exception:
            continue

    pos_variants = [
        (model_name, num_classes, dropout_p),
        (model_name, num_classes),
        (model_name,),
        (num_classes, dropout_p),
        (num_classes,),
        tuple(),
    ]
    for args in pos_variants:
        try:
            return factory(*args)
        except TypeError:
            continue
        except Exception:
            continue

    raise RuntimeError("Factory exists but could not be called with any known signature.")


def build_model_from_repo(model_name: str, num_classes: int, dropout_p: float) -> nn.Module:
    factory_names = [
        "create_model",
        "get_model",
        "build_model",
        "make_model",
        "get_network",
        "get_net",
        "model_factory",
    ]
    for fn_name in factory_names:
        if hasattr(models, fn_name) and callable(getattr(models, fn_name)):
            out = _try_call_factory(getattr(models, fn_name), model_name, num_classes, dropout_p)
            m = unwrap_model(out)
            print(f"[Model] Built using models.{fn_name}(...)")
            return m
    raise RuntimeError("No known model factory found in models.py (create_model/get_model/...).")


# ----------------------------
# Dropout setter (repo-adaptive)
# ----------------------------
def set_dropout_probability(model: nn.Module, p: float) -> None:
    setter_names = [
        "set_dropout_p",
        "set_mc_dropout_p",
        "set_mc_dropout",
        "set_dropout",
        "configure_dropout",
    ]
    for fn_name in setter_names:
        if hasattr(models, fn_name) and callable(getattr(models, fn_name)):
            fn = getattr(models, fn_name)
            try:
                try:
                    fn(model, p)
                except TypeError:
                    fn(p, model)
                return
            except Exception:
                pass

    # fallback: update Dropout modules
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = float(p)


# ----------------------------
# Stats computation (aligned)
# ----------------------------
def _std(x: List[float]) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


@torch.no_grad()
def per_sample_mc_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_mc: int,
    max_samples: int = 0,
    desc: str = "",
) -> Dict[str, float]:
    acc_mean_per_sample: List[float] = []
    auc_mean_per_sample: List[float] = []
    acc_std_per_sample: List[float] = []
    auc_std_per_sample: List[float] = []

    model.to(device)
    model.train()  # dropout ON

    # progress length: either full loader len, or max_samples (if provided)
    total = None
    if max_samples and max_samples > 0:
        total = max_samples
    elif hasattr(loader, "__len__"):
        try:
            total = len(loader) * loader.batch_size
        except Exception:
            total = None

    seen = 0
    pbar = tqdm(total=total, desc=desc, leave=False, dynamic_ncols=True)

    for x, y in loader:
        if max_samples > 0 and seen >= max_samples:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        acc_runs: List[float] = []
        prob_runs: List[float] = []

        for _ in range(num_mc):
            logits = model(x)
            probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)
            correct = (preds == y).float().mean().item()
            acc_runs.append(float(correct))

            true_probs = probs[torch.arange(probs.size(0), device=device), y]
            prob_runs.append(float(true_probs.mean().item()))

        acc_mean_per_sample.append(float(np.mean(acc_runs)))
        auc_mean_per_sample.append(float(np.mean(prob_runs)))
        acc_std_per_sample.append(_std(acc_runs))
        auc_std_per_sample.append(_std(prob_runs))

        seen += x.size(0)
        pbar.update(x.size(0))

    pbar.close()

    ACC_std_of_means = float(np.std(acc_mean_per_sample, ddof=1)) if len(acc_mean_per_sample) > 1 else 0.0
    AUCproxy_std_of_means = float(np.std(auc_mean_per_sample, ddof=1)) if len(auc_mean_per_sample) > 1 else 0.0
    ACC_mean_of_stds = float(np.mean(acc_std_per_sample)) if len(acc_std_per_sample) > 0 else 0.0
    AUCproxy_mean_of_stds = float(np.mean(auc_std_per_sample)) if len(auc_std_per_sample) > 0 else 0.0

    return {
        "ACC_std_of_means": ACC_std_of_means,
        "AUCproxy_std_of_means": AUCproxy_std_of_means,
        "ACC_mean_of_stds": ACC_mean_of_stds,
        "AUCproxy_mean_of_stds": AUCproxy_mean_of_stds,
    }


# ----------------------------
# Parsing utils
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if ":" in s:
        a, b, step = [float(x) for x in s.split(":")]
        vals, v = [], a
        while v <= b + 1e-12:
            vals.append(round(v, 10))
            v += step
        return vals
    return [float(x) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if ":" in s:
        a, b, step = [int(x) for x in s.split(":")]
        return list(range(a, b + 1, step))
    return [int(x) for x in s.split(",") if x.strip()]


# ----------------------------
# Ablations
# ----------------------------
def run_ablation_A(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    dropout_values: List[float],
    num_mc: int,
    max_train_samples: int,
    max_test_samples: int,
) -> pd.DataFrame:
    rows = []
    outer = tqdm(dropout_values, desc=f"[A] sweep p (T={num_mc})", dynamic_ncols=True)
    for p in outer:
        set_dropout_probability(model, p)

        tr = per_sample_mc_stats(
            model, train_loader, device, num_mc=num_mc, max_samples=max_train_samples,
            desc=f"train p={p:.3f} T={num_mc}"
        )
        te = per_sample_mc_stats(
            model, test_loader, device, num_mc=num_mc, max_samples=max_test_samples,
            desc=f"test  p={p:.3f} T={num_mc}"
        )

        rows.append({"split": "train", "p": p, "T": num_mc, **tr})
        rows.append({"split": "test", "p": p, "T": num_mc, **te})

        outer.set_postfix({
            "p": f"{p:.3f}",
            "train_ACC_mos": f"{tr['ACC_mean_of_stds']:.4g}",
            "test_ACC_mos": f"{te['ACC_mean_of_stds']:.4g}",
        })

    return pd.DataFrame(rows)


def run_ablation_B(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    p_fixed: float,
    T_values: List[int],
    max_train_samples: int,
    max_test_samples: int,
) -> pd.DataFrame:
    rows = []
    set_dropout_probability(model, p_fixed)

    outer = tqdm(T_values, desc=f"[B] sweep T (p={p_fixed})", dynamic_ncols=True)
    for T in outer:
        tr = per_sample_mc_stats(
            model, train_loader, device, num_mc=T, max_samples=max_train_samples,
            desc=f"train p={p_fixed:.3f} T={T}"
        )
        te = per_sample_mc_stats(
            model, test_loader, device, num_mc=T, max_samples=max_test_samples,
            desc=f"test  p={p_fixed:.3f} T={T}"
        )

        rows.append({"split": "train", "p_fixed": p_fixed, "T": T, **tr})
        rows.append({"split": "test", "p_fixed": p_fixed, "T": T, **te})

        outer.set_postfix({
            "T": T,
            "train_ACC_mos": f"{tr['ACC_mean_of_stds']:.4g}",
            "test_ACC_mos": f"{te['ACC_mean_of_stds']:.4g}",
        })

    return pd.DataFrame(rows)


# ----------------------------
# Plotting
# ----------------------------
def plot_ablation_A(df: pd.DataFrame, out_dir: str, dataset: str, model_name: str, num_mc: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    for metric, tag in [
        ("ACC_mean_of_stds", "ACC_mean_of_stds"),
        ("AUCproxy_mean_of_stds", "AUCproxy_mean_of_stds"),
    ]:
        plt.figure()
        for split, marker in [("train", "o"), ("test", "s")]:
            sub = df[df["split"] == split].sort_values("p")
            plt.plot(sub["p"], sub[metric], marker=marker, label=split.capitalize())
        plt.xlabel("Dropout probability p")
        plt.ylabel(tag)
        plt.title(f"{dataset} - {model_name} - {tag} vs dropout (num_mc={num_mc})")
        plt.grid(True, alpha=0.4)
        plt.legend()
        out_path = os.path.join(out_dir, f"{dataset}_{model_name}_A_{tag}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_path}")


def plot_ablation_B(df: pd.DataFrame, out_dir: str, dataset: str, model_name: str, p_fixed: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    for metric, tag in [
        ("ACC_mean_of_stds", "ACC_mean_of_stds"),
        ("AUCproxy_mean_of_stds", "AUCproxy_mean_of_stds"),
    ]:
        plt.figure()
        for split, marker in [("train", "o"), ("test", "s")]:
            sub = df[df["split"] == split].sort_values("T")
            plt.plot(sub["T"], sub[metric], marker=marker, label=split.capitalize())
        plt.xlabel("Number of stochastic passes per sample (T)")
        plt.ylabel(tag)
        plt.title(f"{dataset} - {model_name} - {tag} vs T (dropout p={p_fixed})")
        plt.grid(True, alpha=0.4)
        plt.legend()
        out_path = os.path.join(out_dir, f"{dataset}_{model_name}_B_{tag}_p{p_fixed}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {out_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)

    ap.add_argument("--dropout_values", type=str, default="0.01:0.10:0.01")
    ap.add_argument("--num_mc", type=int, default=5)

    ap.add_argument("--p_fixed", type=float, default=0.05)
    ap.add_argument("--T_values", type=str, default="1,2,5,10,20")

    ap.add_argument("--out_dir", type=str, default="ablation_outputs")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else (args.device if torch.cuda.is_available() else "cpu"))
    print(f"[Device] {device}")

    train_ds, test_ds, num_classes = get_dataset(args.dataset, args.data_dir, img_size=args.img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = extract_state_dict(ckpt)

    model = build_model_from_repo(args.model, num_classes=num_classes, dropout_p=0.0)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if args.strict and (missing or unexpected):
        raise RuntimeError(f"Strict load failed.\nMissing: {missing}\nUnexpected: {unexpected}")
    if missing:
        print(f"[Warn] Missing keys (strict=False): {len(missing)}")
    if unexpected:
        print(f"[Warn] Unexpected keys (strict=False): {len(unexpected)}")

    out_dir = os.path.join(args.out_dir, args.dataset, args.model)
    ensure_dir(out_dir)
    print(f"[Out] {out_dir}")

    dropout_values = parse_float_list(args.dropout_values)
    T_values = parse_int_list(args.T_values)

    # Ablation A
    dfA = run_ablation_A(
        model, train_loader, test_loader, device,
        dropout_values, args.num_mc, args.max_train_samples, args.max_test_samples
    )
    dfA.insert(0, "dataset", args.dataset)
    dfA.insert(1, "model", args.model)
    csvA = os.path.join(out_dir, f"{args.dataset}_{args.model}_A_p_sweep_num_mc{args.num_mc}.csv")
    dfA.to_csv(csvA, index=False)
    print(f"[Saved] {csvA}")
    plot_ablation_A(dfA, out_dir, args.dataset, args.model, args.num_mc)

    # Ablation B
    dfB = run_ablation_B(
        model, train_loader, test_loader, device,
        args.p_fixed, T_values, args.max_train_samples, args.max_test_samples
    )
    dfB.insert(0, "dataset", args.dataset)
    dfB.insert(1, "model", args.model)
    csvB = os.path.join(out_dir, f"{args.dataset}_{args.model}_B_T_sweep_p{args.p_fixed}.csv")
    dfB.to_csv(csvB, index=False)
    print(f"[Saved] {csvB}")
    plot_ablation_B(dfB, out_dir, args.dataset, args.model, args.p_fixed)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
