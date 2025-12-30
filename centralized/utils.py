import os
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (can slow a bit; toggle if you want)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def to_serializable(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert argparse Namespace / dict containing non-serializable objects to JSON-friendly.
    """
    out = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


def pretty_ckpt_dir(output_dir: str,
                    dataset: str,
                    model: str,
                    seed: int,
                    epochs: int,
                    lr: float,
                    weight_decay: float,
                    train_dropout_p: float) -> Path:
    """
    Folder convention:
    <output-dir>/checkpoints/<dataset>/<model>/seed=<seed>/epochs=<E>_lr=<LR>_wd=<WD>_trainDp=<p>/
    """
    ckpt_dir = Path(output_dir) / "checkpoints" / dataset / model / f"seed={seed}"
    run = f"epochs={epochs}_lr={lr}_wd={weight_decay}_trainDp={train_dropout_p}"
    return ckpt_dir / run


def save_checkpoint(path: str | Path,
                    model_state: Dict[str, Any],
                    meta: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(model_state, path)
    save_json(meta, path.parent / "meta.json")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    """
    Loads both:
      - our dict format: {'model_state_dict': ..., 'num_classes': ..., 'arch': ..., ...}
      - raw state_dict only
      - other wrappers with key 'state_dict'
    """
    path = Path(path)
    obj = torch.load(path, map_location=map_location)

    if isinstance(obj, dict):
        # common patterns
        if "model_state_dict" in obj:
            return obj
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            # e.g. {'state_dict': weights, ...}
            obj["model_state_dict"] = obj["state_dict"]
            return obj
        # Maybe it's already a plain state dict (param_name -> tensor)
        if all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values()):
            return {"model_state_dict": obj}
        return obj

    # Unexpected format
    raise ValueError(f"Unsupported checkpoint format at {path} (type={type(obj)})")
