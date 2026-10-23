"""Shared config, seeding, and device selection."""

from __future__ import annotations

import random
from pathlib import Path

import yaml

from moe.data import project_root


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else project_root() / "config.yaml"
    if not config_path.is_absolute() and not config_path.exists():
        config_path = project_root() / config_path
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} is empty")
    config["_config_path"] = str(config_path.resolve())
    return config


def pick_device(requested: str | None = None):
    """MPS on Apple Silicon, then CUDA, then CPU."""
    import torch

    if requested:
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
