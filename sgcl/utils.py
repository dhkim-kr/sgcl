"""Reproducibility, device selection, and result-saving helpers."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fix every random source used by the pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "cuda:0") -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[sgcl] {device} requested but CUDA is unavailable -- falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)


def save_np(dir_path: str, file_name: str, array) -> str:
    os.makedirs(dir_path, exist_ok=True)
    path = os.path.join(dir_path, file_name)
    np.save(path, array)
    return path + ".npy"
