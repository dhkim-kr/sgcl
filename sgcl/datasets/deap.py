"""DEAP loader.

Features are precomputed DE_LDS / PSD_LDS .mat files (one per subject) with
arrays of shape (32 channels, 40 trials, 63 windows, 4 bands); windows are 1 s
at 128 Hz over the theta / alpha / beta / gamma bands, LDS-smoothed.
Labels come from the official `data_preprocessed_matlab/sXX.mat` files
(`labels` array, columns [valence, arousal, ...], ratings 1-9), tiled per
trial over its 63 windows.
"""

import os

import natsort
import numpy as np
from scipy import io


def load_deap_features(feature_dir: str, var_name: str) -> np.ndarray:
    """Returns (subjects, 2520, bands, channels)."""
    file_list = natsort.natsorted(f for f in os.listdir(feature_dir)
                                  if f.endswith(".mat"))
    subjects = []
    for fname in file_list:
        data = io.loadmat(os.path.join(feature_dir, fname))[var_name]
        swapped = data.transpose(1, 2, 3, 0)  # (40, 63, 4, 32)
        shape = swapped.shape
        subjects.append(swapped.reshape(shape[0] * shape[1], shape[2], shape[3]))
    return np.stack(subjects)


def load_deap_labels(label_dir: str, n_windows_per_trial: int = 63,
                     n_columns: int = 2) -> np.ndarray:
    """Returns (subjects, 2520, 2) per-window [valence, arousal] ratings."""
    file_list = natsort.natsorted(f for f in os.listdir(label_dir) if f.endswith(".mat"))
    subjects = []
    for fname in file_list:
        ratings = io.loadmat(os.path.join(label_dir, fname))["labels"][:, :n_columns]
        subjects.append(np.repeat(ratings, n_windows_per_trial, axis=0))
    return np.stack(subjects)
