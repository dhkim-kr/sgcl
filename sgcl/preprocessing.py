"""Feature flattening, standardization, and labeled-sample selection.

Labeled-node selection reproduces the original experiments exactly, including
the fixed `random.seed` reset before every draw. The trial-stratified sampler
draws indices from `range(start, end - 1)`, i.e. the last window of each trial
is never selected — kept as-is for reproducibility.
"""

import random

import numpy as np


def flatten_features(arr: np.ndarray) -> np.ndarray:
    """(N, bands, channels) -> (N, bands * channels), band-major."""
    return arr.reshape(arr.shape[0], -1)


def standardize(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return (x - x.mean(axis=axis)) / x.std(axis=axis)


def select_labeled_by_trial(sample_counts, n_per_trial: int, n_samples: int,
                            seed: int) -> np.ndarray:
    """SEED: sample n_per_trial labeled windows inside each trial's index range."""
    identifier = np.zeros(n_samples, dtype=bool)
    random.seed(seed)
    start = end = 0
    for trial_idx, count in enumerate(sample_counts):
        end += int(count)
        if count - 1 < n_per_trial:
            raise ValueError(
                f"trial {trial_idx} has {count} windows; cannot draw "
                f"{n_per_trial} labeled samples from its first {count - 1}")
        idcs = random.sample(range(start, end - 1), n_per_trial)
        identifier[idcs] = True
        start += int(count)
    return identifier


def select_labeled_by_class(labels: np.ndarray, n_per_class: int, n_classes: int,
                            seed: int) -> np.ndarray:
    """SEED-IV: class-stratified sampling over the whole sample pool."""
    identifier = np.zeros(labels.shape[0], dtype=bool)
    random.seed(seed)
    for c in range(n_classes):
        indices = np.where(labels == c)[0]
        if len(indices) < n_per_class:
            raise ValueError(f"class {c} has only {len(indices)} samples; "
                             f"cannot label {n_per_class} of them")
        labeled_idx = random.sample(sorted(indices), n_per_class)
        identifier[labeled_idx] = True
    return identifier


def select_labeled_deap(vlc_labels: np.ndarray, ars_labels: np.ndarray,
                        n_per_class: int, n_classes: int, seed: int):
    """DEAP: independent class-balanced masks for valence and arousal.

    Both masks come from a single RNG stream seeded once (valence classes
    first, then arousal), matching the original draw order.
    """
    n_samples = vlc_labels.shape[0]
    vlc_identifier = np.zeros(n_samples, dtype=bool)
    ars_identifier = np.zeros(n_samples, dtype=bool)
    random.seed(seed)
    for task, labels, identifier in (("valence", vlc_labels, vlc_identifier),
                                     ("arousal", ars_labels, ars_identifier)):
        for c in range(n_classes):
            indices = np.where(labels == c)[0]
            if len(indices) < n_per_class:
                raise ValueError(f"{task} class {c} has only {len(indices)} "
                                 f"samples; cannot label {n_per_class} of them")
            identifier[random.sample(sorted(indices), n_per_class)] = True
    return vlc_identifier, ars_identifier


def binarize_deap_labels(ratings: np.ndarray, threshold: float = 5.0):
    """Per-window [valence, arousal] ratings -> binary {0: low, 1: high} labels."""
    vlc = (ratings[:, 0] >= threshold).astype(np.int64)
    ars = (ratings[:, 1] >= threshold).astype(np.int64)
    return vlc, ars
