"""SEED-IV loader (precomputed eeg_feature_smooth .mat files).

Expected layout:
    <data_dir>/1/, <data_dir>/2/, <data_dir>/3/   one directory per session,
    each holding 15 subject .mat files with keys `de_LDS1..24` / `psd_LDS1..24`
    of shape (62 channels, W windows, 5 bands); windows are 4 s.

Per subject the 3 sessions are concatenated along the sample axis
(851 + 832 + 822 = 2505 windows). Trial labels are the official
session-dependent lists (0 neutral, 1 sad, 2 fear, 3 happy).
"""

import os

import natsort
import numpy as np
from scipy import io

SESSION_LABELS = np.array([
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
])


def load_seed_iv_data(data_dir: str, feature_name: str = "de_LDS", n_trials: int = 24):
    """Returns (subjects, samples, bands, channels), labels, per-trial sample counts."""
    session_dirs = natsort.natsorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))

    per_subject_features, per_subject_labels, per_subject_counts = None, None, None
    for ses_idx, ses_dir in enumerate(session_dirs):
        ses_path = os.path.join(data_dir, ses_dir)
        file_list = natsort.natsorted(os.listdir(ses_path))
        ses_features, ses_labels, ses_counts = [], [], []
        for fname in file_list:
            data = io.loadmat(os.path.join(ses_path, fname))
            trials, labels, counts = [], [], []
            for trial_idx in range(1, n_trials + 1):
                trial = data[feature_name + str(trial_idx)].transpose(1, 2, 0)  # (W, 5, 62)
                trials.append(trial)
                labels.append(np.full(trial.shape[0], SESSION_LABELS[ses_idx][trial_idx - 1]))
                counts.append(trial.shape[0])
            ses_features.append(np.vstack(trials))
            ses_labels.append(np.concatenate(labels))
            ses_counts.append(np.array(counts))
        ses_features = np.stack(ses_features)
        ses_labels = np.stack(ses_labels)
        ses_counts = np.stack(ses_counts)
        if per_subject_features is None:
            per_subject_features, per_subject_labels, per_subject_counts = (
                ses_features, ses_labels, ses_counts)
        else:
            per_subject_features = np.concatenate((per_subject_features, ses_features), axis=1)
            per_subject_labels = np.concatenate((per_subject_labels, ses_labels), axis=1)
            per_subject_counts = np.concatenate((per_subject_counts, ses_counts), axis=1)

    return (per_subject_features, per_subject_labels.astype(np.int64), per_subject_counts)
