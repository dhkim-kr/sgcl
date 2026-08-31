"""SEED loader (precomputed ExtractedFeatures .mat files).

Expected layout:
    <data_dir>/   45 .mat files ("<subject>_<date>.mat", 15 subjects x 3 sessions),
                  each holding `de_LDS1..15` / `psd_LDS1..15` arrays of shape
                  (62 channels, T windows, 5 bands)
    <label_dir>/label.mat   with key 'label': 15 trial labels in {-1, 0, 1}

Each subject's 3 sessions are concatenated along the sample axis
(3 x 3394 = 10182 one-second windows). Labels are shifted to {0, 1, 2}
(negative / neutral / positive).
"""

import os

import natsort
import numpy as np
from scipy import io


def load_seed_data(data_dir: str, label_dir: str, feature_name: str = "de_LDS",
                   n_trials: int = 15):
    """Returns (subjects, samples, bands, channels), labels, per-trial sample counts."""
    file_list = natsort.natsorted(os.listdir(data_dir))
    label_mat = io.loadmat(os.path.join(label_dir, "label.mat"))["label"]
    trial_labels = [1 + int(label_mat[0][i]) for i in range(n_trials)]

    subject_features, subject_labels, subject_counts = [], [], []
    session_features, session_labels, session_counts = [], [], []

    for idx, fname in enumerate(file_list):
        data = io.loadmat(os.path.join(data_dir, fname))
        for trial_idx in range(1, n_trials + 1):
            trial = data[feature_name + str(trial_idx)].transpose(1, 2, 0)  # (T, 5, 62)
            session_features.append(trial)
            session_labels.append(np.full(trial.shape[0], trial_labels[trial_idx - 1]))
            session_counts.append(trial.shape[0])
        if (idx + 1) % 3 == 0:  # every 3 consecutive files = one subject's sessions
            subject_features.append(np.vstack(session_features))
            subject_labels.append(np.concatenate(session_labels))
            subject_counts.append(np.array(session_counts))
            session_features, session_labels, session_counts = [], [], []

    return (np.stack(subject_features), np.stack(subject_labels).astype(np.int64),
            np.stack(subject_counts))
