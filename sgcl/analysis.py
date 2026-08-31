"""Analysis utilities: saliency topomaps (paper Fig. 5), t-SNE embedding
scatters (Fig. 4), and the paired t-test used for Table 2 significance marks."""

import os

import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 62-channel ESI NeuroScan order used by SEED / SEED-IV
SEED_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz',
    'O2', 'CB2']
DROP_CHANNELS = ['CB1', 'CB2']  # not present in the standard_1020 montage
FREQUENCY_BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
SEED_IV_EMOTIONS = ['Neutral', 'Sad', 'Fear', 'Happy']


def saliency_map_analysis(models, x_list, adj_list, labels_list, test_mask_list,
                          out_path, emotion_labels=SEED_IV_EMOTIONS,
                          band_names=FREQUENCY_BANDS, channel_names=SEED_CHANNELS,
                          drop_channels=DROP_CHANNELS, sfreq=200, device=None):
    """Gradient-based saliency topomaps (paper Fig. 5).

    For every correctly classified unlabeled sample, the true-class logit is
    backpropagated to the 620-d input; gradients are reshaped to
    [DE/PSD, band, channel], accumulated per emotion, fused by relative
    magnitude, min-max normalized, and drawn as emotion x band topomaps.
    """
    import matplotlib.pyplot as plt
    import mne

    if device is None:
        device = x_list[0].device
    n_bands = len(band_names)
    n_channels = len(channel_names)
    mne_channels = [ch for ch in channel_names if ch not in drop_channels]
    channel_index_map = [i for i, ch in enumerate(channel_names)
                        if ch in mne_channels]

    info = mne.create_info(ch_names=mne_channels, sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"),
                     match_case=True)
    for ch in info['chs']:  # shift electrode positions down as in the original
        ch['loc'][1] -= 0.02

    saliency = {emo: torch.zeros(n_channels, n_bands, 2) for emo in emotion_labels}
    counts = {emo: 0 for emo in emotion_labels}

    for subj_idx, model in enumerate(models):
        model = model.to(device)
        model.eval()
        x = x_list[subj_idx].to(device).clone().detach().requires_grad_(True)
        labels = labels_list[subj_idx].to(device)
        test_mask = test_mask_list[subj_idx].to(device)
        adj = adj_list[subj_idx].to(device)

        with torch.no_grad():
            pred = model.classification(model.projection(model(x, adj))).argmax(dim=1)
        correct = (pred == labels) & test_mask
        print(f"subject {subj_idx + 1}: correct samples = {correct.sum().item()}")

        for i in range(x.shape[0]):
            if not correct[i]:
                continue
            model.zero_grad()
            output = model.classification(model.projection(model(x, adj)))
            output[i, labels[i]].backward(retain_graph=True)
            grad = x.grad[i].detach().abs().view(2, n_bands, n_channels)
            emotion = emotion_labels[labels[i].item()]
            saliency[emotion] += grad.permute(2, 1, 0).cpu()
            counts[emotion] += 1
            x.grad.zero_()

    fused_by_emotion = {}
    fig, axes = plt.subplots(len(emotion_labels), n_bands, figsize=(15, 10))
    im_for_colorbar = None
    for row, emo in enumerate(emotion_labels):
        if counts[emo] == 0:
            continue
        sal = (saliency[emo] / counts[emo])[channel_index_map]  # [60, bands, 2]
        de, psd = sal[:, :, 0], sal[:, :, 1]
        total = de + psd + 1e-8
        fused = (de / total) * de + (psd / total) * psd
        if (fused.max() - fused.min()) > 0:
            fused = (fused - fused.min()) / (fused.max() - fused.min())
        else:
            fused = torch.zeros_like(fused)
        fused_by_emotion[emo] = fused

        for col, band in enumerate(band_names):
            im, _ = mne.viz.plot_topomap(
                fused[:, col].numpy(), info, axes=axes[row, col], show=False,
                contours=0, sphere=(0., 0., 0., 0.09), outlines='head',
                extrapolate='head', image_interp='cubic', res=128)
            im.set_clim(0, 1)
            im.set_cmap("Reds")
            if im_for_colorbar is None:
                im_for_colorbar = im
            if row == 0:
                axes[row, col].set_title(band, fontsize=18)
            if col == 0:
                axes[row, col].set_ylabel(emo, fontsize=18)

    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(im_for_colorbar, cax=cbar_ax, orientation="vertical")
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"saliency topomap saved to {out_path}")
    return fused_by_emotion, counts


# ---------------------------------------------------------------------------
# t-SNE embedding scatters (Fig. 4)
# ---------------------------------------------------------------------------

CLASS_STYLES = {
    3: (["negative", "neutral", "positive"],
        ['#FF6600', '#CCFF66', '#33CCFF'], ['red', 'green', 'blue']),
    4: (["neutral", "sad", "fear", "happy"],
        ['#999999', '#CCFF66', '#FF6600', '#33CCFF'],
        ['black', 'green', 'red', 'blue']),
    2: (["low", "high"], ['#FF6600', '#33CCFF'], ['red', 'blue']),
}


def tsne_embedding(sample: np.ndarray, perplexity: float = 20,
                   n_iter: int = 500, standardize_first: bool = True) -> np.ndarray:
    if standardize_first:
        sample = StandardScaler().fit_transform(sample)
    return TSNE(n_components=2, perplexity=perplexity,
                max_iter=n_iter).fit_transform(sample)


def save_scatter(result: np.ndarray, labels: np.ndarray, labeled_mask: np.ndarray,
                 title: str, out_path: str, class_names=None):
    """2-D scatter of an embedding, colored by class, with 'x' markers on the
    labeled (ground-truth) samples — the Fig. 4 style."""
    import matplotlib.pyplot as plt

    n_classes = int(labels.max()) + 1
    names, colors, gt_colors = CLASS_STYLES[n_classes]
    if class_names is not None:
        names = class_names

    plt.clf()
    handles, legend = [], []
    for c in range(n_classes):
        cw = np.where(labels == c)[0]
        handles.append(plt.scatter(result[cw, 0], result[cw, 1], marker="o",
                                   color=colors[c], s=1.))
        legend.append(names[c])
    for c in range(n_classes):
        gw = np.where((labels == c) & labeled_mask)[0]
        handles.append(plt.scatter(result[gw, 0], result[gw, 1], marker="x",
                                   color=gt_colors[c], s=2.))
        legend.append(names[c] + " - gt")
    plt.title(title)
    plt.xlabel("Reduced axis - 1")
    plt.ylabel("Reduced axis - 2")
    plt.legend(handles=handles, labels=legend)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, facecolor="#eeeeee", bbox_inches='tight')
    print(f"scatter saved to {out_path}")


# ---------------------------------------------------------------------------
# paired t-test (Table 2 significance)
# ---------------------------------------------------------------------------

def paired_t_test(scores_by_method: dict, reference: str = "SGCL",
                  alpha: float = 0.001):
    """Paired t-tests of the reference method against every other method.

    `scores_by_method` maps method name -> list of per-run accuracies (equal
    lengths). Returns a list of (method, t_statistic, p_value, significant).
    """
    ref = np.asarray(scores_by_method[reference], dtype=float)
    rows = []
    for method, scores in scores_by_method.items():
        if method == reference:
            continue
        t_stat, p_val = ttest_rel(ref, np.asarray(scores, dtype=float))
        rows.append((method, float(t_stat), float(p_val), bool(p_val < alpha)))
    return rows
