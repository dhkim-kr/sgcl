"""Experiment drivers: subject-dependent runs for SEED / SEED-IV / DEAP and the
SEED-IV cross-session run (all transductive, single fixed seed, results are
mean +/- std of per-subject best test accuracies)."""

import json
import os

import numpy as np
import torch

from . import graph
from .datasets import (load_deap_features, load_deap_labels, load_seed_data,
                       load_seed_iv_data)
from .models import Encoder, SGCL
from .preprocessing import (binarize_deap_labels, flatten_features,
                            select_labeled_by_class, select_labeled_by_trial,
                            select_labeled_deap, standardize)
from .training import train_transductive
from .utils import get_device, save_np, seed_everything

CACHE_FILES = ("subject_de.npy", "subject_psd.npy", "subject_label.npy",
               "subject_sample_counts.npy")


# ---------------------------------------------------------------------------
# data caching
# ---------------------------------------------------------------------------

def _cache_ready(cache_dir: str, names=CACHE_FILES) -> bool:
    return all(os.path.exists(os.path.join(cache_dir, n)) for n in names)


def load_or_build_cache(cfg: dict):
    """Load per-subject feature/label arrays, building the .npy cache from the
    raw .mat files on first use. A manifest guards against silently reusing a
    cache built from different source paths or feature names."""
    data = cfg["data"]
    cache_dir = data["cache_dir"]
    dataset = cfg["dataset"]
    names = CACHE_FILES if dataset != "deap" else CACHE_FILES[:3]
    manifest = {k: data[k] for k in sorted(data) if k != "cache_dir"}
    manifest_path = os.path.join(cache_dir, "cache_manifest.json")

    if _cache_ready(cache_dir, names):
        if not os.path.exists(manifest_path):
            raise ValueError(
                f"cache in {cache_dir} has no manifest, so its source cannot "
                "be verified; delete the cache directory to rebuild it")
        with open(manifest_path) as f:
            cached = json.load(f)
        if cached != manifest:
            raise ValueError(
                f"cache in {cache_dir} was built from a different data "
                f"configuration ({cached}); delete the cache directory or "
                "point data.cache_dir elsewhere")
        arrays = [np.load(os.path.join(cache_dir, n)) for n in names]
        return arrays if dataset != "deap" else arrays + [None]

    print(f"[sgcl] cache not found in {cache_dir} -- loading raw data")
    if dataset == "seed":
        de, label, counts = load_seed_data(data["feature_dir"], data["label_dir"],
                                           data["de_feature"])
        psd, _, _ = load_seed_data(data["feature_dir"], data["label_dir"],
                                   data["psd_feature"])
    elif dataset in ("seed_iv", "seed_iv_cross_session"):
        de, label, counts = load_seed_iv_data(data["feature_dir"], data["de_feature"])
        psd, _, _ = load_seed_iv_data(data["feature_dir"], data["psd_feature"])
    elif dataset == "deap":
        de = load_deap_features(data["de_feature_dir"], data["de_var"])
        psd = load_deap_features(data["psd_feature_dir"], data["psd_var"])
        label = load_deap_labels(data["label_dir"])
        counts = None
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    save_np(cache_dir, "subject_de", de)
    save_np(cache_dir, "subject_psd", psd)
    save_np(cache_dir, "subject_label", label)
    if counts is not None:
        save_np(cache_dir, "subject_sample_counts", counts)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return de, psd, label, counts


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------

def _subject_indices(cfg: dict, n_subjects: int):
    subjects = cfg["experiment"].get("subjects", "all")
    if subjects == "all":
        return list(range(n_subjects))
    bad = [s for s in subjects if not 1 <= s <= n_subjects]
    if bad:
        raise ValueError(f"subject numbers {bad} out of range 1..{n_subjects} "
                         "(subjects are 1-based)")
    return [s - 1 for s in subjects]


def _build_model(in_channels: int, cfg: dict, device: torch.device) -> SGCL:
    m = cfg["model"]
    encoder = Encoder(in_channels, m["hid_channels"], m["out_channels"])
    model = SGCL(encoder, m["out_channels"], m["proj_hid_channels"],
                 m["num_classes"], cfg["train"]["tau"])
    return model.to(device)


def _train_cfg(cfg: dict, **overrides) -> dict:
    t = dict(cfg["train"])
    t.update(overrides)
    return t


def _run_one(feature: np.ndarray, adjacency: np.ndarray, labels: np.ndarray,
             identifier: np.ndarray, cfg: dict, device: torch.device,
             train_cfg: dict, verbose: bool, return_model: bool = False):
    feature_t = torch.from_numpy(feature).to(torch.float32).to(device)
    adj_t = torch.from_numpy(adjacency).to(torch.float32).to(device)
    label_t = torch.from_numpy(labels).to(torch.long).to(device)
    train_mask = torch.from_numpy(identifier).to(device)
    test_mask = ~train_mask

    model = _build_model(feature.shape[1], cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    result = train_transductive(model, optimizer, feature_t, adj_t, label_t,
                                train_mask, test_mask, train_cfg, verbose=verbose,
                                timed=train_cfg.get("timed", False))
    if return_model:
        return result, model, (feature_t, adj_t, label_t, test_mask)
    return result


def select_subject_labels(cfg: dict, arrays, i: int, n_labeled: int):
    """Dataset-specific labels + labeled mask for subject i."""
    dataset = cfg["dataset"]
    _, _, subject_label, subject_counts = arrays
    seed = cfg["train"]["seed"]
    if dataset == "seed":
        labels = subject_label[i]
        identifier = select_labeled_by_trial(subject_counts[i], n_labeled,
                                             labels.shape[0], seed)
    elif dataset == "seed_iv":
        labels = subject_label[i]
        identifier = select_labeled_by_class(labels, n_labeled,
                                             cfg["model"]["num_classes"], seed)
    elif dataset == "deap":
        vlc, ars = binarize_deap_labels(subject_label[i])
        vlc_id, ars_id = select_labeled_deap(vlc, ars, n_labeled,
                                             cfg["model"]["num_classes"], seed)
        task = cfg["experiment"]["task"]
        labels, identifier = (vlc, vlc_id) if task == "valence" else (ars, ars_id)
    else:
        raise ValueError(f"unsupported dataset: {dataset}")
    return labels, identifier


def _prepare_subject(cfg: dict, arrays, i: int, n_labeled: int):
    """Standardized features, fused adjacency, labels, and labeled mask."""
    de = standardize(flatten_features(arrays[0][i]))
    psd = standardize(flatten_features(arrays[1][i]))
    labels, identifier = select_subject_labels(cfg, arrays, i, n_labeled)
    adjacency = graph.build_adjacency(de, psd, cfg["graph"])
    feature = np.concatenate((de, psd), axis=1)
    return feature, adjacency, labels, identifier


def _single_case(cfg: dict, mode: str) -> int:
    run_cases = cfg["experiment"]["run_cases"]
    if len(run_cases) > 1:
        print(f"[{mode}] runs a single labeling case -- using case {run_cases[0]}")
    return run_cases[0]


def _summarize(tag: str, results: list, out_dir: str) -> dict:
    accs = [r["best_acc"] for r in results]
    f1s = [r["f1"] for r in results]
    summary = {
        "tag": tag,
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "accs": accs,
        "f1s": f1s,
        "best_epochs": [r["best_epoch"] for r in results],
    }
    save_np(out_dir, f"{tag}_best_acc_list", np.array(accs))
    save_np(out_dir, f"{tag}_f1_list", np.array(f1s))
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{tag}] acc {summary['acc_mean']:.2f} +/- {summary['acc_std']:.2f}"
          f"   f1 {summary['f1_mean']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# subject-dependent experiments
# ---------------------------------------------------------------------------

def run_subject_dependent(cfg: dict) -> list:
    exp = cfg["experiment"]
    device = get_device(exp.get("device", "cuda:0"))
    out_dir = exp["output_dir"]
    verbose = exp.get("verbose", False)
    seed = cfg["train"]["seed"]
    task_prefix = exp["task"] + "_" if cfg["dataset"] == "deap" else ""

    arrays = load_or_build_cache(cfg)
    subjects = _subject_indices(cfg, arrays[0].shape[0])
    summaries = []

    for case in exp["run_cases"]:
        n_labeled = exp["cases"][case]
        results = []
        for i in subjects:
            seed_everything(seed)
            feature, adjacency, labels, identifier = _prepare_subject(
                cfg, arrays, i, n_labeled)

            train_cfg = _train_cfg(cfg)
            result = _run_one(feature, adjacency, labels, identifier, cfg, device,
                              train_cfg, verbose)
            results.append(result)
            subject_dir = os.path.join(out_dir, f"case{case}", f"sub{i + 1}")
            save_np(subject_dir, f"{task_prefix}confusion_matrix",
                    result["confusion_matrix"])
            if exp.get("save_embeddings", False):
                save_np(subject_dir, f"{task_prefix}embedding", result["embedding"])
            print(f"  case {case}  subject {i + 1:2d}  {task_prefix or ''}"
                  f"acc {result['best_acc']:.2f}  f1 {result['f1']:.4f}"
                  f"  (epoch {result['best_epoch']})")
        summaries.append(_summarize(f"{task_prefix}case{case}", results, out_dir))
        if cfg["train"].get("timed", False):
            times = [r["train_time"] for r in results]
            print(f"  average train time per subject: {np.mean(times):.3f} s")
    return summaries




# ---------------------------------------------------------------------------
# SEED-IV cross-session experiment (labeled source session -> unlabeled target)
# ---------------------------------------------------------------------------

def run_cross_session(cfg: dict) -> list:
    exp = cfg["experiment"]
    g = cfg["graph"]
    if g.get("variant", "full") != "full":
        raise ValueError("graph variants apply only to the fused-graph "
                         "pipeline, not the cross-session run")
    device = get_device(exp.get("device", "cuda:0"))
    out_dir = exp["output_dir"]
    verbose = exp.get("verbose", False)
    seed = cfg["train"]["seed"]
    src, tgt = exp["source_session"], exp["target_session"]  # 1-based
    if src == tgt:
        raise ValueError("source_session and target_session must differ")

    subject_de, _, subject_label, subject_counts = load_or_build_cache(cfg)
    session_sizes = subject_counts[0].reshape(3, -1).sum(axis=-1)
    bounds = np.concatenate(([0], np.cumsum(session_sizes)))
    subjects = _subject_indices(cfg, subject_de.shape[0])

    results = []
    for i in subjects:
        seed_everything(seed)
        src_slice = slice(bounds[src - 1], bounds[src])
        tgt_slice = slice(bounds[tgt - 1], bounds[tgt])
        de = np.concatenate((subject_de[i][src_slice], subject_de[i][tgt_slice]))
        labels = np.concatenate((subject_label[i][src_slice],
                                 subject_label[i][tgt_slice]))
        de = standardize(flatten_features(de))

        n_src = int(session_sizes[src - 1])
        identifier = np.zeros(de.shape[0], dtype=bool)
        identifier[:n_src] = True

        # supervised graph pruning: remove edges between labeled nodes of
        # different classes (labels of the source session only)
        labeled_edge = identifier[None, :] & identifier[:, None]
        same_class = labels[None, :] == labels[:, None]
        diff_class_mask = (~same_class) & labeled_edge

        dm = graph.distance_matrix(de)
        neighbors = graph.kneighbors(dm, g["k_broad"])
        de_ssm, _ = graph.ssm_construction(dm, neighbors, g["kernel_scale"],
                                           g["symmetrization"],
                                           drop_mask=diff_class_mask)

        train_cfg = _train_cfg(cfg)
        result = _run_one(de, de_ssm, labels, identifier, cfg, device,
                          train_cfg, verbose)
        results.append(result)
        subject_dir = os.path.join(out_dir, f"session{src}to{tgt}", f"sub{i + 1}")
        save_np(subject_dir, "confusion_matrix", result["confusion_matrix"])
        if exp.get("save_embeddings", False):
            save_np(subject_dir, "embedding", result["embedding"])
        print(f"  session {src}->{tgt}  subject {i + 1:2d}  "
              f"acc {result['best_acc']:.2f} (epoch {result['best_epoch']})")
    if cfg["train"].get("timed", False):
        print(f"  average train time per subject: "
              f"{np.mean([r['train_time'] for r in results]):.3f} s")
    return [_summarize(f"cross_session_{src}to{tgt}", results, out_dir)]


# ---------------------------------------------------------------------------
# hyperparameter sensitivity grid (paper Fig. 6)
# ---------------------------------------------------------------------------

def run_grid(cfg: dict) -> dict:
    """Sweep the edge-drop / feature-mask probabilities over a 9x9 grid
    (both views share each probability), retraining the full pipeline per cell.

    The subject loop is outermost so each subject's graph is built once; with
    per-run reseeding this is numerically equivalent to the original
    grid-outermost ordering."""
    exp = cfg["experiment"]
    device = get_device(exp.get("device", "cuda:0"))
    out_dir = exp["output_dir"]
    seed = cfg["train"]["seed"]
    values = exp.get("grid_values", [i / 10 for i in range(1, 10)])
    case = _single_case(cfg, "grid")
    n_labeled = exp["cases"][case]
    task_prefix = exp["task"] + "_" if cfg["dataset"] == "deap" else ""

    arrays = load_or_build_cache(cfg)
    subjects = _subject_indices(cfg, arrays[0].shape[0])
    n = len(values)
    accs = [[[] for _ in range(n)] for _ in range(n)]

    for i in subjects:
        seed_everything(seed)
        feature, adjacency, labels, identifier = _prepare_subject(
            cfg, arrays, i, n_labeled)
        for ei, pe in enumerate(values):
            for fi, pf in enumerate(values):
                seed_everything(seed)
                train_cfg = _train_cfg(cfg, pe1=pe, pe2=pe, pf1=pf, pf2=pf,
                                       timed=False)
                result = _run_one(feature, adjacency, labels, identifier, cfg,
                                  device, train_cfg, verbose=False)
                accs[ei][fi].append(result["best_acc"])
                print(f"  subject {i + 1:2d}  pe {pe:.1f}  pf {pf:.1f}  "
                      f"acc {result['best_acc']:.2f}")

    acc_grid = np.array([[np.mean(cell) for cell in row] for row in accs])
    std_grid = np.array([[np.std(cell) for cell in row] for row in accs])
    save_np(os.path.join(out_dir, "grid"),
            f"{task_prefix}case{case}_results_acc", acc_grid)
    save_np(os.path.join(out_dir, "grid"),
            f"{task_prefix}case{case}_results_std", std_grid)
    best = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
    print(f"[grid] best acc {acc_grid[best]:.2f} at pe={values[best[0]]}, "
          f"pf={values[best[1]]}")
    return {"acc": acc_grid, "std": std_grid, "values": values}


# ---------------------------------------------------------------------------
# analysis drivers (paper Figs. 4-5)
# ---------------------------------------------------------------------------

def run_saliency(cfg: dict):
    """Retrain per-subject models (short schedule, as in the original 200-epoch
    saliency run) and draw the gradient-saliency topomap grid (paper Fig. 5)."""
    import mne  # noqa: F401 -- fail before hours of training, not after
    from .analysis import CLASS_STYLES, saliency_map_analysis

    if cfg["dataset"] not in ("seed", "seed_iv"):
        raise ValueError("saliency topomaps require the 62-channel SEED montage")
    exp = cfg["experiment"]
    device = get_device(exp.get("device", "cuda:0"))
    seed = cfg["train"]["seed"]
    case = _single_case(cfg, "saliency")
    n_labeled = exp["cases"][case]
    epochs = exp.get("saliency_epochs", 200)

    arrays = load_or_build_cache(cfg)
    subjects = _subject_indices(cfg, arrays[0].shape[0])

    # tensors are parked on CPU between subjects; the analysis moves them to
    # the device one subject at a time
    models, x_list, adj_list, label_list, mask_list = [], [], [], [], []
    for i in subjects:
        seed_everything(seed)
        feature, adjacency, labels, identifier = _prepare_subject(
            cfg, arrays, i, n_labeled)
        train_cfg = _train_cfg(cfg, epochs=epochs)
        result, model, (feature_t, adj_t, label_t, test_mask) = _run_one(
            feature, adjacency, labels, identifier, cfg, device, train_cfg,
            verbose=False, return_model=True)
        print(f"  subject {i + 1:2d}  acc {result['best_acc']:.2f}")
        models.append(model.cpu())
        x_list.append(feature_t.cpu())
        adj_list.append(adj_t.cpu())
        label_list.append(label_t.cpu())
        mask_list.append(test_mask.cpu())

    emotions = CLASS_STYLES[cfg["model"]["num_classes"]][0]
    out_path = os.path.join(exp["output_dir"], "saliency", "saliency_topomap.png")
    return saliency_map_analysis(models, x_list, adj_list, label_list, mask_list,
                                 out_path,
                                 emotion_labels=[e.capitalize() for e in emotions],
                                 device=device)


def run_tsne(cfg: dict):
    """t-SNE scatter (paper Fig. 4) of a saved embedding: run the main
    experiment with `save_embeddings: true` first, or point `tsne.source` at
    any saved embedding .npy (e.g. an ablation scheme's)."""
    from .analysis import save_scatter, tsne_embedding

    exp = cfg["experiment"]
    t = exp.get("tsne", {})
    case = t.get("case", _single_case(cfg, "tsne"))
    subject = t.get("subject", (exp.get("subjects") or [1])[0]
                    if exp.get("subjects") != "all" else 1)
    task_prefix = exp["task"] + "_" if cfg["dataset"] == "deap" else ""
    if t.get("source") and "subject" not in t:
        raise ValueError("experiment.tsne.source is set -- also set "
                         "experiment.tsne.subject so labels match the embedding")
    source = t.get("source") or os.path.join(
        exp["output_dir"], f"case{case}", f"sub{subject}",
        f"{task_prefix}embedding.npy")
    if not os.path.exists(source):
        raise FileNotFoundError(
            f"{source} not found -- run the main experiment with "
            "`save_embeddings: true` first, or set experiment.tsne.source")

    arrays = load_or_build_cache(cfg)
    labels, identifier = select_subject_labels(cfg, arrays, subject - 1,
                                               exp["cases"][case])
    embedding = np.load(source)
    result = tsne_embedding(embedding, perplexity=t.get("perplexity", 20),
                            n_iter=t.get("n_iter", 500))
    out_dir = os.path.join(exp["output_dir"], "tsne")
    save_np(out_dir, f"sub{subject}_{task_prefix}case{case}_tsne", result)
    save_scatter(result, labels, identifier,
                 title=f"sub{subject} ({cfg['dataset']})",
                 out_path=os.path.join(
                     out_dir, f"sub{subject}_{task_prefix}case{case}_tsne.pdf"))
    return result


def run(cfg: dict) -> list:
    mode = cfg["experiment"].get("mode", "main")
    dataset = cfg["dataset"]
    if dataset == "seed_iv_cross_session":
        if mode != "main":
            raise ValueError("the cross-session config only supports the main "
                             "mode (it has no labeling cases to sweep)")
        return run_cross_session(cfg)

    for case in cfg["experiment"].get("run_cases", []):
        if case not in cfg["experiment"].get("cases", {}):
            raise ValueError(f"case {case} is not defined in experiment.cases "
                             f"({sorted(cfg['experiment'].get('cases', {}))})")

    if mode == "ablation":
        from .ablation import run_ablation
        return run_ablation(cfg)
    if mode == "grid":
        return [run_grid(cfg)]
    if mode == "saliency":
        return [run_saliency(cfg)]
    if mode == "tsne":
        return [run_tsne(cfg)]
    if mode != "main":
        raise ValueError(f"unknown mode: {mode}")

    if dataset in ("seed", "seed_iv", "deap"):
        return run_subject_dependent(cfg)
    raise ValueError(f"unknown dataset: {dataset}")
