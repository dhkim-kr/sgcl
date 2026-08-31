"""Method ablation G1-G9 (paper Tables 7-8).

Scheme = model arm x feature embedding:
    arm "pca_gcn" (G1-G3): PCA features + supervised GCN (`CompareModel`)
    arm "ge"      (G4-G6): raw features + SGCL model trained supervised-only
    arm "gcl"     (G7-G9): full graph contrastive training
    features "de" / "psd" (single-modality graph, SSN) / "both" (SSNF fusion)

G9 (gcl + both) is the main result and equals `run_subject_dependent` /
`run_deap`; it is included here so a full ablation table can be produced in
one command.
"""

import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import graph
from .experiment import (_single_case, _subject_indices, _summarize, _train_cfg,
                         load_or_build_cache, select_subject_labels)
from .models import CompareModel, Encoder, SGCL
from .preprocessing import flatten_features, standardize
from .training import (train_supervised_encoder, train_supervised_gcn,
                       train_transductive)
from .utils import get_device, save_np, seed_everything

SCHEMES = {
    "de_pca_gcn": ("pca_gcn", "de"),
    "psd_pca_gcn": ("pca_gcn", "psd"),
    "de_psd_pca_snf_gcn": ("pca_gcn", "both"),
    "de_ge": ("ge", "de"),
    "psd_ge": ("ge", "psd"),
    "de_psd_snf_ge": ("ge", "both"),
    "de_gcl": ("gcl", "de"),
    "psd_gcl": ("gcl", "psd"),
    "de_psd_snf_gcl": ("gcl", "both"),
}

# the original G1 cell runs the DE PCA without prior standardization
PCA_STANDARDIZE = {"de_pca_gcn": {"de": False}}


def _ablation_params(cfg: dict) -> dict:
    params = {
        "epochs": 200,
        "learning_rate": 0.001,
        "pca_components": {"de": 9, "psd": 6},
        "ge_proj_hid_channels": {},
    }
    params.update(cfg.get("ablation", {}))
    return params


def _pca_features(flat: np.ndarray, n_components: int, standardize_first: bool):
    if standardize_first:
        flat = StandardScaler().fit_transform(flat)
    return PCA(n_components=n_components).fit_transform(flat)


def _single_graph(feat: np.ndarray, g: dict):
    """Sparse SSM of a single modality; returns (raw ssm, normalized ssm)."""
    dm = graph.distance_matrix(feat)
    neighbors = graph.kneighbors(dm, g["k_broad"])
    return graph.ssm_construction(dm, neighbors, g["kernel_scale"],
                                  g["symmetrization"])


def _to_torch(feature, adjacency, labels, identifier, device):
    return (torch.from_numpy(feature).to(torch.float32).to(device),
            torch.from_numpy(adjacency).to(torch.float32).to(device),
            torch.from_numpy(labels).to(torch.long).to(device),
            torch.from_numpy(identifier).to(device))


def _run_scheme_subject(scheme, cfg, params, arrays, i, n_labeled, device, verbose):
    arm, features = SCHEMES[scheme]
    g = cfg["graph"]
    m = cfg["model"]
    subject_de, subject_psd = arrays[0], arrays[1]
    labels, identifier = select_subject_labels(cfg, arrays, i, n_labeled)
    de_raw = flatten_features(subject_de[i])
    psd_raw = flatten_features(subject_psd[i])

    if arm == "pca_gcn":
        std_flags = PCA_STANDARDIZE.get(scheme, {})
        de_f = _pca_features(de_raw, params["pca_components"]["de"],
                             std_flags.get("de", True))
        psd_f = _pca_features(psd_raw, params["pca_components"]["psd"],
                              std_flags.get("psd", True))
    else:
        de_f = standardize(de_raw)
        psd_f = standardize(psd_raw)

    if features == "de":
        feature = de_f
        ssm, nssm = _single_graph(de_f, g)
        adjacency = nssm if arm == "gcl" else ssm
    elif features == "psd":
        feature = psd_f
        ssm, nssm = _single_graph(psd_f, g)
        adjacency = nssm if arm == "gcl" else ssm
    else:
        feature = np.concatenate((de_f, psd_f), axis=1)
        de_ssm, de_nssm = _single_graph(de_f, g)
        psd_ssm, psd_nssm = _single_graph(psd_f, g)
        adjacency = graph.ssm_fusion(de_ssm, psd_ssm, de_nssm, psd_nssm,
                                     g["k_local"], g["diffusion_steps"],
                                     g["symmetrization"])

    feature_t, adj_t, label_t, train_mask = _to_torch(feature, adjacency, labels,
                                                      identifier, device)
    test_mask = ~train_mask
    eye = torch.eye(feature_t.shape[0], device=device)
    in_dim = feature_t.shape[1]
    num_classes = m["num_classes"]

    if arm == "pca_gcn":
        adj_t = graph.normalize_adj(adj_t, eye)
        model = CompareModel(in_dim, in_dim, 2 * in_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        return train_supervised_gcn(model, optimizer, feature_t, adj_t, label_t,
                                    train_mask, test_mask, params["epochs"], verbose)

    if arm == "ge":
        adj_t = graph.normalize_adj(adj_t, eye)
        if features == "both":
            hid, out = m["hid_channels"], m["out_channels"]
        else:
            hid, out = m["hid_channels"] // 2, m["out_channels"] // 2
        proj = params["ge_proj_hid_channels"].get(features, m["proj_hid_channels"])
        model = SGCL(Encoder(in_dim, hid, out), out, proj, num_classes,
                     cfg["train"]["tau"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        return train_supervised_encoder(model, optimizer, feature_t, adj_t, label_t,
                                        train_mask, test_mask, params["epochs"],
                                        verbose)

    # arm == "gcl": full SGCL training; single-modality runs keep the full
    # encoder width and only shrink the input dimension (as in the notebooks)
    model = SGCL(Encoder(in_dim, m["hid_channels"], m["out_channels"]),
                 m["out_channels"], m["proj_hid_channels"], num_classes,
                 cfg["train"]["tau"]).to(device)
    train_cfg = _train_cfg(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    return train_transductive(model, optimizer, feature_t, adj_t, label_t,
                              train_mask, test_mask, train_cfg, verbose=verbose,
                              timed=train_cfg.get("timed", False))


def run_ablation(cfg: dict) -> list:
    exp = cfg["experiment"]
    if cfg["graph"].get("variant", "full") != "full":
        raise ValueError("graph variants (Table 10) apply to the main pipeline, "
                         "not the method ablation -- drop --graph-variant")
    device = get_device(exp.get("device", "cuda:0"))
    out_dir = exp["output_dir"]
    verbose = exp.get("verbose", False)
    params = _ablation_params(cfg)
    schemes = exp.get("schemes") or list(SCHEMES)
    unknown = [s for s in schemes if s not in SCHEMES]
    if unknown:
        raise ValueError(f"unknown ablation scheme(s) {unknown}; "
                         f"choose from {list(SCHEMES)}")
    case = _single_case(cfg, "ablation")
    n_labeled = exp["cases"][case]
    seed = cfg["train"]["seed"]

    arrays = load_or_build_cache(cfg)
    subjects = _subject_indices(cfg, arrays[0].shape[0])
    task_prefix = cfg["experiment"]["task"] + "_" if cfg["dataset"] == "deap" else ""

    summaries = []
    for scheme in schemes:
        print(f"===== ablation scheme {scheme} (case {case}) =====")
        results = []
        for i in subjects:
            seed_everything(seed)
            result = _run_scheme_subject(scheme, cfg, params, arrays, i,
                                         n_labeled, device, verbose)
            results.append(result)
            save_np(os.path.join(out_dir, "ablation",
                                 f"{task_prefix}{scheme}", f"sub{i + 1}"),
                    "embedding", result["embedding"])
            print(f"  subject {i + 1:2d}  acc {result['best_acc']:.2f}  "
                  f"f1 {result['f1']:.4f}")
        summaries.append(_summarize(f"{task_prefix}{scheme}_case{case}", results,
                                    os.path.join(out_dir, "ablation")))
        if cfg["train"].get("timed", False):
            times = [r["train_time"] for r in results]
            print(f"  average train time per subject: {np.mean(times):.3f} s "
                  "(GCL arm only; supervised arms are untimed)")
    return summaries
