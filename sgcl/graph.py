"""Symmetric Similarity Network Fusion (SSNF) graph construction.

Builds the fused adjacency matrix A from DE and PSD node features
(paper Sec. 3.3, Eqs. (1)-(5)):

    similarity kernel -> broad k-NN sparsification -> symmetric normalization
    -> local k-NN kernel matrices -> cross-diffusion fusion -> A

`symmetrization` preserves the two variants found in the original
notebooks: "additive" (SEED / SEED-IV: S + S^T - diag, so edges selected
by both endpoints are doubled) and "mirror" (DEAP: S[i,j] = S[j,i] = s_ij).
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist


def distance_matrix(x: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix of row vectors."""
    return cdist(x, x, metric="euclidean")


def kneighbors(matrix: np.ndarray, k: int) -> np.ndarray:
    """Per-row indices of the k entries whose value is closest to the diagonal entry.

    On a distance matrix (diag = 0) this selects the k nearest neighbors; on a
    similarity matrix (diag = max) the k most similar ones. The row's own index
    is always among the k selected.
    """
    diff = np.abs(matrix - np.diag(matrix)[:, None])
    return np.argsort(diff, axis=1, kind="stable")[:, :k].astype(np.int64)


def _symmetrize(matrix: np.ndarray, mode: str) -> np.ndarray:
    if mode == "additive":
        return matrix + matrix.T - np.diag(matrix.diagonal())
    if mode == "mirror":
        # kernel values are symmetric in (i, j), so mirroring == max
        return np.maximum(matrix, matrix.T)
    raise ValueError(f"unknown symmetrization mode: {mode}")


def ssm_construction(dm: np.ndarray, neighbors: np.ndarray, kernel_scale: float = 0.05,
                     symmetrization: str = "mirror", drop_mask: np.ndarray = None):
    """Sparse similarity matrix S(i, j) = exp(-d_ij / (kernel_scale * mean(D))).

    `drop_mask` (optional, boolean) zeroes the marked entries before and after
    normalization — used by the cross-session experiment to remove edges
    between labeled nodes of different classes.

    Returns the sparse similarity matrix and its symmetrically normalized form.
    """
    n = dm.shape[0]
    scale = -dm.mean() * kernel_scale
    ssm = np.zeros((n, n), dtype=np.float64)
    rows = np.repeat(np.arange(n), neighbors.shape[1])
    cols = neighbors.ravel()
    ssm[rows, cols] = np.exp(dm[rows, cols] / scale)
    ssm = _symmetrize(ssm, symmetrization)
    if drop_mask is not None:
        ssm[drop_mask] = 0.0
    nssm = normalize_ssm(ssm)
    if drop_mask is not None:
        nssm[drop_mask] = 0.0
    return ssm, nssm


def normalize_ssm(matrix: np.ndarray) -> np.ndarray:
    """D^{-1/2} S D^{-1/2} (no self-loop added; diagonal of S is already 1)."""
    norm_deg = np.power(matrix.sum(axis=1), -0.5)
    return norm_deg[:, None] * matrix * norm_deg[None, :]


def ssm_fusion(ssm_1: np.ndarray, ssm_2: np.ndarray, nssm_1: np.ndarray, nssm_2: np.ndarray,
               k_local: int, t: int = 1, symmetrization: str = "mirror") -> np.ndarray:
    """Cross-diffusion fusion of two similarity networks (paper Eqs. (4)-(5)).

    Local kernel matrices keep only each node's k_local most similar neighbors;
    the normalized broad networks are diffused through the other modality's
    kernel for t steps and averaged.
    """
    n = ssm_1.shape[0]
    skm_1 = np.zeros((n, n), dtype=np.float64)
    skm_2 = np.zeros((n, n), dtype=np.float64)

    rows = np.repeat(np.arange(n), k_local)
    n1 = kneighbors(ssm_1, k_local).ravel()
    n2 = kneighbors(ssm_2, k_local).ravel()
    skm_1[rows, n1] = ssm_1[rows, n1]
    skm_2[rows, n2] = ssm_2[rows, n2]

    skm_1 = normalize_ssm(_symmetrize(skm_1, symmetrization))
    skm_2 = normalize_ssm(_symmetrize(skm_2, symmetrization))

    for step in range(t):
        prev = nssm_1
        nssm_1 = skm_1 @ nssm_2 @ skm_1
        nssm_2 = skm_2 @ prev @ skm_2
        if t > 1 and step < t - 1:
            nssm_1 = normalize_ssm(nssm_1)
            nssm_2 = normalize_ssm(nssm_2)

    return (nssm_1 + nssm_2) / 2


def build_fused_adjacency(de: np.ndarray, psd: np.ndarray, k_broad: int, k_local: int,
                          t: int = 1, kernel_scale: float = 0.05,
                          symmetrization: str = "mirror") -> np.ndarray:
    """Full SSNF pipeline from standardized DE / PSD feature matrices to A."""
    de_dm = distance_matrix(de)
    psd_dm = distance_matrix(psd)
    de_ssm, de_nssm = ssm_construction(de_dm, kneighbors(de_dm, k_broad),
                                       kernel_scale, symmetrization)
    psd_ssm, psd_nssm = ssm_construction(psd_dm, kneighbors(psd_dm, k_broad),
                                         kernel_scale, symmetrization)
    return ssm_fusion(de_ssm, psd_ssm, de_nssm, psd_nssm, k_local, t, symmetrization)


def full_kernel_ssm(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Dense (KNN-free) SSNF variant of the "SGCL w/o KNN" ablation (paper Table 10).

    The exponent reproduces the original implementation's operator precedence:
    the 1e-8/scale terms are added to, not divided into, the squared distances.
    """
    n1 = np.sum(x1 ** 2, axis=1, keepdims=True)
    n1 = n1 + n1.T - 2 * (x1 @ x1.T)
    s1 = -n1.mean() * 0.05
    n2 = np.sum(x2 ** 2, axis=1, keepdims=True)
    n2 = n2 + n2.T - 2 * (x2 @ x2.T)
    s2 = -n2.mean() * 0.05

    x1_ssm = np.exp(-n1 + 1e-8 / s1 + 1e-8)
    x2_ssm = np.exp(-n2 + 1e-8 / s2 + 1e-8)

    x1_nssm = normalize_ssm(x1_ssm)
    x2_nssm = normalize_ssm(x2_ssm)

    x1_fsm = x1_nssm @ x2_nssm @ x1_nssm
    x2_fsm = x2_nssm @ x1_nssm @ x2_nssm
    return (x1_fsm + x2_fsm) * 0.5


def build_adjacency(de: np.ndarray, psd: np.ndarray, g: dict) -> np.ndarray:
    """Adjacency dispatcher. `g["variant"]` selects the Table-10 KNN ablations:

    - "full" (default): the standard SSNF pipeline.
    - "no_knn": dense kernel SSMs, dense cross-diffusion (`full_kernel_ssm`).
    - "no_broad_knn": dense kernel SSMs fed into the local-KNN fusion
      ("SGCL w/o k1-NN"; reconstructed — no dedicated cell survives in the
      original notebooks).
    - "no_local_knn": broad-KNN SSMs fused by dense cross-diffusion
      ("SGCL w/o k2-NN"; reconstructed likewise).
    """
    variant = g.get("variant", "full")
    if variant == "full":
        return build_fused_adjacency(de, psd, g["k_broad"], g["k_local"],
                                     g["diffusion_steps"], g["kernel_scale"],
                                     g["symmetrization"])
    if variant == "no_knn":
        return full_kernel_ssm(de, psd)

    if variant == "no_broad_knn":
        # reconstruction: dense similarity with the same exp(-d/(s*mean(D)))
        # kernel as ssm_construction (no broad-KNN sparsification), then the
        # standard local-KNN fusion
        de_dm = distance_matrix(de)
        psd_dm = distance_matrix(psd)
        x1_ssm = np.exp(de_dm / (-de_dm.mean() * g["kernel_scale"]))
        x2_ssm = np.exp(psd_dm / (-psd_dm.mean() * g["kernel_scale"]))
        return ssm_fusion(x1_ssm, x2_ssm, normalize_ssm(x1_ssm), normalize_ssm(x2_ssm),
                          g["k_local"], g["diffusion_steps"], g["symmetrization"])

    if variant == "no_local_knn":
        de_dm = distance_matrix(de)
        psd_dm = distance_matrix(psd)
        _, de_nssm = ssm_construction(de_dm, kneighbors(de_dm, g["k_broad"]),
                                      g["kernel_scale"], g["symmetrization"])
        _, psd_nssm = ssm_construction(psd_dm, kneighbors(psd_dm, g["k_broad"]),
                                       g["kernel_scale"], g["symmetrization"])
        x1_fsm = de_nssm @ psd_nssm @ de_nssm
        x2_fsm = psd_nssm @ de_nssm @ psd_nssm
        return (x1_fsm + x2_fsm) * 0.5

    raise ValueError(f"unknown graph variant: {variant}")


def normalize_adj(adj: torch.Tensor, eye: torch.Tensor = None) -> torch.Tensor:
    """D^{-1/2} (A + I) D^{-1/2} on the (possibly edge-dropped) adjacency."""
    if eye is None:
        eye = torch.eye(adj.size(0), device=adj.device)
    tilde_adj = adj + eye
    norm_deg = torch.pow(tilde_adj.sum(dim=1), -0.5)
    return norm_deg.unsqueeze(1) * tilde_adj * norm_deg.unsqueeze(0)
