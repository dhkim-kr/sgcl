"""Uniform graph augmentation (paper Sec. 3.4.1, Eqs. (6)-(7)).

Feature masking is element-wise Bernoulli with keep probability 1 - p_f;
edge dropping samples a keep mask on the strict upper triangle with keep
probability 1 - p_e, symmetrizes it, and always keeps the diagonal.
"""

import torch


def build_feature_probs(feature: torch.Tensor, p_f: float) -> torch.Tensor:
    return torch.full(feature.shape, 1 - p_f, device=feature.device)


def build_edge_probs(adj: torch.Tensor, p_e: float) -> torch.Tensor:
    return torch.triu(torch.full(adj.shape, 1 - p_e, device=adj.device), diagonal=1)


def uniform_drop_features(features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    keep_mask = torch.bernoulli(probs).to(torch.bool)
    return features.where(keep_mask, torch.zeros_like(features))


def uniform_drop_edges(edge_weights: torch.Tensor, probs: torch.Tensor,
                       eye: torch.Tensor) -> torch.Tensor:
    keep_mask = torch.bernoulli(probs)
    keep_mask = keep_mask + keep_mask.T + eye
    return edge_weights.where(keep_mask.to(torch.bool), torch.zeros_like(edge_weights))
