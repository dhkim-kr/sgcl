"""SGCL model: 2-layer GCN encoder, projection head, classifier, InfoNCE loss.

The contrastive machinery follows GRACE (Zhu et al., 2020); the classifier and
the contrastive loss both operate on the projection-head output, matching the
original experiments.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = adj @ (x @ self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class Encoder(nn.Module):
    """Shared graph encoder (paper Eq. (8)): two GCN layers with CELU."""

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int,
                 activation: nn.Module = None):
        super().__init__()
        self.gcn1 = GraphConvolution(in_channels, hid_channels, bias=True)
        self.gcn2 = GraphConvolution(hid_channels, out_channels, bias=True)
        self.activation = activation if activation is not None else nn.CELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.gcn1(x, adj))
        return self.activation(self.gcn2(x1, adj))


class CompareModel(nn.Module):
    """PCA+GCN baseline of the method ablation (paper Table 8, G1-G3):
    BatchNorm -> two GCN layers -> concatenated layer outputs -> linear classifier."""

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int,
                 num_classes: int):
        super().__init__()
        self.norm1d = nn.BatchNorm1d(in_channels)
        self.gcn1 = GraphConvolution(in_channels, hid_channels, bias=True)
        self.gcn2 = GraphConvolution(hid_channels, out_channels, bias=True)
        self.fc = nn.Linear(hid_channels + out_channels, num_classes)
        self.activation = nn.CELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        x = self.norm1d(x)
        h1 = self.activation(self.gcn1(x, adj))
        h2 = self.activation(self.gcn2(h1, adj))
        h = torch.cat((h1, h2), dim=1)
        return self.fc(h), h


class SGCL(nn.Module):
    """Encoder + projection head (paper Eq. (9)) + classifier + InfoNCE (Eq. (10))."""

    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 num_out: int, tau: float):
        super().__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)
        self.fc_classifier = nn.Linear(num_hidden, num_out)
        self.activation = nn.CELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(z)))

    def classification(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc_classifier(self.activation(z))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return F.normalize(z1) @ F.normalize(z2).t()

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(between_sim.diag()
                          / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor,
                         mean: bool = True) -> torch.Tensor:
        ret = (self.semi_loss(h1, h2) + self.semi_loss(h2, h1)) * 0.5
        return ret.mean() if mean else ret.sum()
