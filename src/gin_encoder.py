"""
gin_encoder.py - Graph Isomorphism Network (GIN) with Edge Attributes, SupCon, and Proxy-Anchor Loss
CS328: Introduction to Data Science
Author: Laxmidhar Panda (24110185)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_edges(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    p: float = 0.1
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Stochastic GraphCL Edge Dropout view augmentation.
    Randomly drops a fraction p of edges.
    """
    if p <= 0.0 or edge_index.size(1) == 0:
        return edge_index, edge_attr
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) >= p
    new_edge_index = edge_index[:, mask]
    new_edge_attr = edge_attr[mask] if edge_attr is not None else None
    return new_edge_index, new_edge_attr


def mask_features(x: torch.Tensor, p: float = 0.1, mask_val: int = 0) -> torch.Tensor:
    """
    Stochastic GraphCL Feature Masking view augmentation.
    Randomly replaces node tokens with a mask token.
    """
    if p <= 0.0:
        return x
    mask = torch.rand(x.size(0), device=x.device) < p
    new_x = x.clone()
    new_x[mask] = mask_val
    return new_x


class GINConvWithEdge(nn.Module):
    """
    Graph Isomorphism Network (GIN) convolution layer incorporating bond/edge attributes.
    Injective multiset aggregation proven maximally expressive (equivalent to 1-WL test):
        h_v^{(l)} = MLP^{(l)}( (1 + eps) * h_v^{(l-1)} + sum_{u in N(v)} ReLU(h_u^{(l-1)} + e_{uv}) )
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, eps_trainable: bool = True):
        super().__init__()
        if eps_trainable:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

        self.edge_proj = nn.Linear(edge_dim, in_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        row, col = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # Transform edge features to node dimension
        e_emb = self.edge_proj(edge_attr)
        msg = F.relu(x[row] + e_emb)

        # Injective multiset sum aggregation
        agg = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        agg.index_add_(0, col, msg)

        # Combine central node with aggregated neighborhood
        out = (1.0 + self.eps) * x + agg
        return self.mlp(out)


def global_add_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Sum pooling over graphs in a mini-batch (injective on multisets, captures atom counts)."""
    num_graphs = int(batch.max().item() + 1) if batch.numel() > 0 else 0
    out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    return out


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Mean pooling over graphs in a mini-batch (size-invariant, captures structural density)."""
    sum_pooled = global_add_pool(x, batch)
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    counts = global_add_pool(ones, batch).clamp(min=1.0)
    return sum_pooled / counts


class GINEncoder(nn.Module):
    """
    4-Layer GIN Encoder with bond attribute injection, dual sum+mean pooling,
    and hyperspherical L2 normalization onto S^{255}.
    """

    def __init__(
        self,
        num_atom_types: int = 28,
        num_bond_types: int = 5,
        hidden_dim: int = 64,
        proj_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.bond_embedding = nn.Embedding(num_bond_types, hidden_dim)
        self.dropout = dropout

        self.convs = nn.ModuleList([
            GINConvWithEdge(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Dual pooling concatenates global_add_pool (hidden_dim) and global_mean_pool (hidden_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward_representation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Extract un-normalized pooled representation before projection."""
        if x.dim() > 1 and x.size(1) == 1:
            x = x.squeeze(1)
        h = self.atom_embedding(x)

        if edge_attr.dim() > 1 and edge_attr.size(1) == 1:
            edge_attr = edge_attr.squeeze(1)
        e = self.bond_embedding(edge_attr)

        for conv in self.convs:
            h_new = conv(h, edge_index, e)
            h = F.relu(h + h_new)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        h_sum = global_add_pool(h, batch)
        h_mean = global_mean_pool(h, batch)
        return torch.cat([h_sum, h_mean], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Outputs strictly L2-normalized 256-D continuous embeddings."""
        h_graph = self.forward_representation(x, edge_index, edge_attr, batch)
        z_unnorm = self.projection_head(h_graph)
        z = F.normalize(z_unnorm, p=2, dim=-1)
        return z


class GINClassifier(nn.Module):
    """
    GIN encoder with a linear classification head for Cross-Entropy training (GIN-CE).
    Allows separating the encoder architecture from metric learning loss.
    """

    def __init__(self, encoder: GINEncoder, num_classes: int, proj_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch_size, num_classes)
            z: (batch_size, proj_dim) normalized embedding
        """
        z = self.encoder(x, edge_index, edge_attr, batch)
        logits = self.classifier(z)
        return logits, z


class SupConLoss(nn.Module):
    """
    Supervised Contrastive (SupCon) Loss (Khosla et al., NeurIPS 2020).
    Pulls graphs of the same structural class together on the hypersphere,
    while pushing different structural classes apart with temperature tau = 0.07.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        sim_matrix = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        pos_per_anchor = mask.sum(1)
        valid_anchors = pos_per_anchor > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / pos_per_anchor[valid_anchors]
        return -mean_log_prob_pos.mean()


class ProxyAnchorLoss(nn.Module):
    """
    Proxy-Anchor Loss for Deep Metric Learning (Kim et al., CVPR 2020).
    Introduces learnable class proxy vectors in the embedding space:
        L = (1/|P|) * sum_{p in P} log(1 + sum_{x in X_p^+} exp(-alpha * (s(x, p) - delta)))
          + (1/|C|) * sum_{p in C} log(1 + sum_{x in X_p^-} exp(alpha * (s(x, p) + delta)))
    """

    def __init__(self, num_classes: int, embedding_dim: int = 256, margin: float = 0.1, alpha: float = 32.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha

        # Learnable proxies: one vector per class, unit-normalized
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (B, d) normalized embeddings
        labels: (B,) class indices
        """
        P = F.normalize(self.proxies, p=2, dim=-1)
        cos = F.linear(features, P)  # shape (B, num_classes)

        device = features.device
        batch_size = features.size(0)

        one_hot = torch.zeros(batch_size, self.num_classes, device=device)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Classes with at least one sample in the batch
        has_pos = (one_hot.sum(dim=0) > 0)
        num_valid_proxies = has_pos.sum().float()

        if num_valid_proxies == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Positive proxy loss
        pos_term = torch.exp(-self.alpha * (cos - self.margin)) * one_hot
        pos_loss = torch.log(1.0 + pos_term.sum(dim=0))
        pos_loss = pos_loss[has_pos].mean()

        # Negative proxy loss
        neg_term = torch.exp(self.alpha * (cos + self.margin)) * (1.0 - one_hot)
        neg_loss = torch.log(1.0 + neg_term.sum(dim=0)).mean()

        return pos_loss + neg_loss
