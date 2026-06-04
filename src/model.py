import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class Hypergraph(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim)
        self.hyperedge_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, incidence_matrix):

        transformed_nodes = self.theta(x)

        if incidence_matrix.dim() == 2:
            H_T = incidence_matrix.transpose(0, 1)
        else:
            H_T = incidence_matrix.transpose(1, 2)

        dim_to_sum = 1 if incidence_matrix.dim() == 2 else 2
        edge_degrees = H_T.sum(dim=dim_to_sum, keepdim=True).clamp(min=1.0)

        hyperedge_feats = torch.matmul(H_T, transformed_nodes) / edge_degrees

        edge_emb = self.hyperedge_mlp(hyperedge_feats)

        dim_to_sum_node = 1 if incidence_matrix.dim() == 2 else 2
        node_degrees = incidence_matrix.sum(dim=dim_to_sum_node, keepdim=True).clamp(min=1.0)

        updated_node_feats = torch.matmul(incidence_matrix, edge_emb) / node_degrees

        return x + updated_node_feats


class MultiHopNodeEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, num_hops=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim * 2,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hops)

    def forward(self, x):
        is_2d = x.dim() == 2
        if is_2d:
            x = x.unsqueeze(0)

        out = self.transformer_encoder(x)

        if is_2d:
            return out.squeeze(0)
        return out


class ConditionEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.time_ga = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.null_embedding = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self, age_ga_norm, force_null=False):
        if force_null:
            return self.null_embedding.expand(age_ga_norm.shape[0], -1)
        return self.time_ga(age_ga_norm)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class NodeStructureEmbedding(nn.Module):
    def __init__(self, num_regions, num_lobes, num_surfaces, num_hemispheres, embedding_dim=64):
        super().__init__()
        self.region_emb = nn.Embedding(num_regions, embedding_dim)
        self.lobe_emb = nn.Embedding(num_lobes, embedding_dim)
        self.surface_emb = nn.Embedding(num_surfaces, embedding_dim)
        self.hemi_emb = nn.Embedding(num_hemispheres, embedding_dim)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, region_ids, lobe_ids, surface_ids, hemi_ids, coords):
        return (self.region_emb(region_ids) +
                self.lobe_emb(lobe_ids) +
                self.surface_emb(surface_ids) +
                self.hemi_emb(hemi_ids) +
                self.coord_mlp(coords))


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class BrainDiffusionModel(nn.Module):
    def __init__(self, num_nodes, num_regions, num_lobes, num_surfaces, num_hemispheres, hidden_dim=128, internal_dim=2048,
                 use_hypergraph=True, use_transformer=True, use_ga=True):
        super().__init__()
        self.data_dim = int(num_nodes * (num_nodes - 1) // 2)
        self.use_hypergraph = use_hypergraph
        self.use_transformer = use_transformer
        self.use_ga = use_ga

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_encoder = ConditionEncoder(hidden_dim)

        self.node_structure_emb = NodeStructureEmbedding(
            num_regions, num_lobes, num_surfaces, num_hemispheres, embedding_dim=hidden_dim
        )
        self.multihop_node_emb = MultiHopNodeEmbedding(hidden_dim, num_hops=2)
        self.hypergraph = Hypergraph(hidden_dim, hidden_dim)

        self.edge_projector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.internal_dim = internal_dim
        input_total = self.data_dim + hidden_dim + hidden_dim

        self.input_proj = nn.Linear(input_total, self.internal_dim)
        self.layers = nn.ModuleList([
            ResidualBlock(self.internal_dim),
            ResidualBlock(self.internal_dim),
            ResidualBlock(self.internal_dim),
            ResidualBlock(self.internal_dim),
        ])
        self.output_proj = nn.Linear(self.internal_dim, self.data_dim)

    def forward(self, x_t, t, age_ga_norm, node_feats, edge_indices, incidence_matrix, force_null=False):
        t_emb = self.time_mlp(t)
        
        # If use_ga is False, we bypass continuous GA conditioning
        if not self.use_ga:
            ga_emb = self.cond_encoder(age_ga_norm, force_null=True)
        else:
            ga_emb = self.cond_encoder(age_ga_norm, force_null=force_null)

        regions, lobes, surfaces, hemis, coords = node_feats

        raw_node_embs = self.node_structure_emb(regions, lobes, surfaces, hemis, coords)

        # If use_transformer is False, skip multihop transformer
        if self.use_transformer:
            refined_node_embs = self.multihop_node_emb(raw_node_embs)
        else:
            refined_node_embs = raw_node_embs

        # If use_hypergraph is False, skip dynamic hypergraph layer
        if self.use_hypergraph:
            hyper_node_embs = self.hypergraph(refined_node_embs, incidence_matrix)
        else:
            hyper_node_embs = refined_node_embs

        u_inds = edge_indices[:, 0]
        v_inds = edge_indices[:, 1]

        if hyper_node_embs.dim() == 3:
            node_u = hyper_node_embs[:, u_inds, :]
            node_v = hyper_node_embs[:, v_inds, :]
            edge = torch.cat([node_u, node_v], dim=-1)
            structure_bias = self.edge_projector(edge).squeeze(-1)
        else:
            node_u = hyper_node_embs[u_inds]
            node_v = hyper_node_embs[v_inds]
            edge = torch.cat([node_u, node_v], dim=-1)
            structure_bias = self.edge_projector(edge).squeeze(-1)

        combined_input = torch.cat([x_t, t_emb, ga_emb], dim=-1)
        x = self.input_proj(combined_input)
        for layer in self.layers:
            x = layer(x)
        output = self.output_proj(x)

        return output + structure_bias




# BrainGraphDataset and vector_to_adjacency_matrix live in dataset.py.
# Training loop and generation live in train.py.
# This file contains only the NN architecture modules.

@torch.no_grad()
def sample_brain_graph(
    model,
    target_age: float,
    device,
    node_feats,
    edge_indices,
    incidence_matrix,
    global_mean,
    global_std,
    guidance_scale: float = 3.0,
    num_steps: int = 100,
) -> np.ndarray:
    """
    Generate a synthetic brain connectome via Euler integration of the
    flow-matching ODE with classifier-free guidance.

    The ODE is integrated from t = 1 (noise) to t = 0 (data):
        x_{t+dt} = x_t + v_guided(x_t, t) · dt
    where:
        v_guided = v_uncond + guidance_scale · (v_cond - v_uncond)

    Classifier-free guidance is implemented by comparing the model's
    conditional prediction (with GA conditioning) against the unconditional
    prediction (null embedding, force_null=True).

    A dynamic 60th-percentile threshold is applied post-generation to match
    the biological sparsity of real connectomes (~60% zero entries).

    Args:
        model: Trained BrainDiffusionModel in eval mode.
        target_age (float): Gestational age in weeks (unnormalised).
        device: Torch device.
        node_feats: Tuple of static node feature tensors.
        edge_indices: Upper-triangular edge index tensor.
        incidence_matrix: Biological hypergraph incidence matrix.
        global_mean, global_std: Dataset-level normalisation statistics.
        guidance_scale (float): Classifier-free guidance strength (paper: 3.0).
        num_steps (int): Number of Euler integration steps.

    Returns:
        np.ndarray: Flat upper-triangular connectome vector (non-negative).
    """
    model.eval()
    ga_tensor = torch.tensor([[target_age / 100.0]], dtype=torch.float32).to(device)
    x = torch.randn(1, model.data_dim).to(device)
    timesteps = torch.linspace(1, 0, num_steps).to(device)

    for i in range(num_steps - 1):
        t_curr = timesteps[i].view(1, 1)
        t_next = timesteps[i + 1].view(1, 1)
        dt = t_next - t_curr

        v_cond   = model(x, t_curr, ga_tensor, node_feats, edge_indices, incidence_matrix, force_null=False)
        v_uncond = model(x, t_curr, ga_tensor, node_feats, edge_indices, incidence_matrix, force_null=True)
        pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        x = x + pred_v * dt

    x_unscaled = x * global_std + global_mean
    x_reconstructed = np.expm1(x_unscaled.cpu().numpy().flatten())
    x_reconstructed = np.maximum(x_reconstructed, 0)

    # Biological sparsity enforcement: real connectomes are ~60% zeros.
    threshold = np.percentile(x_reconstructed, 60)
    x_reconstructed[x_reconstructed < threshold] = 0.0

    return x_reconstructed
