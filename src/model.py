import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
import math
from torch.optim.lr_scheduler import OneCycleLR
import os

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
    def __init__(self, num_nodes, num_regions, num_lobes, num_surfaces, num_hemispheres, hidden_dim=128, internal_dim=2048):
        super().__init__()
        self.data_dim = int(num_nodes * (num_nodes - 1) // 2)

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
        ga_emb = self.cond_encoder(age_ga_norm, force_null=force_null)

        regions, lobes, surfaces, hemis, coords = node_feats

        raw_node_embs = self.node_structure_emb(regions, lobes, surfaces, hemis, coords)

        refined_node_embs = self.multihop_node_emb(raw_node_embs)

        hyper_node_embs = self.hypergraph(refined_node_embs, incidence_matrix)

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


class BrainGraphDataset(Dataset):
    def __init__(self, node_csv, metadata_csv, matrices_npy):
        self.node_data = pd.read_csv(node_csv)
        self.metadata = pd.read_csv(metadata_csv)
        self.matrices = np.load(matrices_npy, mmap_mode='r')

        self.region_mapping = {k: v for v, k in enumerate(self.node_data['RegionName'].unique())}
        self.lobe_mapping = {k: v for v, k in enumerate(self.node_data['Lobe'].unique())}
        self.node_data['Surface'] = self.node_data['Surface'].fillna('unknown')
        self.surface_mapping = {k: v for v, k in enumerate(self.node_data['Surface'].unique())}
        self.hemi_mapping = {'Left': 0, 'Right': 1}

        self.node_regions = torch.tensor([self.region_mapping[r] for r in self.node_data['RegionName']],
                                         dtype=torch.long)
        self.node_lobes = torch.tensor([self.lobe_mapping[l] for l in self.node_data['Lobe']], dtype=torch.long)
        self.node_surfaces = torch.tensor([self.surface_mapping[s] for s in self.node_data['Surface']],
                                          dtype=torch.long)
        self.node_hemis = torch.tensor([self.hemi_mapping[h] for h in self.node_data['Hemisphere']], dtype=torch.long)
        self.node_coords = torch.tensor(self.node_data[['X', 'Y', 'Z']].values, dtype=torch.float32)

        dummy_matrix = np.zeros((len(self.node_data), len(self.node_data)))
        rows, cols = np.triu_indices_from(dummy_matrix, k=1)
        self.edge_indices = torch.tensor(np.stack([rows, cols], axis=1), dtype=torch.long)

        self.incidence_matrix = self.load_hypergraph_structure()

    def __len__(self):
        return len(self.metadata)


    def load_hypergraph_structure(self):
        """
        Builds a biologically meaningful incidence matrix H (Nodes x Hyperedges).
        Hyperedges will represent:
        1. Lobes (e.g., all Frontal nodes form one hyperedge)
        2. Hemispheres (Left / Right)
        3. Surfaces (if available)
        """
        num_nodes = len(self.node_data)
        
        # 1. Lobe Hyperedges
        unique_lobes = self.node_data['Lobe'].unique()
        lobe_edges = []
        for lobe in unique_lobes:
            # Create a column of zeros
            col = np.zeros(num_nodes)
            # Set 1 for nodes belonging to this lobe
            indices = self.node_data.index[self.node_data['Lobe'] == lobe].tolist()
            col[indices] = 1.0
            lobe_edges.append(col)
            
        # 2. Hemisphere Hyperedges
        unique_hemis = self.node_data['Hemisphere'].unique()
        hemi_edges = []
        for hemi in unique_hemis:
            col = np.zeros(num_nodes)
            indices = self.node_data.index[self.node_data['Hemisphere'] == hemi].tolist()
            col[indices] = 1.0
            hemi_edges.append(col)
            
        # Stack them: (Num_Hyperedges x Num_Nodes) -> Transpose to (Nodes x Hyperedges)
        all_edges = lobe_edges + hemi_edges
        H_np = np.stack(all_edges, axis=1) # Shape: (Nodes, Num_Hyperedges)
        
        print(f"Built Biological Hypergraph with {H_np.shape[1]} hyperedges (Lobes + Hemispheres).")
        
        return torch.tensor(H_np, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        matrix_patient = np.array(self.matrices[idx])
        upper_tri = np.triu_indices_from(matrix_patient, k=1)
        raw_vals = matrix_patient[upper_tri]
        x_0 = torch.tensor(np.log1p(raw_vals), dtype=torch.float32)

        age_ga_norm = torch.tensor([row['AgeGA'] / 100.00], dtype=torch.float32)

        return {
            'x_0': x_0,
            'age_ga': age_ga_norm,
        }


def vector_to_adjacency_matrix(vector, num_nodes):
    matrix = np.zeros((num_nodes, num_nodes))
    rows, cols = np.triu_indices(num_nodes, k=1)
    matrix[rows, cols] = vector
    matrix = matrix + matrix.T
    return matrix


@torch.no_grad()
def sample_brain_graph(model, target_age, device, node_feats, edge_indices, incidence_matrix, global_mean, global_std,
                       guidance_scale=3.0, num_steps=100):
    """
    Euler integration of the flow-matching ODE: x_{t+dt} = x_t + v(x_t, t) * dt
    where v is the learned velocity field. Uses classifier-free guidance:
        v_guided = v_uncond + scale * (v_cond - v_uncond)
    """
    model.eval()
    ga_tensor = torch.tensor([[target_age / 100.0]], dtype=torch.float32).to(device)
    x = torch.randn(1, model.data_dim).to(device)

    timesteps = torch.linspace(1, 0, num_steps).to(device)

    for i in range(num_steps - 1):
        t_curr = timesteps[i].view(1, 1)
        t_next = timesteps[i + 1].view(1, 1)
        dt = t_next - t_curr

        v_cond = model(x, t_curr, ga_tensor, node_feats, edge_indices, incidence_matrix, force_null=False)
        v_uncond = model(x, t_curr, ga_tensor, node_feats, edge_indices, incidence_matrix, force_null=True)

        pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)

        x = x + pred_v * dt

    x_unscaled = x * global_std + global_mean
    x_reconstructed = np.expm1(x_unscaled.cpu().numpy().flatten())
    x_reconstructed = np.maximum(x_reconstructed, 0)
    
    # --- BIAS FIX: Dynamic Thresholding ---
    # Real brains are sparse (~60% zeros). Diffusion models produce small noise instead of zeros.
    # We set the bottom 60% of values to 0 to match biological sparsity.
    percentile_threshold = np.percentile(x_reconstructed, 60)
    x_reconstructed[x_reconstructed < percentile_threshold] = 0.0
    
    return x_reconstructed


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    hidden_dim = 32
    epochs = 500
    internal_dim = 256
    batch_size = 16
    LR = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node_csv = 'files/BrainNodeDefs.csv'
    metadata_csv = 'files/patient_metadata.csv'
    matrices_npy = 'files/matrices.npy'

    print("Loading Dataset...")
    dataset = BrainGraphDataset(node_csv, metadata_csv, matrices_npy)
    data_loaders = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    all_data = []
    for batch in data_loaders:
        all_data.append(batch['x_0'])
    all_data_tensor = torch.cat(all_data, dim=0)
    global_mean = all_data_tensor.mean().to(device)
    global_std = all_data_tensor.std().to(device)

    num_nodes = len(dataset.node_data)

    model = BrainDiffusionModel(
        num_nodes,
        len(dataset.region_mapping),
        len(dataset.lobe_mapping),
        len(dataset.surface_mapping),
        len(dataset.hemi_mapping),
        hidden_dim,
        internal_dim=internal_dim
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


    static_node_features = (
        dataset.node_regions.to(device),
        dataset.node_lobes.to(device),
        dataset.node_surfaces.to(device),
        dataset.node_hemis.to(device),
        dataset.node_coords.to(device)
    )
    edge_indices = dataset.edge_indices.to(device)
    incidence_matrix = dataset.incidence_matrix.to(device)

    print("Start training loop")
    save_model_fname = f"brain_diffusion_model_best_{hidden_dim}_{internal_dim}.pth"
    
    # Validation Helper
    def validate_model(model, dataset, device, epoch, global_mean, global_std):
        model.eval()
        print(f"\n[Validation Epoch {epoch}] Generating samples...")
        generated_matrices = []
        real_matrices = []
        
        # Taking a few real samples directly from dataset for comparison
        # Just use the first 5 samples from the dataset
        for i in range(5):
            datum = dataset[i]
            x_raw = datum['x_0'].numpy()
            # expm1 to reverse log1p
            real_mat_flat = np.expm1(x_raw)
            real_matrices.append(real_mat_flat)
            
            # Generate conditional on same age
            age_scalar = datum['age_ga'].item() * 100.0
            
            gen_vec = sample_brain_graph(
                model, 
                target_age=age_scalar, 
                device=device, 
                node_feats=static_node_features,
                edge_indices=edge_indices,
                incidence_matrix=incidence_matrix,
                global_mean=global_mean,
                global_std=global_std,
                num_steps=50 # Faster sampling for val
            )
            # sample_brain_graph returns reconstructed flat vector (upper tri) or matrix?
            # Looking at sample_brain_graph implementation: it returns x_reconstructed which is FLATTENED 
            # at line 293: x_reconstructed = np.expm1(...).flatten()
            generated_matrices.append(gen_vec)
            
        # Calculate Correlation
        corrs = []
        for r, g in zip(real_matrices, generated_matrices):
            # Ensure lengths match
            min_len = min(len(r), len(g))
            c = np.corrcoef(r[:min_len], g[:min_len])[0,1]
            corrs.append(c)
            
        avg_corr = np.mean(corrs)
        print(f"[Validation] Average Correlation with Real Data: {avg_corr:.4f}")
        model.train()
        return avg_corr

    best_loss = 1e10
    
    # Use existing best model if available logic was here, but let's train from scratch to see effect
    model.train()
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(data_loaders),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    
    for epoch in range(epochs):
        batch_loss = []
        for batch in data_loaders:
            x_0_raw = batch['x_0'].to(device)
            age_ga = batch['age_ga'].to(device)

            x_0 = (x_0_raw - global_mean) / global_std

            t = torch.rand(x_0.shape[0], 1).to(device)
            x_1 = torch.randn_like(x_0).to(device)
            x_t = t * x_1 + (1 - t) * x_0
            target_v = x_1 - x_0
            force_null = np.random.rand() < 0.1

            pred_v = model(x_t, t, age_ga, static_node_features, edge_indices, incidence_matrix, force_null=force_null)

            loss = F.l1_loss(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batch_loss.append(loss.item())

        avg_loss = sum(batch_loss) / len(batch_loss)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % 50 == 0:
            val_corr = validate_model(model, dataset, device, epoch+1, global_mean, global_std)
        
        if best_loss > avg_loss:
            best_loss = avg_loss
            # Only save occasionally to save time/IO or if significantly better
            torch.save(model.state_dict(), save_model_fname)
            # print(f"  -> New best model saved with loss {best_loss:.4f}")

    print("Generating final samples...")
    model.eval()
    
    generated_dataset = []
    # Generate fewer for quick test
    for i in range(10): 
        print(f"Generating sample {i+1}/10...")
        generated_vector = sample_brain_graph(
            model,
            target_age=30,
            device=device,
            guidance_scale=3.0,
            node_feats=static_node_features,
            edge_indices=edge_indices,
            incidence_matrix=incidence_matrix,
            global_mean=global_mean,
            global_std=global_std,
            num_steps=100
        )
        # Reconstruct Matrix
        generated_matrix = vector_to_adjacency_matrix(generated_vector, num_nodes)
        
        # Thresholding similar to original code
        generated_matrix[generated_matrix < 0.01] = 0 # Simple threshold
        
        generated_dataset.append(generated_matrix)

    data_to_plot = generated_dataset[0]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(data_to_plot, cmap='viridis', vmax=np.percentile(data_to_plot, 99))
    plt.title("Generated Graph (Fixed Kat Model)")
    plt.colorbar()
    plt.savefig("kat_fixed_generation.png")
    print("Saved kat_fixed_generation.png")
