"""
train.py
--------
Training loop and sample generation for the Brain-Connectome flow-matching
model (BrainDiffusionModel).

The model is trained to predict the flow-matching velocity field v(x_t, t)
using an L1 reconstruction loss.  Classifier-free guidance is applied at
inference time via the null-conditioning mechanism of ConditionEncoder.

References
----------
Paper, Algorithm 1 — Flow-Matching Training Procedure.
Paper, Section 3.4 — Classifier-Free Guidance.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from .dataset import BrainGraphDataset, vector_to_adjacency_matrix
from .model import BrainDiffusionModel, sample_brain_graph


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_model(model, dataset, device, epoch, global_mean, global_std):
    """
    Generate synthetic connectomes for the first 5 subjects and compute the
    average Pearson correlation against the real upper-triangular vectors.

    Args:
        model: Trained BrainDiffusionModel.
        dataset: BrainGraphDataset instance.
        device: Torch device.
        epoch (int): Current epoch (for logging only).
        global_mean, global_std: Dataset-level statistics used for rescaling.

    Returns:
        float: Mean Pearson correlation across the 5 subjects.
    """
    model.eval()
    static_node_features = (
        dataset.node_regions.to(device),
        dataset.node_lobes.to(device),
        dataset.node_surfaces.to(device),
        dataset.node_hemis.to(device),
        dataset.node_coords.to(device),
    )
    edge_indices = dataset.edge_indices.to(device)
    incidence_matrix = dataset.incidence_matrix.to(device)

    print(f"\n[Validation Epoch {epoch}] Generating 5 samples...")
    corrs = []
    for i in range(5):
        datum = dataset[i]
        real_vec = np.expm1(datum["x_0"].numpy())
        age_scalar = datum["age_ga"].item() * 100.0

        gen_vec = sample_brain_graph(
            model,
            target_age=age_scalar,
            device=device,
            node_feats=static_node_features,
            edge_indices=edge_indices,
            incidence_matrix=incidence_matrix,
            global_mean=global_mean,
            global_std=global_std,
            num_steps=50,
        )
        min_len = min(len(real_vec), len(gen_vec))
        c = np.corrcoef(real_vec[:min_len], gen_vec[:min_len])[0, 1]
        corrs.append(c)

    avg_corr = float(np.nanmean(corrs))
    print(f"[Validation] Average Pearson correlation: {avg_corr:.4f}")
    model.train()
    return avg_corr


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main():
    """
    Train BrainDiffusionModel on structural brain connectivity matrices.

    Hyperparameters match those reported in the paper (Table 3).
    Update the file paths below to point to your data directory.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Hyperparameters (paper Table 3) ---
    hidden_dim   = 128
    internal_dim = 2048
    batch_size   = 16
    LR           = 1e-4
    epochs       = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data paths ---
    node_csv      = "files/BrainNodeDefs.csv"
    metadata_csv  = "files/patient_metadata.csv"
    matrices_npy  = "files/matrices.npy"

    print("Loading dataset...")
    dataset = BrainGraphDataset(node_csv, metadata_csv, matrices_npy)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Compute global statistics for normalisation ---
    all_x0 = torch.cat([b["x_0"] for b in loader], dim=0)
    global_mean = all_x0.mean().to(device)
    global_std  = all_x0.std().to(device)

    # --- Static graph features (shared across all samples) ---
    static_node_features = (
        dataset.node_regions.to(device),
        dataset.node_lobes.to(device),
        dataset.node_surfaces.to(device),
        dataset.node_hemis.to(device),
        dataset.node_coords.to(device),
    )
    edge_indices     = dataset.edge_indices.to(device)
    incidence_matrix = dataset.incidence_matrix.to(device)

    # --- Model ---
    num_nodes = len(dataset.node_data)
    model = BrainDiffusionModel(
        num_nodes=num_nodes,
        num_regions=len(dataset.region_mapping),
        num_lobes=len(dataset.lobe_mapping),
        num_surfaces=len(dataset.surface_mapping),
        num_hemispheres=len(dataset.hemi_mapping),
        hidden_dim=hidden_dim,
        internal_dim=internal_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        steps_per_epoch=len(loader),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    save_fname = f"brain_diffusion_model_best_{hidden_dim}_{internal_dim}.pth"
    best_loss  = float("inf")

    # --- Training loop ---
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        batch_losses = []
        for batch in loader:
            x_0_raw = batch["x_0"].to(device)
            age_ga  = batch["age_ga"].to(device)

            # Normalise
            x_0 = (x_0_raw - global_mean) / global_std

            # Sample time t ~ U[0, 1] and noise x_1 ~ N(0, I)
            t   = torch.rand(x_0.shape[0], 1, device=device)
            x_1 = torch.randn_like(x_0)

            # Linear interpolation: x_t = t·x_1 + (1-t)·x_0
            x_t      = t * x_1 + (1 - t) * x_0
            target_v = x_1 - x_0  # target velocity field

            # Classifier-free guidance: randomly drop conditioning with p = 0.1
            force_null = np.random.rand() < 0.1

            pred_v = model(
                x_t, t, age_ga,
                static_node_features, edge_indices, incidence_matrix,
                force_null=force_null,
            )
            loss = F.l1_loss(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss.item())

        avg_loss = sum(batch_losses) / len(batch_losses)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")

        if (epoch + 1) % 50 == 0:
            validate_model(model, dataset, device, epoch + 1, global_mean, global_std)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_fname)

    # --- Final generation ---
    print("\nGenerating final samples...")
    model.eval()
    for i in range(10):
        print(f"  Sample {i+1}/10")
        gen_vec = sample_brain_graph(
            model,
            target_age=30,
            device=device,
            guidance_scale=3.0,
            node_feats=static_node_features,
            edge_indices=edge_indices,
            incidence_matrix=incidence_matrix,
            global_mean=global_mean,
            global_std=global_std,
            num_steps=100,
        )
        gen_mat = vector_to_adjacency_matrix(gen_vec, num_nodes)
        gen_mat[gen_mat < 0.01] = 0.0

    # Visualise first generated matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(gen_mat, cmap="viridis", vmax=np.percentile(gen_mat, 99))
    plt.title("Generated Brain Connectome")
    plt.colorbar()
    plt.savefig("generated_connectome.png")
    print("Saved generated_connectome.png")


if __name__ == "__main__":
    main()
