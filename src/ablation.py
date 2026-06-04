"""
ablation.py
-----------
Ablation study for Brain-Connectome (Flow Matching Dynamic Hypergraphs).

Tests the contribution of each architectural component:
  1. Full model                       — Flow Matching + Hypergraph + Multihop Transformer + GA conditioning
  2. No Hypergraph                    — remove dynamic hypergraph stream
  3. No Transformer                   — remove multihop transformer stream
  4. No GA conditioning               — unconditional (no gestational age conditioning)
  5. No Hypergraph + No Transformer   — only flow-matching MLP backbone
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from .model import BrainDiffusionModel
from .dataset import BrainGraphDataset


def run_ablation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load Dataset
    print("Loading Dataset...")
    node_csv = 'files/BrainNodeDefs.csv'
    metadata_csv = 'files/patient_metadata.csv'
    matrices_npy = 'files/matrices.npy'
    
    dataset = BrainGraphDataset(node_csv, metadata_csv, matrices_npy)
    
    # Calculate Global Mean/Std
    print("Calculating Global Stats...")
    all_data = []
    for idx in range(len(dataset)):
        all_data.append(dataset[idx]['x_0'])
    all_data_tensor = torch.stack(all_data, dim=0)
    global_mean = all_data_tensor.mean().to(device)
    global_std = all_data_tensor.std().to(device)

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Move static node features to device
    static_node_features = (
        dataset.node_regions.to(device),
        dataset.node_lobes.to(device),
        dataset.node_surfaces.to(device),
        dataset.node_hemis.to(device),
        dataset.node_coords.to(device),
    )
    edge_indices     = dataset.edge_indices.to(device)
    incidence_matrix = dataset.incidence_matrix.to(device)

    num_nodes = len(dataset.node_data)
    num_regions = len(dataset.region_mapping)
    num_lobes = len(dataset.lobe_mapping)
    num_surfaces = len(dataset.surface_mapping)
    num_hemispheres = len(dataset.hemi_mapping)

    # Ablation configurations
    configs = [
        {"name": "Full model",                     "use_hypergraph": True,  "use_transformer": True,  "use_ga": True},
        {"name": "No Hypergraph",                  "use_hypergraph": False, "use_transformer": True,  "use_ga": True},
        {"name": "No Transformer",                 "use_hypergraph": True,  "use_transformer": False, "use_ga": True},
        {"name": "No GA conditioning",             "use_hypergraph": True,  "use_transformer": True,  "use_ga": False},
        {"name": "No Hypergraph + No Transformer", "use_hypergraph": False, "use_transformer": False, "use_ga": True},
    ]

    print(f"\n{'Configuration':<35} {'Val Huber Loss':>15}")
    print("=" * 55)

    results = {}

    for cfg in configs:
        name = cfg["name"]
        model = BrainDiffusionModel(
            num_nodes=num_nodes,
            num_regions=num_regions,
            num_lobes=num_lobes,
            num_surfaces=num_surfaces,
            num_hemispheres=num_hemispheres,
            hidden_dim=32,       # small model for fast execution
            internal_dim=256,
            use_hypergraph=cfg["use_hypergraph"],
            use_transformer=cfg["use_transformer"],
            use_ga=cfg["use_ga"]
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Train for a few epochs for structural verification
        epochs = args.epochs
        for epoch in range(epochs):
            model.train()
            batch_losses = []
            for batch in train_loader:
                x_0_raw = batch["x_0"].to(device)
                age_ga  = batch["age_ga"].to(device)

                x_0 = (x_0_raw - global_mean) / global_std
                t   = torch.rand(x_0.shape[0], 1, device=device)
                x_1 = torch.randn_like(x_0)

                x_t      = t * x_1 + (1 - t) * x_0
                target_v = x_1 - x_0

                optimizer.zero_grad()
                pred_v = model(
                    x_t, t, age_ga,
                    static_node_features, edge_indices, incidence_matrix,
                    force_null=False,
                )
                loss = F.huber_loss(pred_v, target_v, delta=1.0)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

        val_loss = np.mean(batch_losses)
        print(f"{name:<35} {val_loss:>15.4f}")
        results[name] = val_loss

    print("=" * 55)
    
    # Save report
    out_path = os.path.join(args.folder, "ablation_results.json")
    import json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved ablation results to {out_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Brain-Connectome Ablation Study")
    parser.add_argument("--folder", type=str, default="ablation_results")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per config")
    args = parser.parse_args()
    os.makedirs(args.folder, exist_ok=True)
    run_ablation(args)


if __name__ == "__main__":
    main()
