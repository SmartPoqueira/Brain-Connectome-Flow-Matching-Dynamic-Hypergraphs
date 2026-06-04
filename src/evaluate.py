import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
from generation_code_fixed import BrainDiffusionModel, BrainGraphDataset, sample_brain_graph, vector_to_adjacency_matrix

def validate_and_report(num_samples=50):
    """
    Generates 'num_samples' brains matching the ages of real patients
    and compares them to the ground truth.
    """
    # 1. Setup
    print("--- Starting Validation Report ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Data
    print("Loading Dataset...")
    node_csv = 'files/BrainNodeDefs.csv'
    metadata_csv = 'files/patient_metadata.csv'
    matrices_npy = 'files/matrices.npy'
    
    dataset = BrainGraphDataset(node_csv, metadata_csv, matrices_npy)
    
    # Global Stats (needed for reconstruction)
    all_data = []
    # Use a subset to calc stats quickly if dataset is huge, but here it's fine.
    # We'll trust the model was trained with stats from the full set.
    # Let's recalculate on the fly for correctness.
    print("Calculating Global Stats...")
    idxs = np.random.choice(len(dataset), min(1000, len(dataset)), replace=False)
    for idx in idxs:
        all_data.append(dataset[idx]['x_0'])
    all_data_tensor = torch.cat(all_data, dim=0)
    global_mean = all_data_tensor.mean().to(device)
    global_std = all_data_tensor.std().to(device)

    # 2. Load Model
    hidden_dim = 32
    internal_dim = 256
    model_path = "brain_diffusion_model_best_32_256.pth"
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
    
    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Static features
    static_node_features = (
        dataset.node_regions.to(device),
        dataset.node_lobes.to(device),
        dataset.node_surfaces.to(device),
        dataset.node_hemis.to(device),
        dataset.node_coords.to(device)
    )
    edge_indices = dataset.edge_indices.to(device)
    incidence_matrix = dataset.incidence_matrix.to(device)

    # 3. Validation Loop
    print(f"\nValidating on {num_samples} random samples...")
    
    # Randomly select indices from dataset
    test_indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    correlations = []
    mses = []
    
    real_vals_all = []
    gen_vals_all = []
    
    gt_means = []
    gen_means = []

    for i, idx in enumerate(test_indices):
        datum = dataset[idx]
        
        # Real Data
        x_raw = datum['x_0'].numpy()
        real_vec = np.expm1(x_raw) # Flat vector
        real_vals_all.extend(real_vec)
        gt_means.append(np.mean(real_vec))
        
        # Target Age
        target_age = datum['age_ga'].item() * 100.0
        
        # Generate
        # print(f"Sample {i+1}/{num_samples}: Generating for Age {target_age:.1f}...")
        gen_vec = sample_brain_graph(
            model, 
            target_age=target_age, 
            device=device, 
            node_feats=static_node_features, 
            edge_indices=edge_indices, 
            incidence_matrix=incidence_matrix, 
            global_mean=global_mean, 
            global_std=global_std,
            num_steps=50 # Faster sampling
        )
        gen_vals_all.extend(gen_vec)
        gen_means.append(np.mean(gen_vec))
        
        # Metrics
        # Correlation
        if len(real_vec) != len(gen_vec):
            min_len = min(len(real_vec), len(gen_vec))
            real_vec = real_vec[:min_len]
            gen_vec = gen_vec[:min_len]
            
        corr, _ = pearsonr(real_vec, gen_vec)
        correlations.append(corr)
        
        # MSE
        mse = np.mean((real_vec - gen_vec)**2)
        mses.append(mse)

    # 4. Report & Plots
    avg_corr = np.mean(correlations)
    avg_mse = np.mean(mses)
    
    print("\n" + "="*30)
    print("   VALIDATION RESULTS")
    print("="*30)
    print(f"Num Samples: {num_samples}")
    print(f"Avg Pearson Correlation: {avg_corr:.4f}")
    print(f"Avg MSE: {avg_mse:.4f}")
    print("="*30)

    # Plot 1: Correlation Distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(correlations, kde=True, color='blue', bins=15)
    plt.title("Distribution of Sample Correlations")
    plt.xlabel("Pearson Correlation (r)")
    plt.axvline(avg_corr, color='r', linestyle='--', label=f'Mean: {avg_corr:.2f}')
    plt.legend()

    # Plot 2: Global Connectivity Comparison (Did we capture the density?)
    plt.subplot(1, 2, 2)
    plt.scatter(gt_means, gen_means, alpha=0.6, color='purple')
    
    # Diagonal line
    min_val = min(min(gt_means), min(gen_means))
    max_val = max(max(gt_means), max(gen_means))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.title("Mean Connectivity: Real vs Generated")
    plt.xlabel("Real Mean Connectivity")
    plt.ylabel("Generated Mean Connectivity")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("validation_report_1.png")
    print("Saved plot: validation_report_1.png")
    
    # Plot 3: Value Distribution (Real vs Fake)
    plt.figure(figsize=(8, 5))
    # Downsample for plotting speed if needed
    sns.kdeplot(np.array(real_vals_all)[::10], label='Real Data', fill=True, alpha=0.3, color='green')
    sns.kdeplot(np.array(gen_vals_all)[::10], label='Generated Data', fill=True, alpha=0.3, color='red')
    plt.title("Connectivity Value Distribution")
    plt.xlabel("Connectivity Strength")
    plt.xlim(0, 5) # Zoom in on frequent values
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("validation_report_dist.png")
    print("Saved plot: validation_report_dist.png")

if __name__ == "__main__":
    validate_and_report(num_samples=20) # 20 samples for quick demo, user can increase
