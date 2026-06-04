import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from generation_code_fixed import BrainDiffusionModel, BrainGraphDataset, sample_brain_graph

def compute_laplacian_eigenvalues(adj_matrix):
    """
    Computes the eigenvalues of the Normalized Laplacian.
    This captures global topology (clustering, paths, modularity).
    """
    # Ensure symmetry
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    # Binarize/Threshold slightly to ensure stability (common in brain graphs)
    # adj_matrix[adj_matrix < 0.1] = 0
    
    # Degree matrix
    degrees = np.sum(adj_matrix, axis=1)
    # Avoid division by zero
    degrees[degrees == 0] = 1e-10
    
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    
    # Normalized Laplacian: L = I - D^-1/2 * A * D^-1/2
    identity = np.eye(adj_matrix.shape[0])
    laplacian = identity - d_inv_sqrt @ adj_matrix @ d_inv_sqrt
    
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return np.sort(eigenvalues)

def validate_spectral(num_samples=20):
    print("--- SPECTRAL VALIDATION (Topology Check) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data & Model (Copying logic from previous script)
    node_csv = 'files/BrainNodeDefs.csv'
    metadata_csv = 'files/patient_metadata.csv'
    matrices_npy = 'files/matrices.npy'
    dataset = BrainGraphDataset(node_csv, metadata_csv, matrices_npy)
    
    # Model
    model = BrainDiffusionModel(
        len(dataset.node_data),
        len(dataset.region_mapping),
        len(dataset.lobe_mapping),
        len(dataset.surface_mapping),
        len(dataset.hemi_mapping),
        32, internal_dim=256
    ).to(device)
    model.load_state_dict(torch.load("brain_diffusion_model_best_32_256.pth", map_location=device))
    model.eval()

    # Features
    static_node_features = (
        dataset.node_regions.to(device), dataset.node_lobes.to(device),
        dataset.node_surfaces.to(device), dataset.node_hemis.to(device),
        dataset.node_coords.to(device)
    )
    edge_indices = dataset.edge_indices.to(device)
    incidence_matrix = dataset.incidence_matrix.to(device)
    
    # Global Stats
    idxs = np.random.choice(len(dataset), 500, replace=False)
    all_data = [dataset[i]['x_0'] for i in idxs]
    global_mean = torch.cat(all_data).mean().to(device)
    global_std = torch.cat(all_data).std().to(device)

    # Loop
    real_spectra = []
    gen_spectra = []
    
    print(f"Comparing Spectra for {num_samples} brains...")
    
    for i in range(num_samples):
        # Pick random patient
        idx = np.random.randint(len(dataset))
        datum = dataset[idx]
        
        # Real Matrix
        x_raw = datum['x_0'].numpy()
        real_flat = np.expm1(x_raw)
        real_mat = np.zeros((90,90))
        r, c = np.triu_indices(90, k=1)
        real_mat[r, c] = real_flat
        real_mat = real_mat + real_mat.T
        
        # Generated Matrix
        target_age = datum['age_ga'].item() * 100.0
        gen_vec = sample_brain_graph(model, target_age, device, static_node_features, edge_indices, incidence_matrix, global_mean, global_std, num_steps=50)
        gen_mat = np.zeros((90,90))
        gen_mat[r, c] = gen_vec
        gen_mat = gen_mat + gen_mat.T
        
        # Compute Spectra
        real_spectra.append(compute_laplacian_eigenvalues(real_mat))
        gen_spectra.append(compute_laplacian_eigenvalues(gen_mat))

    # Average Spectra
    avg_real_spec = np.mean(np.stack(real_spectra), axis=0)
    avg_gen_spec = np.mean(np.stack(gen_spectra), axis=0)
    
    # Distance Metric (Wasserstein / Earth Mover's Distance)
    spectral_dist = wasserstein_distance(avg_real_spec, avg_gen_spec)
    
    print(f"\nSpectral Distance (Lower is Better): {spectral_dist:.4f}")
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(avg_real_spec, label='Real Spectrum', linewidth=2, color='blue')
    plt.plot(avg_gen_spec, label='Generated Spectrum', linewidth=2, linestyle='--', color='red')
    plt.fill_between(range(90), avg_real_spec, avg_gen_spec, color='gray', alpha=0.2)
    plt.title(f"Graph Topology Comparison (Spectral Distance: {spectral_dist:.4f})")
    plt.xlabel("Eigenvalue Index (0 to 89)")
    plt.ylabel("Eigenvalue Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("spectral_validation.png")
    print("Saved spectral_validation.png")

if __name__ == "__main__":
    validate_spectral()
