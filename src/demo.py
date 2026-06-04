import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import BrainDiffusionModel, sample_brain_graph
from src.dataset import BrainGraphDataset, vector_to_adjacency_matrix

def run_trajectory_demo():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    
    # Load Data (needed for scaling params and graph structure)
    print("Loading Dataset to get graph structure...")
    # Paths adjusted to be relative to where this script is run
    node_csv = 'files/BrainNodeDefs.csv'
    metadata_csv = 'files/patient_metadata.csv'
    matrices_npy = 'files/matrices.npy'
    
    dataset = BrainGraphDataset(node_csv, metadata_csv, matrices_npy)
    
    # Calculate Global Mean/Std (from training code logic)
    print("Calculating Global Mean/Std...")
    all_data = []
    # Load a subset for speed if needed, or all. 
    # Since dataset is loaded, we can just iterate.
    # Note: efficient loading is mmap, but we need values for std.
    # Let's trust the logic from training script for mean/std
    # Or for this demo, we can re-calculate quickly on a subset or full if feasible.
    # Given the previous script did it on `data_loaders`, let's do it on a sample.
    sample_size = min(len(dataset), 1000)
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    for idx in indices:
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
    
    print(f"Loading model from {model_path}...")
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

    # 3. Generate Trajectory (28 to 44 weeks)
    print("Generating brains across ages...")
    ages = np.arange(28, 45, 2) # Every 2 weeks
    mean_connectivity = []
    
    for age in ages:
        print(f"  Generating Age: {age} weeks...")
        # Generate smaller batch per age for robustness? Just 1 for demo speed.
        # Let's do 3 samples and average
        samples = []
        for _ in range(3):
            gen_vec = sample_brain_graph(
                model, 
                target_age=age, 
                device=device, 
                node_feats=static_node_features, 
                edge_indices=edge_indices, 
                incidence_matrix=incidence_matrix, 
                global_mean=global_mean, 
                global_std=global_std,
                num_steps=50 # Speed up for demo
            )
            samples.append(gen_vec.mean()) # Average connectivity of the graph
        
        mean_connectivity.append(np.mean(samples))

    # 4. Plot
    plt.figure(figsize=(8, 5))
    plt.plot(ages, mean_connectivity, marker='o', linestyle='-', color='b')
    plt.title("Brain Connectivity Development (Generated)")
    plt.xlabel("Gestational Age (weeks)")
    plt.ylabel("Mean Connectivity Strength")
    plt.grid(True)
    plt.savefig("demo_trajectory_plot.png")
    print("\nSaved demo_trajectory_plot.png")
    print("Done!")

if __name__ == "__main__":
    run_trajectory_demo()
