# Configuration for Brain Connectivity Diffusion Model
# Multi-Objective Loss Function Experiments

import torch

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Paths ---
DATA_DIR = "data/files"
NODE_CSV = f"{DATA_DIR}/BrainNodeDefs.csv"
METADATA_CSV = f"{DATA_DIR}/patient_metadata.csv"
MATRICES_NPY = f"{DATA_DIR}/matrices.npy"

# --- Model Architecture ---
NUM_NODES = 90
HIDDEN_DIM = 32
INTERNAL_DIM = 256

# --- Training ---
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 3e-4
GRADIENT_CLIP = 1.0
PATIENCE = 30  # Early stopping: stop if no improvement for N epochs

# --- Loss Function Selection ---
# Options: "l1", "spectral", "laplacian", "sparse", "combined"
LOSS_TYPE = "l1"

# --- Loss Weights (for combined loss) ---
# NOTE: Keep these small! L1 must dominate, auxiliary losses are regularizers.
ALPHA_SPECTRAL = 0.001   # Weight for spectral loss (reduced from 0.1)
BETA_LAPLACIAN = 0.0005  # Weight for Laplacian smoothness (reduced from 0.05)
GAMMA_SPARSE = 0.0       # Weight for sparsity (not used in combined by default)

# --- Spectral Loss Config ---
SPECTRAL_TOP_K = 10  # Number of eigenvalues to compare

# --- Sparsity Loss Config ---
SPARSITY_ZERO_WEIGHT = 0.3    # Weight for errors on zero-valued connections
SPARSITY_NONZERO_WEIGHT = 1.0 # Weight for errors on actual connections

# --- Output ---
OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
EXPERIMENT_DIR = "experiments/loss_comparison"

# --- Diffusion ---
NUM_DIFFUSION_STEPS = 100  # For training
NUM_SAMPLE_STEPS = 50      # For generation (faster)
CFG_SCALE = 2.0            # Classifier-free guidance scale

# --- Validation ---
VALIDATE_EVERY_N_EPOCHS = 50
NUM_VALIDATION_SAMPLES = 5
