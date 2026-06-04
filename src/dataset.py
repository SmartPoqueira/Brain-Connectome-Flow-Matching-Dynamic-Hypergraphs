"""
dataset.py
----------
BrainGraphDataset: loads structural connectome matrices, node metadata,
and builds the biological hypergraph incidence matrix for the
Brain-Connectome flow-matching model.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BrainGraphDataset(Dataset):
    """
    Dataset of brain structural connectivity matrices.

    Each sample corresponds to one subject.  The upper-triangular entries of
    the connectivity matrix are extracted (log1p-transformed) as the data
    vector x_0, and the normalised gestational age (GA / 100) is provided
    as the conditioning signal.

    The biological hypergraph incidence matrix H (Nodes × Hyperedges) is
    built once from lobe and hemisphere membership to encode anatomical
    structure.

    Args:
        node_csv (str): Path to CSV with columns RegionName, Lobe, Surface,
            Hemisphere, X, Y, Z.
        metadata_csv (str): Path to CSV with column AgeGA.
        matrices_npy (str): Path to .npy array of shape (N, n_nodes, n_nodes).
    """

    def __init__(self, node_csv: str, metadata_csv: str, matrices_npy: str) -> None:
        self.node_data = pd.read_csv(node_csv)
        self.metadata  = pd.read_csv(metadata_csv)
        self.matrices  = np.load(matrices_npy, mmap_mode="r")

        # Build categorical mappings
        self.region_mapping  = {k: v for v, k in enumerate(self.node_data["RegionName"].unique())}
        self.lobe_mapping    = {k: v for v, k in enumerate(self.node_data["Lobe"].unique())}
        self.node_data["Surface"] = self.node_data["Surface"].fillna("unknown")
        self.surface_mapping = {k: v for v, k in enumerate(self.node_data["Surface"].unique())}
        self.hemi_mapping    = {"Left": 0, "Right": 1}

        # Static node feature tensors (shared across all samples)
        self.node_regions  = torch.tensor(
            [self.region_mapping[r] for r in self.node_data["RegionName"]], dtype=torch.long)
        self.node_lobes    = torch.tensor(
            [self.lobe_mapping[l]  for l in self.node_data["Lobe"]],        dtype=torch.long)
        self.node_surfaces = torch.tensor(
            [self.surface_mapping[s] for s in self.node_data["Surface"]],   dtype=torch.long)
        self.node_hemis    = torch.tensor(
            [self.hemi_mapping[h]  for h in self.node_data["Hemisphere"]],  dtype=torch.long)
        self.node_coords   = torch.tensor(
            self.node_data[["X", "Y", "Z"]].values, dtype=torch.float32)

        # Upper-triangular edge index (i < j) — used to vectorise the matrix
        dummy = np.zeros((len(self.node_data), len(self.node_data)))
        rows, cols = np.triu_indices_from(dummy, k=1)
        self.edge_indices = torch.tensor(np.stack([rows, cols], axis=1), dtype=torch.long)

        # Biological hypergraph
        self.incidence_matrix = self._build_hypergraph()

    # ------------------------------------------------------------------
    # Hypergraph construction
    # ------------------------------------------------------------------

    def _build_hypergraph(self) -> torch.Tensor:
        """
        Build the incidence matrix H of shape (Nodes, Hyperedges).

        Hyperedges encode anatomical groupings:
          - One hyperedge per lobe  (e.g., all Frontal nodes)
          - One hyperedge per hemisphere (Left / Right)

        This encodes the biological prior that structurally connected regions
        tend to belong to the same lobe or hemisphere.
        """
        num_nodes = len(self.node_data)
        columns = []

        # Lobe hyperedges
        for lobe in self.node_data["Lobe"].unique():
            col = np.zeros(num_nodes)
            col[self.node_data.index[self.node_data["Lobe"] == lobe].tolist()] = 1.0
            columns.append(col)

        # Hemisphere hyperedges
        for hemi in self.node_data["Hemisphere"].unique():
            col = np.zeros(num_nodes)
            col[self.node_data.index[self.node_data["Hemisphere"] == hemi].tolist()] = 1.0
            columns.append(col)

        H = np.stack(columns, axis=1)  # (Nodes, Hyperedges)
        print(f"Biological hypergraph: {H.shape[1]} hyperedges (lobes + hemispheres).")
        return torch.tensor(H, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        matrix = np.array(self.matrices[idx])
        rows, cols = np.triu_indices_from(matrix, k=1)
        x_0 = torch.tensor(np.log1p(matrix[rows, cols]), dtype=torch.float32)
        age_ga_norm = torch.tensor([row["AgeGA"] / 100.0], dtype=torch.float32)
        return {"x_0": x_0, "age_ga": age_ga_norm}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def vector_to_adjacency_matrix(vector: np.ndarray, num_nodes: int) -> np.ndarray:
    """
    Reconstruct a symmetric adjacency matrix from an upper-triangular vector.

    Args:
        vector: Flat array of length n_nodes*(n_nodes-1)/2.
        num_nodes: Number of graph nodes.

    Returns:
        Symmetric (num_nodes, num_nodes) matrix.
    """
    matrix = np.zeros((num_nodes, num_nodes))
    rows, cols = np.triu_indices(num_nodes, k=1)
    matrix[rows, cols] = vector
    return matrix + matrix.T
