"""
EEG Graph Dataset for PyTorch Geometric.

Creates graph representations from EEG data where:
- Nodes: 83 EEG channels
- Node features: 5D spectral power (delta, theta, alpha, beta, gamma)
- Edges: Top-k strongest correlations per node
- Labels: Sleep stage (0-4)
"""

import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
from typing import List, Optional

from config import DATA_DIR, N_CHANNELS

# Data paths
SPECTRAL_DIR = DATA_DIR / 'processed' / 'spectral_features'
CORRELATION_DIR = DATA_DIR / 'processed' / 'correlation_matrices'
LABELS_DIR = DATA_DIR / 'processed' / 'sleep_stage_labels'


class EEGGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for EEG graphs.

    Each graph represents one 30-second epoch of EEG data:
    - Nodes: 83 EEG channels
    - Node features: 5D spectral power per frequency band
    - Edges: Top-k strongest correlations between channels
    - Label: Sleep stage (W=0, N1=1, N2=2, N3=3, R=4)

    Args:
        patient_ids: List of patient IDs (e.g., ['P01', 'P02', ...])
        k: Number of top correlations to keep per node (default: 10)
        transform: Optional transform to apply to graphs
        pre_transform: Optional pre-transform to apply to graphs
    """

    def __init__(
        self,
        patient_ids: List[str],
        k: int = 10,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None
    ):
        self.patient_ids = patient_ids
        self.k = k

        self.spectral_dir = SPECTRAL_DIR
        self.corr_dir = CORRELATION_DIR
        self.labels_dir = LABELS_DIR

        # Store graph data
        self.graphs = []
        self.labels = []
        self.patient_indices = []  # Track which patient each graph belongs to

        super().__init__(None, transform, pre_transform)

        # Load all data
        self._load_data()

    def _load_data(self):
        """Load all patient data and create graph representations."""
        print(f"Loading EEG graph dataset for {len(self.patient_ids)} patients (k={self.k})...")

        for patient_idx, patient_id in enumerate(self.patient_ids):
            try:
                # Load spectral features: (num_epochs, 83, 5)
                spectral_file = self.spectral_dir / f'{patient_id}_spectral_features.npy'
                spectral_features = np.load(spectral_file)

                # Load correlation matrices: (num_epochs, 3403) - upper triangle
                corr_file = self.corr_dir / f'{patient_id}_corr.npy'
                corr_matrices = np.load(corr_file)

                # Load labels: (num_epochs, 2) - [epoch_idx, label]
                label_file = self.labels_dir / f'{patient_id}_labels.npy'
                labels = np.load(label_file)

                # Handle variable-length recordings
                num_epochs = min(len(spectral_features), len(corr_matrices), len(labels))

                # Create graph for each epoch
                for epoch_idx in range(num_epochs):
                    # Get label for this epoch
                    label = labels[epoch_idx, 1]

                    # Skip invalid labels (L = -1)
                    if label < 0:
                        continue

                    # Node features: (83, 5)
                    node_features = spectral_features[epoch_idx]

                    # Reconstruct full correlation matrix from upper triangle
                    corr_upper = corr_matrices[epoch_idx]
                    corr_matrix = self._upper_to_matrix(corr_upper)

                    # Create edge_index from top-k correlations per node
                    edge_index, edge_attr = self._create_edges(corr_matrix, self.k)

                    # Create PyG Data object
                    graph = Data(
                        x=torch.FloatTensor(node_features),
                        edge_index=torch.LongTensor(edge_index),
                        edge_attr=torch.FloatTensor(edge_attr),
                        y=torch.LongTensor([label]),
                        patient_id=patient_id,
                        epoch_idx=epoch_idx
                    )

                    # Apply pre-transform if provided
                    if self.pre_transform is not None:
                        graph = self.pre_transform(graph)

                    self.graphs.append(graph)
                    self.labels.append(int(label))
                    self.patient_indices.append(patient_idx)

                print(f"  {patient_id}: {num_epochs} epochs → {len([l for l in labels[:, 1] if l >= 0])} valid graphs")

            except FileNotFoundError as e:
                print(f"  {patient_id}: ERROR - Missing file: {e}")
            except Exception as e:
                print(f"  {patient_id}: ERROR - {e}")

        print(f"\nTotal graphs loaded: {len(self.graphs)}")

        # Print label distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print(f"Label distribution:")
        stage_names = ['W', 'N1', 'N2', 'N3', 'R']
        for label, count in zip(unique_labels, counts):
            pct = count / len(self.labels) * 100
            print(f"  {stage_names[label]} ({label}): {count} ({pct:.1f}%)")

    def _upper_to_matrix(self, upper_triangle: np.ndarray) -> np.ndarray:
        """
        Convert upper triangle vector to full symmetric correlation matrix.

        Args:
            upper_triangle: (3403,) array containing upper triangle of 83x83 matrix

        Returns:
            (83, 83) symmetric correlation matrix
        """
        n = N_CHANNELS
        matrix = np.zeros((n, n), dtype=np.float32)

        # Fill upper triangle (excluding diagonal)
        triu_idx = np.triu_indices(n, k=1)
        matrix[triu_idx] = upper_triangle

        # Make symmetric
        matrix = matrix + matrix.T

        # Diagonal is 1.0 (self-correlation)
        np.fill_diagonal(matrix, 1.0)

        return matrix

    def _create_edges(self, corr_matrix: np.ndarray, k: int):
        """
        Create edge_index and edge_attr by selecting top-k correlations per node.

        Args:
            corr_matrix: (83, 83) correlation matrix
            k: Number of top correlations per node

        Returns:
            edge_index: (2, num_edges) edge list
            edge_attr: (num_edges,) edge weights (absolute correlation values)
        """
        n_nodes = corr_matrix.shape[0]
        edges = []
        weights = []

        for i in range(n_nodes):
            # Get correlations for node i (excluding self)
            correlations = np.abs(corr_matrix[i].copy())
            correlations[i] = -1  # Exclude self-loops

            # Get top-k indices
            top_k_indices = np.argsort(correlations)[-k:]

            # Add edges with their weights
            for j in top_k_indices:
                edges.append([i, j])
                weights.append(correlations[j])

        # Convert to edge_index format: (2, num_edges)
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(weights, dtype=np.float32)

        return edge_index, edge_attr

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        """
        Get a single graph by index.

        Args:
            idx: Index of the graph to retrieve

        Returns:
            PyG Data object containing the graph
        """
        graph = self.graphs[idx]

        # Apply transform if provided
        if self.transform is not None:
            graph = self.transform(graph)

        return graph

    def get_patient_graphs(self, patient_id: str) -> List[Data]:
        """
        Get all graphs for a specific patient.

        Args:
            patient_id: Patient ID (e.g., 'P01')

        Returns:
            List of PyG Data objects for that patient
        """
        return [g for g in self.graphs if g.patient_id == patient_id]

    def get_stage_graphs(self, stage: int) -> List[Data]:
        """
        Get all graphs for a specific sleep stage.

        Args:
            stage: Sleep stage (0=W, 1=N1, 2=N2, 3=N3, 4=R)

        Returns:
            List of PyG Data objects for that stage
        """
        return [g for g, l in zip(self.graphs, self.labels) if l == stage]


def split_by_patient(
    all_patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """
    Split patient IDs into train/val/test sets.

    Args:
        all_patient_ids: List of all patient IDs
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        seed: Random seed for reproducibility

    Returns:
        (train_ids, val_ids, test_ids) tuple of patient ID lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    np.random.seed(seed)
    n_patients = len(all_patient_ids)

    # Shuffle patient IDs
    shuffled_ids = np.random.permutation(all_patient_ids).tolist()

    # Split
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


if __name__ == '__main__':
    """Test the dataset."""
    from pathlib import Path

    # Get all patient IDs
    all_files = sorted(SPECTRAL_DIR.glob("P*_spectral_features.npy"))
    all_patient_ids = [f.name.split('_')[0] for f in all_files]

    print(f"Found {len(all_patient_ids)} patients: {all_patient_ids[:5]}...\n")

    # Split patients
    train_ids, val_ids, test_ids = split_by_patient(all_patient_ids)
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test\n")

    # Create dataset for first 3 patients
    test_patients = all_patient_ids[:3]
    dataset = EEGGraphDataset(test_patients, k=10)

    print(f"\nDataset size: {len(dataset)}")

    # Test getting a graph
    graph = dataset[0]
    print(f"\nFirst graph:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Node features shape: {graph.x.shape}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Edge features shape: {graph.edge_attr.shape}")
    print(f"  Label: {graph.y.item()}")
    print(f"  Patient: {graph.patient_id}")
    print(f"  Epoch: {graph.epoch_idx}")
