"""
EEG Graph Dataset for PyTorch Geometric.

Creates graph representations from EEG data where:
- Nodes: 83 EEG channels
- Node features: 5D spectral power (delta, theta, alpha, beta, gamma)
- Edges: Top-k strongest correlations per node
- Labels: Sleep stage (0-4)

Graphs are cached to disk after first creation for fast loading.
"""

import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
from typing import List, Optional
import hashlib

from config import DATA_DIR, N_CHANNELS

# Data paths
SPECTRAL_DIR = DATA_DIR / 'processed' / 'spectral_features'
CORRELATION_DIR = DATA_DIR / 'processed' / 'correlation_matrices'
LABELS_DIR = DATA_DIR / 'processed' / 'sleep_stage_labels'
CACHE_DIR = DATA_DIR / 'graph_cache'


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
        pre_transform: Optional[callable] = None,
        use_cache: bool = True,
        verbose: bool = True
    ):
        self.patient_ids = sorted(patient_ids)  # Sort for consistent cache key
        self.k = k
        self.use_cache = use_cache
        self.verbose = verbose

        self.spectral_dir = SPECTRAL_DIR
        self.corr_dir = CORRELATION_DIR
        self.labels_dir = LABELS_DIR

        # Store graph data
        self.graphs = []
        self.labels = []
        self.patient_indices = []  # Track which patient each graph belongs to

        super().__init__(None, transform, pre_transform)

        # Try to load from cache, otherwise build and cache
        if use_cache and self._load_from_cache():
            pass  # Loaded successfully
        else:
            self._load_data()
            if use_cache:
                self._save_to_cache()

    def _load_data(self):
        """Load all patient data and create graph representations."""
        if self.verbose:
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

                if self.verbose:
                    print(f"  {patient_id}: {num_epochs} epochs → {len([l for l in labels[:, 1] if l >= 0])} valid graphs")

            except FileNotFoundError as e:
                if self.verbose:
                    print(f"  {patient_id}: ERROR - Missing file: {e}")
            except Exception as e:
                if self.verbose:
                    print(f"  {patient_id}: ERROR - {e}")

        if self.verbose:
            print(f"\nTotal graphs loaded: {len(self.graphs)}")

            # Print label distribution
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            print(f"Label distribution:")
            stage_names = ['W', 'N1', 'N2', 'N3', 'R']
            for label, count in zip(unique_labels, counts):
                pct = count / len(self.labels) * 100
                print(f"  {stage_names[label]} ({label}): {count} ({pct:.1f}%)")

    def _get_cache_key(self) -> str:
        """Generate a unique cache key based on dataset parameters."""
        key_str = f"single_{'_'.join(self.patient_ids)}_k{self.k}"
        # Use hash to handle long patient lists
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self) -> Path:
        """Get the path to the cache file."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"single_{self._get_cache_key()}.pt"

    def _load_from_cache(self) -> bool:
        """Try to load dataset from cache. Returns True if successful."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False

        try:
            if self.verbose:
                print(f"Loading from cache: {cache_path.name}...", end=" ", flush=True)
            cached = torch.load(cache_path, weights_only=False)

            # Verify cache matches current parameters
            if (cached['k'] != self.k or
                cached['patient_ids'] != self.patient_ids):
                if self.verbose:
                    print("cache mismatch, rebuilding")
                return False

            self.graphs = cached['graphs']
            self.labels = cached['labels']
            self.patient_indices = cached['patient_indices']
            if self.verbose:
                print(f"done ({len(self.graphs)} graphs)")
            return True
        except Exception as e:
            if self.verbose:
                print(f"cache load failed ({e}), rebuilding")
            return False

    def _save_to_cache(self):
        """Save dataset to cache for fast loading next time."""
        cache_path = self._get_cache_path()
        if self.verbose:
            print(f"Saving to cache: {cache_path.name}...", end=" ", flush=True)
        try:
            torch.save({
                'patient_ids': self.patient_ids,
                'k': self.k,
                'graphs': self.graphs,
                'labels': self.labels,
                'patient_indices': self.patient_indices
            }, cache_path)
            if self.verbose:
                print("done")
        except Exception as e:
            if self.verbose:
                print(f"failed ({e})")

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


if __name__ == '__main__':
    """Test the dataset."""
    from pathlib import Path
    from config import load_patient_splits

    # Load canonical patient splits
    splits = load_patient_splits()
    train_ids = splits['train']
    val_ids = splits['val']
    test_ids = splits['test']

    print(f"Loaded patient splits:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients\n")

    # Create dataset for first 3 train patients
    test_patients = train_ids[:3]
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
