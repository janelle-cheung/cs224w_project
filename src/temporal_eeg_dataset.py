"""
Temporal EEG Graph Dataset for PyTorch Geometric.

Creates spatio-temporal graph representations from EEG data where:
- Nodes: 83 * n_temporal_steps (channels replicated across time)
- Node features: 5 spectral power bands per node
- Spatial edges: Top-k strongest correlations within each epoch (between channels at same time)
- Temporal edges: Channel i at time t → channel i at time t+1 (bidirectional, weight=1.0)
- Labels: Sleep stage (0-4)

Node indexing: node_idx = t * 83 + channel_idx
  - t=0: nodes 0-82
  - t=1: nodes 83-165
  - t=2: nodes 166-248
  etc.

Only stacks consecutive epochs with the same sleep stage label.

Graphs are cached to disk after first creation for fast loading.
"""

import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from pathlib import Path
from typing import List, Optional, Tuple
import hashlib

from config import DATA_DIR, N_CHANNELS

# Data paths
SPECTRAL_DIR = DATA_DIR / 'processed' / 'spectral_features'
CORRELATION_DIR = DATA_DIR / 'processed' / 'correlation_matrices'
LABELS_DIR = DATA_DIR / 'processed' / 'sleep_stage_labels'
CACHE_DIR = DATA_DIR / 'graph_cache'


class TemporalEEGGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for temporal EEG graphs.

    Each graph represents n_temporal_steps consecutive 30-second epochs
    with the same sleep stage:
    - Nodes: 83 * n_temporal_steps (channels at each time step)
    - Node features: 5 spectral power bands
    - Spatial edges: Top-k correlations between channels (within each epoch)
    - Temporal edges: Channel i at t ↔ channel i at t+1 (bidirectional, weight=1.0)
    - Label: Sleep stage (W=0, N1=1, N2=2, N3=3, R=4)

    Node indexing: node_idx = t * 83 + channel_idx

    Args:
        patient_ids: List of patient IDs (e.g., ['P01', 'P02', ...])
        n_temporal_steps: Number of consecutive epochs to stack (default: 3)
        k: Number of top correlations to keep per node for spatial edges (default: 10)
        transform: Optional transform to apply to graphs
        pre_transform: Optional pre-transform to apply to graphs
    """

    def __init__(
        self,
        patient_ids: List[str],
        n_temporal_steps: int = 3,
        k: int = 10,
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        use_cache: bool = True,
        verbose: bool = True
    ):
        self.patient_ids = sorted(patient_ids)  # Sort for consistent cache key
        self.n_temporal_steps = n_temporal_steps
        self.k = k
        self.use_cache = use_cache
        self.verbose = verbose

        self.spectral_dir = SPECTRAL_DIR
        self.corr_dir = CORRELATION_DIR
        self.labels_dir = LABELS_DIR

        # Store graph data
        self.graphs = []
        self.labels = []
        self.patient_indices = []

        super().__init__(None, transform, pre_transform)

        # Try to load from cache, otherwise build and cache
        if use_cache and self._load_from_cache():
            pass  # Loaded successfully
        else:
            self._load_data()
            if use_cache:
                self._save_to_cache()

    def _load_data(self):
        """Load all patient data and create temporal graph representations."""
        if self.verbose:
            print(f"Loading Temporal EEG dataset for {len(self.patient_ids)} patients "
                  f"(n_steps={self.n_temporal_steps}, k={self.k})...")

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

                # Find consecutive epoch windows with same label
                windows = self._find_valid_windows(labels[:num_epochs, 1])

                n_graphs = 0
                for start_idx, label in windows:
                    # Stack node features across time: (n_steps * 83, 5)
                    node_features = self._stack_features(
                        spectral_features, start_idx, self.n_temporal_steps
                    )

                    # Create spatial edges from correlations (with time-shifted node indices)
                    spatial_edges, spatial_weights = self._create_spatial_edges(
                        corr_matrices, start_idx, self.n_temporal_steps
                    )

                    # Create temporal edges (bidirectional between time steps)
                    temporal_edges, temporal_weights = self._create_temporal_edges()

                    # Combine spatial and temporal edges
                    edge_index = np.concatenate([spatial_edges, temporal_edges], axis=1)
                    edge_attr = np.concatenate([spatial_weights, temporal_weights])

                    # Create PyG Data object
                    graph = Data(
                        x=torch.FloatTensor(node_features),
                        edge_index=torch.LongTensor(edge_index),
                        edge_attr=torch.FloatTensor(edge_attr),
                        y=torch.LongTensor([label]),
                        patient_id=patient_id,
                        start_epoch=start_idx,
                        n_temporal_steps=self.n_temporal_steps
                    )

                    # Apply pre-transform if provided
                    if self.pre_transform is not None:
                        graph = self.pre_transform(graph)

                    self.graphs.append(graph)
                    self.labels.append(int(label))
                    self.patient_indices.append(patient_idx)
                    n_graphs += 1

                if self.verbose:
                    print(f"  {patient_id}: {num_epochs} epochs → {n_graphs} temporal graphs")

            except FileNotFoundError as e:
                print(f"  {patient_id}: ERROR - Missing file: {e}")
            except Exception as e:
                print(f"  {patient_id}: ERROR - {e}")

        if self.verbose:
            print(f"\nTotal temporal graphs loaded: {len(self.graphs)}")

            # Print label distribution
            if self.labels:
                unique_labels, counts = np.unique(self.labels, return_counts=True)
                print(f"Label distribution:")
                stage_names = ['W', 'N1', 'N2', 'N3', 'R']
                for label, count in zip(unique_labels, counts):
                    pct = count / len(self.labels) * 100
                    print(f"  {stage_names[label]} ({label}): {count} ({pct:.1f}%)")

    def _get_cache_key(self) -> str:
        """Generate a unique cache key based on dataset parameters."""
        key_str = f"temporal_{'_'.join(self.patient_ids)}_t{self.n_temporal_steps}_k{self.k}"
        # Use hash to handle long patient lists
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self) -> Path:
        """Get the path to the cache file."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"temporal_{self._get_cache_key()}.pt"

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
            if (cached['n_temporal_steps'] != self.n_temporal_steps or
                cached['k'] != self.k or
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
                'n_temporal_steps': self.n_temporal_steps,
                'k': self.k,
                'graphs': self.graphs,
                'labels': self.labels,
                'patient_indices': self.patient_indices
            }, cache_path)
            if self.verbose:
                print("done")
        except Exception as e:
            print(f"failed ({e})")

    def _find_valid_windows(self, labels: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find all valid windows of n_temporal_steps consecutive epochs
        with the same valid sleep stage label.

        Args:
            labels: (num_epochs,) array of sleep stage labels

        Returns:
            List of (start_idx, label) tuples for valid windows
        """
        windows = []
        n_steps = self.n_temporal_steps
        num_epochs = len(labels)

        i = 0
        while i <= num_epochs - n_steps:
            label = labels[i]

            # Skip invalid labels
            if label < 0:
                i += 1
                continue

            # Check if next n_steps-1 epochs have the same label
            valid_window = True
            for j in range(1, n_steps):
                if labels[i + j] != label:
                    valid_window = False
                    break

            if valid_window:
                windows.append((i, int(label)))
                # Move by 1 to allow overlapping windows (or by n_steps for non-overlapping)
                i += 1
            else:
                i += 1

        return windows

    def _stack_features(
        self,
        spectral_features: np.ndarray,
        start_idx: int,
        n_steps: int
    ) -> np.ndarray:
        """
        Stack spectral features across temporal steps.

        Each channel at each time step becomes a separate node.
        Node indexing: node_idx = t * 83 + channel_idx

        Args:
            spectral_features: (num_epochs, 83, 5) array
            start_idx: Starting epoch index
            n_steps: Number of temporal steps

        Returns:
            (n_steps * 83, 5) array of node features
        """
        # Get features for each time step: list of (83, 5) arrays
        features_list = [
            spectral_features[start_idx + t] for t in range(n_steps)
        ]

        # Stack along node dimension: (n_steps * 83, 5)
        stacked = np.vstack(features_list)

        return stacked.astype(np.float32)

    def _create_spatial_edges(
        self,
        corr_matrices: np.ndarray,
        start_idx: int,
        n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial edges from top-k correlations within each epoch.

        Edges are offset by time step: node_idx = t * 83 + channel_idx

        Args:
            corr_matrices: (num_epochs, 3403) upper triangle correlations
            start_idx: Starting epoch index
            n_steps: Number of temporal steps

        Returns:
            edge_index: (2, num_edges) array
            edge_attr: (num_edges,) array of correlation weights
        """
        n_channels = N_CHANNELS
        all_edges = []
        all_weights = []

        for t in range(n_steps):
            corr_upper = corr_matrices[start_idx + t]
            corr_matrix = self._upper_to_matrix(corr_upper)

            # Get top-k edges for this time step (in channel-space 0-82)
            edges, weights = self._topk_edges(corr_matrix, self.k)

            # Offset edges by time step: node_idx = t * 83 + channel_idx
            offset = t * n_channels
            edges_offset = edges + offset

            all_edges.append(edges_offset)
            all_weights.append(weights)

        # Combine all spatial edges
        edge_index = np.concatenate(all_edges, axis=1)
        edge_attr = np.concatenate(all_weights)

        return edge_index, edge_attr

    def _create_temporal_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create bidirectional temporal edges connecting channel i at time t
        to channel i at time t+1.

        Node indexing: node_idx = t * 83 + channel_idx
        Edge format: source nodes at t, target nodes at t+1 (and vice versa)

        Returns:
            edge_index: (2, num_temporal_edges) array
            edge_attr: (num_temporal_edges,) array of weights (all 1.0)
        """
        n_channels = N_CHANNELS
        n_steps = self.n_temporal_steps

        sources = []
        targets = []

        # For each time step transition (t -> t+1)
        for t in range(n_steps - 1):
            # Node indices at time t: [t*83, t*83+1, ..., t*83+82]
            # Node indices at time t+1: [(t+1)*83, (t+1)*83+1, ..., (t+1)*83+82]
            offset_t = t * n_channels
            offset_t1 = (t + 1) * n_channels

            for i in range(n_channels):
                node_t = offset_t + i
                node_t1 = offset_t1 + i

                # Forward edge: t -> t+1
                sources.append(node_t)
                targets.append(node_t1)

                # Backward edge: t+1 -> t
                sources.append(node_t1)
                targets.append(node_t)

        edge_index = np.array([sources, targets], dtype=np.int64)
        edge_attr = np.ones(len(sources), dtype=np.float32)

        return edge_index, edge_attr

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

    def _topk_edges(self, corr_matrix: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
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
    """Test the temporal dataset."""
    # Get all patient IDs
    all_files = sorted(SPECTRAL_DIR.glob("P*_spectral_features.npy"))
    all_patient_ids = [f.name.split('_')[0] for f in all_files]

    print(f"Found {len(all_patient_ids)} patients: {all_patient_ids[:5]}...\n")

    # Create temporal dataset for first 3 patients
    n_temporal_steps = 3
    k = 10
    test_patients = all_patient_ids[:3]
    dataset = TemporalEEGGraphDataset(test_patients, n_temporal_steps=n_temporal_steps, k=k)

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Test getting a graph
        graph = dataset[0]
        print(f"\nFirst temporal graph:")
        print(f"  Nodes: {graph.num_nodes}")
        print(f"  Node features shape: {graph.x.shape}")
        print(f"  Edges: {graph.num_edges}")
        print(f"  Edge features shape: {graph.edge_attr.shape}")
        print(f"  Label: {graph.y.item()}")
        print(f"  Patient: {graph.patient_id}")
        print(f"  Start epoch: {graph.start_epoch}")
        print(f"  Temporal steps: {graph.n_temporal_steps}")

        # Verify dimensions
        n_channels = N_CHANNELS
        expected_nodes = n_temporal_steps * n_channels
        expected_features = 5  # 5 spectral bands
        expected_spatial_edges = n_temporal_steps * n_channels * k  # k edges per channel per time step
        expected_temporal_edges = (n_temporal_steps - 1) * n_channels * 2  # bidirectional

        print(f"\nExpected values:")
        print(f"  Nodes: {expected_nodes} (got {graph.num_nodes})")
        print(f"  Features per node: {expected_features} (got {graph.x.shape[1]})")
        print(f"  Spatial edges: {expected_spatial_edges}")
        print(f"  Temporal edges: {expected_temporal_edges}")
        print(f"  Total edges: {expected_spatial_edges + expected_temporal_edges} (got {graph.num_edges})")

        # Verify edge indices are valid
        max_node_idx = graph.edge_index.max().item()
        print(f"\n  Max node index in edges: {max_node_idx} (should be < {expected_nodes})")
