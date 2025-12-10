"""
GNN Models for EEG Graph Encoding.

Implements graph neural network encoders that map EEG graphs to embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch


class GNNEncoder(nn.Module):
    """
    GNN Encoder using Graph Convolutional Networks (GCN).

    Architecture:
    - Multiple GCN layers with ReLU activation
    - Batch normalization after each layer
    - Dropout for regularization
    - Global mean pooling to get graph-level embedding
    - Final linear projection to embedding dimension

    Args:
        input_dim: Input node feature dimension (default: 5 for spectral bands)
        hidden_dim: Hidden layer dimension (default: 128)
        embedding_dim: Output embedding dimension (default: 64)
        n_layers: Number of GCN layers (default: 3)
        dropout: Dropout rate (default: 0.5)
        pooling: Global pooling method ('mean', 'max', 'add') (default: 'mean')
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.5,
        pooling: str = 'mean'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.pooling = pooling

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Embedding projection
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GNN encoder.

        Args:
            data: PyG Data or Batch object with:
                - x: Node features (num_nodes, input_dim)
                - edge_index: Edge indices (2, num_edges)
                - batch: Batch assignment (num_nodes,) - auto-created if single graph

        Returns:
            embeddings: (batch_size, embedding_dim) graph embeddings
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # If single graph (no batch attribute), create batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GCN layers with batch norm, ReLU, and dropout
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling: (num_nodes, hidden_dim) -> (batch_size, hidden_dim)
        x = self.pool(x, batch)

        # Project to embedding dimension
        embedding = self.embedding_layer(x)

        return embedding

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GNNClassifier(nn.Module):
    """
    GNN Encoder with classification head.

    Combines GNNEncoder with a classification layer for sleep stage prediction.

    Args:
        input_dim: Input node feature dimension (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        embedding_dim: Embedding dimension (default: 64)
        n_layers: Number of GCN layers (default: 3)
        n_classes: Number of output classes (default: 5 for sleep stages)
        dropout: Dropout rate (default: 0.5)
        pooling: Global pooling method (default: 'mean')
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 3,
        n_classes: int = 5,
        dropout: float = 0.5,
        pooling: str = 'mean'
    ):
        super().__init__()

        self.encoder = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            dropout=dropout,
            pooling=pooling
        )

        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, data: Data, return_embedding: bool = False):
        """
        Forward pass through encoder and classifier.

        Args:
            data: PyG Data or Batch object
            return_embedding: If True, also return embeddings

        Returns:
            If return_embedding is False:
                logits: (batch_size, n_classes) classification logits
            If return_embedding is True:
                (logits, embeddings): tuple of logits and embeddings
        """
        # Get embeddings
        embedding = self.encoder(data)

        # Classify
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding
        else:
            return logits

    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get only embeddings without classification.

        Args:
            data: PyG Data or Batch object

        Returns:
            embeddings: (batch_size, embedding_dim)
        """
        return self.encoder(data)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GATEncoder(nn.Module):
    """
    GNN Encoder using Graph Attention Networks (GAT).

    Similar to GNNEncoder but uses attention mechanism.

    Args:
        input_dim: Input node feature dimension (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        embedding_dim: Output embedding dimension (default: 64)
        n_layers: Number of GAT layers (default: 3)
        heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.5)
        pooling: Global pooling method (default: 'mean')
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
        pooling: str = 'mean'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.pooling = pooling

        # GAT layers
        self.convs = nn.ModuleList()
        # First layer: input_dim -> hidden_dim * heads
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))

        # Middle layers: hidden_dim * heads -> hidden_dim * heads
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        # Last layer: hidden_dim * heads -> hidden_dim (concatenate=False for final layer)
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        # Batch normalization
        self.bns = nn.ModuleList()
        for i in range(n_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Embedding projection
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through GAT encoder."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GAT layers
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)  # GAT typically uses ELU
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = self.pool(x, batch)

        # Project to embedding
        embedding = self.embedding_layer(x)

        return embedding

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test the models."""
    print("Testing GNN Models\n" + "=" * 60)

    # Create dummy graph data
    num_nodes = 83
    input_dim = 5
    num_edges = 100

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    data = Data(x=x, edge_index=edge_index)

    print(f"Graph: {num_nodes} nodes, {num_edges} edges")
    print(f"Node features: {x.shape}\n")

    # Test GNNEncoder
    print("=" * 60)
    print("Testing GNNEncoder")
    print("=" * 60)

    encoder = GNNEncoder(
        input_dim=5,
        hidden_dim=128,
        embedding_dim=64,
        n_layers=3,
        dropout=0.5
    )

    print(f"Model parameters: {encoder.get_num_params():,}")

    encoder.eval()
    with torch.no_grad():
        embedding = encoder(data)

    print(f"Output embedding shape: {embedding.shape}")
    print(f"Expected shape: (1, 64)\n")

    # Test with batch
    print("=" * 60)
    print("Testing with Batch")
    print("=" * 60)

    # Create batch of 4 graphs
    data_list = [data, data, data, data]
    batch = Batch.from_data_list(data_list)

    print(f"Batch: {batch.num_graphs} graphs")
    print(f"Total nodes: {batch.num_nodes}")

    with torch.no_grad():
        embeddings = encoder(batch)

    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Expected shape: (4, 64)\n")

    # Test GNNClassifier
    print("=" * 60)
    print("Testing GNNClassifier")
    print("=" * 60)

    classifier = GNNClassifier(
        input_dim=5,
        hidden_dim=128,
        embedding_dim=64,
        n_layers=3,
        n_classes=5,
        dropout=0.5
    )

    print(f"Model parameters: {classifier.get_num_params():,}")

    classifier.eval()
    with torch.no_grad():
        logits, embeddings = classifier(batch, return_embedding=True)

    print(f"Logits shape: {logits.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected: (4, 5) and (4, 64)\n")

    # Test GATEncoder
    print("=" * 60)
    print("Testing GATEncoder")
    print("=" * 60)

    gat_encoder = GATEncoder(
        input_dim=5,
        hidden_dim=128,
        embedding_dim=64,
        n_layers=3,
        heads=4,
        dropout=0.5
    )

    print(f"Model parameters: {gat_encoder.get_num_params():,}")

    gat_encoder.eval()
    with torch.no_grad():
        embedding = gat_encoder(batch)

    print(f"Output embedding shape: {embedding.shape}")
    print(f"Expected shape: (4, 64)\n")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
