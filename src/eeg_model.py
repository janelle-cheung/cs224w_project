"""
GNN Models for EEG Graph Encoding.

Implements graph neural network encoders that map EEG graphs to embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch


class AttentionPooling(nn.Module):
    """
    Attention-based graph pooling that learns which nodes are important.

    Instead of treating all nodes equally (mean/max pool), this learns
    attention weights per node and computes a weighted sum.

    Args:
        hidden_dim: Dimension of node embeddings
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings (num_nodes, hidden_dim)
            batch: Batch assignment (num_nodes,)

        Returns:
            Graph embeddings (batch_size, hidden_dim)
        """
        # Compute attention scores
        attn_scores = self.attention(x)  # (num_nodes, 1)

        # Softmax within each graph
        attn_weights = self._softmax_per_graph(attn_scores, batch)  # (num_nodes, 1)

        # Weighted sum of node embeddings
        weighted_x = x * attn_weights  # (num_nodes, hidden_dim)

        # Sum within each graph
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(weighted_x), weighted_x)

        return out

    def _softmax_per_graph(self, scores: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Compute softmax of scores within each graph."""
        batch_size = batch.max().item() + 1

        # Subtract max for numerical stability
        max_scores = torch.full((batch_size, 1), float('-inf'), device=scores.device)
        max_scores.scatter_reduce_(0, batch.unsqueeze(1), scores, reduce='amax', include_self=False)
        scores_stable = scores - max_scores[batch]

        # Exp
        exp_scores = torch.exp(scores_stable)

        # Sum per graph
        sum_exp = torch.zeros(batch_size, 1, device=scores.device)
        sum_exp.scatter_add_(0, batch.unsqueeze(1), exp_scores)

        # Normalize
        return exp_scores / (sum_exp[batch] + 1e-8)


class ChannelAttentionPooling(nn.Module):
    """
    Attention pooling over the channel dimension for temporal models.

    Takes (batch, n_channels, hidden_dim) and outputs (batch, hidden_dim)
    by learning which channels are most important.

    Args:
        hidden_dim: Dimension of channel embeddings
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Channel embeddings (batch_size, n_channels, hidden_dim)

        Returns:
            Graph embeddings (batch_size, hidden_dim)
        """
        # Compute attention scores for each channel
        attn_scores = self.attention(x)  # (batch_size, n_channels, 1)

        # Softmax over channels
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, n_channels, 1)

        # Weighted sum over channels
        out = (x * attn_weights).sum(dim=1)  # (batch_size, hidden_dim)

        return out


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
        elif pooling == 'attention':
            self.pool = AttentionPooling(hidden_dim)
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
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        # If single graph (no batch attribute), create batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GCN layers with batch norm, ReLU, and dropout
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
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
        elif pooling == 'attention':
            self.pool = AttentionPooling(hidden_dim)
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


class GINEncoder(nn.Module):
    """
    GNN Encoder using Graph Isomorphism Networks (GIN).

    GIN is provably as powerful as the Weisfeiler-Lehman graph isomorphism test,
    making it highly expressive for graph-level tasks.

    Args:
        input_dim: Input node feature dimension (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        embedding_dim: Output embedding dimension (default: 64)
        n_layers: Number of GIN layers (default: 3)
        dropout: Dropout rate (default: 0.5)
        pooling: Global pooling method (default: 'mean')
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

        # GIN layers with MLP
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        # Batch normalization after GIN
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
        elif pooling == 'attention':
            self.pool = AttentionPooling(hidden_dim)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        # Embedding projection
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through GIN encoder."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # GIN layers
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Global pooling
        x = self.pool(x, batch)

        # Project to embedding
        embedding = self.embedding_layer(x)

        return embedding

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalGNNEncoder(nn.Module):
    """
    Temporal GNN Encoder for spatio-temporal EEG graphs.

    Designed for graphs with nodes = n_channels * n_temporal_steps (e.g., 83 * 3 = 249).
    Node indexing: node_idx = t * n_channels + channel_idx

    Architecture:
    1. GCN layers process spatial relationships across all nodes
    2. Reshape node embeddings to (batch, n_temporal_steps, n_channels, hidden_dim)
    3. GRU processes temporal sequence per channel: (n_temporal_steps, hidden_dim) -> hidden_dim
    4. Pool across channels to get graph embedding (mean or attention)
    5. Linear projection to embedding_dim

    Args:
        input_dim: Input node feature dimension (default: 5 for spectral bands)
        hidden_dim: Hidden layer dimension (default: 128)
        embedding_dim: Output embedding dimension (default: 64)
        n_layers: Number of GCN layers (default: 3)
        n_channels: Number of EEG channels (default: 83)
        n_temporal_steps: Number of temporal steps (default: 3)
        dropout: Dropout rate (default: 0.5)
        temporal_model: 'gru' or 'mean' for temporal aggregation (default: 'gru')
        channel_pooling: 'mean' or 'attention' for channel aggregation (default: 'mean')
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 3,
        n_channels: int = 83,
        n_temporal_steps: int = 3,
        dropout: float = 0.5,
        temporal_model: str = 'gru',
        channel_pooling: str = 'mean'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_temporal_steps = n_temporal_steps
        self.dropout = dropout
        self.temporal_model = temporal_model
        self.channel_pooling = channel_pooling

        # GCN layers for spatial processing
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Temporal aggregation
        if temporal_model == 'gru':
            # GRU for temporal sequence: (batch * n_channels, n_temporal_steps, hidden_dim)
            self.temporal_gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0
            )
        elif temporal_model == 'mean':
            self.temporal_gru = None
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}. Use 'gru' or 'mean'.")

        # Channel pooling
        if channel_pooling == 'attention':
            self.channel_pool = ChannelAttentionPooling(hidden_dim)
        elif channel_pooling == 'mean':
            self.channel_pool = None
        else:
            raise ValueError(f"Unknown channel_pooling: {channel_pooling}. Use 'mean' or 'attention'.")

        # Embedding projection
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through Temporal GNN encoder.

        Args:
            data: PyG Data or Batch object with:
                - x: Node features (num_nodes, input_dim) where num_nodes = batch_size * n_channels * n_temporal_steps
                - edge_index: Edge indices (2, num_edges)
                - batch: Batch assignment (num_nodes,)

        Returns:
            embeddings: (batch_size, embedding_dim) graph embeddings
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None

        # If single graph (no batch attribute), create batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Get batch size from batch tensor
        batch_size = batch.max().item() + 1

        # === 1. GCN layers for spatial processing ===
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # x shape: (total_nodes, hidden_dim) where total_nodes = batch_size * nodes_per_graph

        # === 2. Reshape for temporal processing ===
        # Reshape to (batch_size, n_temporal_steps, n_channels, hidden_dim)
        # Node ordering in each graph: [t0_ch0, t0_ch1, ..., t0_ch82, t1_ch0, ..., t2_ch82]
        x = x.view(batch_size, self.n_temporal_steps, self.n_channels, self.hidden_dim)

        # === 3. Temporal aggregation per channel ===
        if self.temporal_model == 'gru':
            # Reshape for GRU: (batch_size * n_channels, n_temporal_steps, hidden_dim)
            x = x.permute(0, 2, 1, 3)  # (batch, n_channels, n_temporal_steps, hidden_dim)
            x = x.reshape(batch_size * self.n_channels, self.n_temporal_steps, self.hidden_dim)

            # GRU forward pass
            _, h_n = self.temporal_gru(x)  # h_n: (1, batch*n_channels, hidden_dim)
            x = h_n.squeeze(0)  # (batch*n_channels, hidden_dim)

            # Reshape back: (batch_size, n_channels, hidden_dim)
            x = x.view(batch_size, self.n_channels, self.hidden_dim)
        else:
            # Simple mean pooling over temporal dimension
            x = x.mean(dim=1)  # (batch_size, n_channels, hidden_dim)

        # === 4. Pool across channels ===
        if self.channel_pool is not None:
            x = self.channel_pool(x)  # (batch_size, hidden_dim)
        else:
            x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # === 5. Project to embedding dimension ===
        embedding = self.embedding_layer(x)  # (batch_size, embedding_dim)

        return embedding

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalGATEncoder(nn.Module):
    """
    Temporal GAT Encoder for spatio-temporal EEG graphs.

    Same architecture as TemporalGNNEncoder but uses GAT layers instead of GCN.
    This allows attention-weighted aggregation of neighbor information.

    Args:
        input_dim: Input node feature dimension (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        embedding_dim: Output embedding dimension (default: 64)
        n_layers: Number of GAT layers (default: 3)
        n_channels: Number of EEG channels (default: 83)
        n_temporal_steps: Number of temporal steps (default: 3)
        heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.5)
        temporal_model: 'gru' or 'mean' for temporal aggregation (default: 'gru')
        channel_pooling: 'mean' or 'attention' for channel aggregation (default: 'mean')
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 3,
        n_channels: int = 83,
        n_temporal_steps: int = 3,
        heads: int = 4,
        dropout: float = 0.5,
        temporal_model: str = 'gru',
        channel_pooling: str = 'mean'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_temporal_steps = n_temporal_steps
        self.heads = heads
        self.dropout = dropout
        self.temporal_model = temporal_model
        self.channel_pooling = channel_pooling

        # GAT layers for spatial processing
        self.convs = nn.ModuleList()
        # First layer: input_dim -> hidden_dim * heads
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout))

        # Middle layers: hidden_dim * heads -> hidden_dim * heads
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))

        # Last layer: hidden_dim * heads -> hidden_dim (concat=False)
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        # Batch normalization layers
        self.bns = nn.ModuleList()
        for i in range(n_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Temporal aggregation
        if temporal_model == 'gru':
            self.temporal_gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0
            )
        elif temporal_model == 'mean':
            self.temporal_gru = None
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}. Use 'gru' or 'mean'.")

        # Channel pooling
        if channel_pooling == 'attention':
            self.channel_pool = ChannelAttentionPooling(hidden_dim)
        elif channel_pooling == 'mean':
            self.channel_pool = None
        else:
            raise ValueError(f"Unknown channel_pooling: {channel_pooling}. Use 'mean' or 'attention'.")

        # Embedding projection
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through Temporal GAT encoder."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1

        # === 1. GAT layers for spatial processing ===
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)  # GAT typically uses ELU
            x = F.dropout(x, p=self.dropout, training=self.training)

        # === 2. Reshape for temporal processing ===
        x = x.view(batch_size, self.n_temporal_steps, self.n_channels, self.hidden_dim)

        # === 3. Temporal aggregation per channel ===
        if self.temporal_model == 'gru':
            x = x.permute(0, 2, 1, 3)  # (batch, n_channels, n_temporal_steps, hidden_dim)
            x = x.reshape(batch_size * self.n_channels, self.n_temporal_steps, self.hidden_dim)

            _, h_n = self.temporal_gru(x)
            x = h_n.squeeze(0)

            x = x.view(batch_size, self.n_channels, self.hidden_dim)
        else:
            x = x.mean(dim=1)

        # === 4. Pool across channels ===
        if self.channel_pool is not None:
            x = self.channel_pool(x)  # (batch_size, hidden_dim)
        else:
            x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # === 5. Project to embedding dimension ===
        embedding = self.embedding_layer(x)

        return embedding

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TemporalGINEncoder(nn.Module):
    """
    Temporal GIN Encoder for spatio-temporal EEG graphs.

    Same architecture as TemporalGNNEncoder but uses GIN layers.
    GIN is maximally expressive for graph-level tasks.

    Args:
        input_dim: Input node feature dimension (default: 5)
        hidden_dim: Hidden layer dimension (default: 128)
        embedding_dim: Output embedding dimension (default: 64)
        n_layers: Number of GIN layers (default: 3)
        n_channels: Number of EEG channels (default: 83)
        n_temporal_steps: Number of temporal steps (default: 3)
        dropout: Dropout rate (default: 0.5)
        temporal_model: 'gru' or 'mean' for temporal aggregation (default: 'gru')
        channel_pooling: 'mean' or 'attention' for channel aggregation (default: 'mean')
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 3,
        n_channels: int = 83,
        n_temporal_steps: int = 3,
        dropout: float = 0.5,
        temporal_model: str = 'gru',
        channel_pooling: str = 'mean'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_temporal_steps = n_temporal_steps
        self.dropout = dropout
        self.temporal_model = temporal_model
        self.channel_pooling = channel_pooling

        # GIN layers with MLP
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        # Batch normalization after GIN
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Temporal aggregation
        if temporal_model == 'gru':
            self.temporal_gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0
            )
        elif temporal_model == 'mean':
            self.temporal_gru = None
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}. Use 'gru' or 'mean'.")

        # Channel pooling
        if channel_pooling == 'attention':
            self.channel_pool = ChannelAttentionPooling(hidden_dim)
        elif channel_pooling == 'mean':
            self.channel_pool = None
        else:
            raise ValueError(f"Unknown channel_pooling: {channel_pooling}. Use 'mean' or 'attention'.")

        # Embedding projection
        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through Temporal GIN encoder."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        batch_size = batch.max().item() + 1

        # === 1. GIN layers for spatial processing ===
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # === 2. Reshape for temporal processing ===
        x = x.view(batch_size, self.n_temporal_steps, self.n_channels, self.hidden_dim)

        # === 3. Temporal aggregation per channel ===
        if self.temporal_model == 'gru':
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(batch_size * self.n_channels, self.n_temporal_steps, self.hidden_dim)

            _, h_n = self.temporal_gru(x)
            x = h_n.squeeze(0)

            x = x.view(batch_size, self.n_channels, self.hidden_dim)
        else:
            x = x.mean(dim=1)

        # === 4. Pool across channels ===
        if self.channel_pool is not None:
            x = self.channel_pool(x)  # (batch_size, hidden_dim)
        else:
            x = x.mean(dim=1)  # (batch_size, hidden_dim)

        # === 5. Project to embedding dimension ===
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

    # Test TemporalGNNEncoder
    print("=" * 60)
    print("Testing TemporalGNNEncoder")
    print("=" * 60)

    # Create temporal graph data (249 nodes = 83 channels * 3 time steps)
    n_channels = 83
    n_temporal_steps = 3
    num_nodes_temporal = n_channels * n_temporal_steps  # 249
    num_edges_temporal = 2822  # typical for k=10

    x_temporal = torch.randn(num_nodes_temporal, input_dim)
    edge_index_temporal = torch.randint(0, num_nodes_temporal, (2, num_edges_temporal))
    data_temporal = Data(x=x_temporal, edge_index=edge_index_temporal)

    print(f"Temporal graph: {num_nodes_temporal} nodes ({n_channels} channels × {n_temporal_steps} steps)")
    print(f"Node features: {x_temporal.shape}")

    # Test with GRU temporal model
    temporal_encoder_gru = TemporalGNNEncoder(
        input_dim=5,
        hidden_dim=128,
        embedding_dim=64,
        n_layers=3,
        n_channels=n_channels,
        n_temporal_steps=n_temporal_steps,
        dropout=0.5,
        temporal_model='gru'
    )

    print(f"\nGRU model parameters: {temporal_encoder_gru.get_num_params():,}")

    temporal_encoder_gru.eval()
    with torch.no_grad():
        embedding_gru = temporal_encoder_gru(data_temporal)

    print(f"GRU output embedding shape: {embedding_gru.shape}")
    print(f"Expected shape: (1, 64)")

    # Test with mean temporal model
    temporal_encoder_mean = TemporalGNNEncoder(
        input_dim=5,
        hidden_dim=128,
        embedding_dim=64,
        n_layers=3,
        n_channels=n_channels,
        n_temporal_steps=n_temporal_steps,
        dropout=0.5,
        temporal_model='mean'
    )

    print(f"\nMean model parameters: {temporal_encoder_mean.get_num_params():,}")

    temporal_encoder_mean.eval()
    with torch.no_grad():
        embedding_mean = temporal_encoder_mean(data_temporal)

    print(f"Mean output embedding shape: {embedding_mean.shape}")
    print(f"Expected shape: (1, 64)")

    # Test TemporalGNNEncoder with batch
    print("\n" + "-" * 40)
    print("Testing TemporalGNNEncoder with Batch")
    print("-" * 40)

    data_list_temporal = [data_temporal, data_temporal, data_temporal, data_temporal]
    batch_temporal = Batch.from_data_list(data_list_temporal)

    print(f"Batch: {batch_temporal.num_graphs} graphs")
    print(f"Total nodes: {batch_temporal.num_nodes}")

    with torch.no_grad():
        embeddings_temporal = temporal_encoder_gru(batch_temporal)

    print(f"Batch output embeddings shape: {embeddings_temporal.shape}")
    print(f"Expected shape: (4, 64)\n")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
