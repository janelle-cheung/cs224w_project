"""
Train GNN Encoder with Contrastive Learning.

Uses NT-Xent loss to learn embeddings where epochs from the same patient
are pulled together and epochs from different patients are pushed apart.

Validation uses separation ratio (inter/intra patient distance) per stage:
- Higher ratio = better patient separation within each sleep stage
- Early stopping based on average separation ratio across stages

Supports multiple dataset types:
- single_epoch: Standard EEGGraphDataset (83 nodes, 5 features)
- temporal: TemporalEEGGraphDataset (83*n_steps nodes, 5 features)

Supports multiple model types:
- gcn: GNNEncoder (standard GCN)
- gat: GATEncoder (Graph Attention)
- temporal_gcn: TemporalGNNEncoder (GCN + GRU for temporal)

Usage:
    # From command line:
    python scripts/training/train_contrastive.py --k 10 --batch_size 32 --epochs 100
    python scripts/training/train_contrastive.py --dataset_type temporal --model_type temporal_gcn --n_temporal_steps 3

    # From another script:
    from train_contrastive import train, TrainConfig
    config = TrainConfig(lr=1e-4, temperature=0.1)
    results = train(config)
"""

import argparse
import copy
import csv
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import sys

sys.path.append('src')

from eeg_dataset import EEGGraphDataset
from temporal_eeg_dataset import TemporalEEGGraphDataset
from eeg_model import GNNEncoder, GATEncoder, GINEncoder, TemporalGNNEncoder, TemporalGATEncoder, TemporalGINEncoder
from contrastive_loss import NTXentLoss
from config import DATA_DIR, MODELS_DIR, LOGS_DIR, set_seed, DEVICE, N_CHANNELS, load_patient_splits, STAGE_NAMES_MAP


@dataclass
class TrainConfig:
    """Configuration for contrastive learning training."""
    # Dataset and model type
    dataset_type: str = 'single_epoch'
    model_type: str = 'gcn'
    n_temporal_steps: int = 3

    # Graph construction
    k: int = 10

    # Model architecture
    n_layers: int = 3
    hidden_dim: int = 128
    embedding_dim: int = 64

    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    dropout: float = 0.5
    temperature: float = 0.07
    patience: int = 5
    seed: int = 42

    # Output control
    verbose: bool = True  # Set False for cleaner output during hyperparam search
    save_model: bool = True  # Set False to skip saving model (for hyperparam search)


def get_dataset_class(dataset_type: str):
    """Return the appropriate dataset class based on type."""
    if dataset_type == 'single_epoch':
        return EEGGraphDataset
    elif dataset_type == 'temporal':
        return TemporalEEGGraphDataset
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'single_epoch' or 'temporal'.")


def get_model(model_type: str, config: TrainConfig):
    """
    Create and return the appropriate model based on type.

    Args:
        model_type: 'gcn', 'gat', 'gin', 'temporal_gcn', 'temporal_gat', or 'temporal_gin'
        config: TrainConfig with model hyperparameters

    Returns:
        Model instance
    """
    if model_type == 'gcn':
        return GNNEncoder(
            input_dim=5,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
            pooling='mean'
        )
    elif model_type == 'gat':
        return GATEncoder(
            input_dim=5,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            heads=4,
            dropout=config.dropout,
            pooling='mean'
        )
    elif model_type == 'gin':
        return GINEncoder(
            input_dim=5,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            dropout=config.dropout,
            pooling='mean'
        )
    elif model_type == 'temporal_gcn':
        return TemporalGNNEncoder(
            input_dim=5,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            n_channels=N_CHANNELS,
            n_temporal_steps=config.n_temporal_steps,
            dropout=config.dropout,
            temporal_model='gru',
            channel_pooling='mean'
        )
    elif model_type == 'temporal_gat':
        return TemporalGATEncoder(
            input_dim=5,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            n_channels=N_CHANNELS,
            n_temporal_steps=config.n_temporal_steps,
            heads=4,
            dropout=config.dropout,
            temporal_model='gru',
            channel_pooling='mean'
        )
    elif model_type == 'temporal_gin':
        return TemporalGINEncoder(
            input_dim=5,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            n_layers=config.n_layers,
            n_channels=N_CHANNELS,
            n_temporal_steps=config.n_temporal_steps,
            dropout=config.dropout,
            temporal_model='gru',
            channel_pooling='mean'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'gcn', 'gat', 'gin', 'temporal_gcn', 'temporal_gat', or 'temporal_gin'.")


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        embeddings = model(batch)
        sleep_stages = batch.y

        loss = criterion(embeddings, sleep_stages)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Extract embeddings for entire dataset."""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_patient_ids = []

    for batch in loader:
        batch = batch.to(device)
        embeddings = model(batch)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_patient_ids.extend(batch.patient_id)

    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)

    return embeddings, labels, all_patient_ids


def compute_separation_ratio(embeddings, patient_labels):
    """
    Compute separation ratio (inter-patient / intra-patient distance).

    A ratio > 1 means patients are more separated from each other than
    their own epochs are from each other (good for fingerprinting).
    """
    dist_matrix = squareform(pdist(embeddings))

    # Vectorized: create mask for same-patient pairs (upper triangle only)
    patient_labels = np.asarray(patient_labels)
    same_patient = patient_labels[:, None] == patient_labels[None, :]
    upper_tri = np.triu(np.ones_like(same_patient, dtype=bool), k=1)

    intra_mask = same_patient & upper_tri
    inter_mask = ~same_patient & upper_tri

    intra_dists = dist_matrix[intra_mask]
    inter_dists = dist_matrix[inter_mask]

    if len(intra_dists) == 0 or len(inter_dists) == 0:
        return 0.0

    return np.mean(inter_dists) / np.mean(intra_dists)


@torch.no_grad()
def compute_validation_metrics(model, loader, criterion, device):
    """
    Compute validation metrics including separation ratios per stage.

    Returns:
        dict with keys: val_loss, stage_sil, sep_ratio_W, sep_ratio_N1,
                        sep_ratio_N2, sep_ratio_N3, sep_ratio_R, avg_sep_ratio
    """
    model.eval()

    # Extract embeddings AND compute loss in single pass
    all_embeddings = []
    all_labels = []
    all_patient_ids = []
    total_loss = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        emb = model(batch)
        loss = criterion(emb, batch.y)

        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        all_patient_ids.extend(batch.patient_id)
        total_loss += loss.item()
        n_batches += 1

    embeddings = np.vstack(all_embeddings)
    stage_labels = np.concatenate(all_labels)
    patient_ids = all_patient_ids
    val_loss = total_loss / n_batches

    # Convert patient IDs to numeric
    unique_patients = sorted(set(patient_ids))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_labels = np.array([patient_to_idx[p] for p in patient_ids])

    # Stage silhouette (all epochs, stages as labels)
    if len(np.unique(stage_labels)) > 1:
        stage_sil = silhouette_score(embeddings, stage_labels)
    else:
        stage_sil = 0.0

    # Separation ratio per stage
    sep_ratios = {}
    for stage_idx, stage_name in STAGE_NAMES_MAP.items():
        mask = stage_labels == stage_idx
        if mask.sum() < 2:
            sep_ratios[stage_name] = 0.0
            continue

        stage_emb = embeddings[mask]
        stage_patients = patient_labels[mask]

        # Need at least 2 unique patients
        if len(np.unique(stage_patients)) < 2:
            sep_ratios[stage_name] = 0.0
        else:
            sep_ratios[stage_name] = compute_separation_ratio(stage_emb, stage_patients)

    # Average separation ratio across stages
    valid_ratios = [v for v in sep_ratios.values() if v != 0.0]
    avg_sep_ratio = np.mean(valid_ratios) if valid_ratios else 0.0

    return {
        'val_loss': val_loss,
        'stage_sil': stage_sil,
        'sep_ratio_W': sep_ratios['W'],
        'sep_ratio_N1': sep_ratios['N1'],
        'sep_ratio_N2': sep_ratios['N2'],
        'sep_ratio_N3': sep_ratios['N3'],
        'sep_ratio_R': sep_ratios['R'],
        'avg_sep_ratio': avg_sep_ratio
    }


def train(config: TrainConfig) -> Dict[str, Any]:
    """
    Train a GNN model with contrastive learning.

    Args:
        config: TrainConfig with all hyperparameters

    Returns:
        dict with keys: best_avg_sep_ratio, test_metrics, run_name
    """
    set_seed(config.seed)
    verbose = config.verbose

    if verbose:
        print("=" * 70)
        print("Contrastive Learning Training Configuration")
        print("=" * 70)
        print(f"Dataset type: {config.dataset_type}")
        print(f"Model type: {config.model_type}")
        if config.dataset_type == 'temporal' or config.model_type == 'temporal_gcn':
            print(f"Temporal steps: {config.n_temporal_steps}")
        print(f"Top-k edges per node: {config.k}")
        print(f"GNN layers: {config.n_layers}")
        print(f"Hidden dim: {config.hidden_dim}")
        print(f"Embedding dim: {config.embedding_dim}")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.epochs}")
        print(f"Learning rate: {config.lr}")
        print(f"Dropout: {config.dropout}")
        print(f"Temperature: {config.temperature}")
        print(f"Patience: {config.patience}")
        print(f"Device: {DEVICE}")
        print("=" * 70 + "\n")

    # Load canonical patient splits
    splits = load_patient_splits()
    train_ids = splits['train']
    val_ids = splits['val']
    test_ids = splits['test']

    # Generate run name for organized saving
    # Convert lr to float in case it was parsed as string from YAML
    lr_val = float(config.lr)
    if config.dataset_type == 'temporal':
        run_name = f"{config.model_type}_{config.dataset_type}_t{config.n_temporal_steps}_k{config.k}_e{config.embedding_dim}_temp{config.temperature:.2f}_lr{lr_val:.0e}_attn"
    else:
        run_name = f"{config.model_type}_{config.dataset_type}_k{config.k}_e{config.embedding_dim}_temp{config.temperature:.2f}_lr{lr_val:.0e}_attn"

    # Create output directories
    model_dir = MODELS_DIR / 'contrastive'
    model_dir.mkdir(exist_ok=True, parents=True)

    log_dir = LOGS_DIR / 'training'
    log_dir.mkdir(exist_ok=True, parents=True)

    emb_dir = DATA_DIR / 'graph_embeddings' / run_name
    emb_dir.mkdir(exist_ok=True, parents=True)

    model_path = model_dir / f"{run_name}.pt"
    csv_path = log_dir / f"{run_name}.csv"

    if verbose:
        n_total = len(train_ids) + len(val_ids) + len(test_ids)
        print(f"Total patients: {n_total}")
        print(f"Split seed: {splits['metadata']['seed']}\n")
        print(f"Train patients ({len(train_ids)}): {train_ids}")
        print(f"Val patients ({len(val_ids)}): {val_ids}")
        print(f"Test patients ({len(test_ids)}): {test_ids}\n")

    # Create datasets based on dataset_type
    if verbose:
        print("Loading datasets...", flush=True)
    DatasetClass = get_dataset_class(config.dataset_type)

    if config.dataset_type == 'temporal':
        train_dataset = DatasetClass(train_ids, n_temporal_steps=config.n_temporal_steps, k=config.k, verbose=verbose)
        val_dataset = DatasetClass(val_ids, n_temporal_steps=config.n_temporal_steps, k=config.k, verbose=verbose)
        test_dataset = DatasetClass(test_ids, n_temporal_steps=config.n_temporal_steps, k=config.k, verbose=verbose)
    else:
        train_dataset = DatasetClass(train_ids, k=config.k, verbose=verbose)
        val_dataset = DatasetClass(val_ids, k=config.k, verbose=verbose)
        test_dataset = DatasetClass(test_ids, k=config.k, verbose=verbose)

    if verbose:
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)} graphs\n")

    # Create dataloaders with parallel workers for faster loading
    num_workers = 4 if DEVICE.type == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'))

    # Create model based on model_type
    model = get_model(config.model_type, config).to(DEVICE)

    # Enable TF32 for faster matmul on A100/Ampere+ GPUs
    if DEVICE.type == 'cuda':
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.set_float32_matmul_precision('high')

    if verbose:
        print(f"Model ({config.model_type}): {model.get_num_params():,} parameters\n")

    # Loss and optimizer
    criterion = NTXentLoss(temperature=config.temperature)
    optimizer = optim.Adam(model.parameters(), lr=lr_val, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # CSV logging setup
    csv_columns = ['epoch', 'train_loss', 'val_loss', 'stage_sil',
                   'sep_ratio_W', 'sep_ratio_N1', 'sep_ratio_N2',
                   'sep_ratio_N3', 'sep_ratio_R', 'avg_sep_ratio', 'lr']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # Training loop
    best_avg_sep_ratio = -float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None

    if verbose:
        print("Training...")
        header = f"{'Ep':<4} {'TrLoss':<8} {'VLoss':<8} {'AvgSep':<8} {'Best':<4}"
        print(header)
        print("-" * len(header))
        epoch_iter = range(1, config.epochs + 1)
    else:
        epoch_iter = tqdm(range(1, config.epochs + 1), desc="Training", leave=False)

    for epoch in epoch_iter:
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate with separation ratio metrics
        metrics = compute_validation_metrics(model, val_loader, criterion, DEVICE)

        # Update learning rate based on avg_sep_ratio
        scheduler.step(metrics['avg_sep_ratio'])
        current_lr = optimizer.param_groups[0]['lr']

        # Check if best
        is_best = metrics['avg_sep_ratio'] > best_avg_sep_ratio
        best_marker = "*" if is_best else ""

        # Print progress
        if verbose:
            print(f"{epoch:<4} {train_loss:<8.4f} {metrics['val_loss']:<8.4f} "
                  f"{metrics['avg_sep_ratio']:<8.3f} {best_marker:<4}")

        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': metrics['val_loss'],
                'stage_sil': metrics['stage_sil'],
                'sep_ratio_W': metrics['sep_ratio_W'],
                'sep_ratio_N1': metrics['sep_ratio_N1'],
                'sep_ratio_N2': metrics['sep_ratio_N2'],
                'sep_ratio_N3': metrics['sep_ratio_N3'],
                'sep_ratio_R': metrics['sep_ratio_R'],
                'avg_sep_ratio': metrics['avg_sep_ratio'],
                'lr': current_lr
            })

        # Save best model (based on avg_sep_ratio)
        if is_best:
            best_avg_sep_ratio = metrics['avg_sep_ratio']
            best_epoch = epoch
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Deep copy to prevent corruption

            if config.save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'metrics': metrics,
                    'run_name': run_name,
                    'config': config.__dict__
                }, model_path)
        else:
            patience_counter += 1

        # Early stopping based on avg_sep_ratio
        if patience_counter >= config.patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        # Stop if learning rate too small
        if current_lr < 1e-6:
            if verbose:
                print(f"Stopping: lr too small ({current_lr})")
            break

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    test_metrics = compute_validation_metrics(model, test_loader, criterion, DEVICE)

    # Log test metrics as final row in CSV (epoch='test')
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writerow({
            'epoch': 'test',
            'train_loss': '',
            'val_loss': test_metrics['val_loss'],  # This is actually test loss
            'stage_sil': test_metrics['stage_sil'],
            'sep_ratio_W': test_metrics['sep_ratio_W'],
            'sep_ratio_N1': test_metrics['sep_ratio_N1'],
            'sep_ratio_N2': test_metrics['sep_ratio_N2'],
            'sep_ratio_N3': test_metrics['sep_ratio_N3'],
            'sep_ratio_R': test_metrics['sep_ratio_R'],
            'avg_sep_ratio': test_metrics['avg_sep_ratio'],
            'lr': ''
        })

    if verbose:
        print(f"\nBest: epoch {best_epoch}, val_sep={best_avg_sep_ratio:.4f}, test_sep={test_metrics['avg_sep_ratio']:.4f}")

    # Extract and save embeddings (only if saving model)
    if config.save_model:
        train_emb, train_labels, train_patients = extract_embeddings(model, train_loader, DEVICE)
        val_emb, val_labels, val_patients = extract_embeddings(model, val_loader, DEVICE)
        test_emb, test_labels, test_patients = extract_embeddings(model, test_loader, DEVICE)

        np.save(emb_dir / 'train_embeddings.npy', train_emb)
        np.save(emb_dir / 'train_labels.npy', train_labels)
        np.save(emb_dir / 'train_patient_ids.npy', np.array(train_patients))

        np.save(emb_dir / 'val_embeddings.npy', val_emb)
        np.save(emb_dir / 'val_labels.npy', val_labels)
        np.save(emb_dir / 'val_patient_ids.npy', np.array(val_patients))

        np.save(emb_dir / 'test_embeddings.npy', test_emb)
        np.save(emb_dir / 'test_labels.npy', test_labels)
        np.save(emb_dir / 'test_patient_ids.npy', np.array(test_patients))

        if verbose:
            print(f"Saved: {model_path.name}")

    return {
        'best_avg_sep_ratio': best_avg_sep_ratio,
        'test_metrics': test_metrics,
        'run_name': run_name,
        'model_state': best_model_state,  # For hyperparam search to save best model later
        'config': config,
    }


def parse_args() -> TrainConfig:
    """Parse command line arguments and return a TrainConfig."""
    parser = argparse.ArgumentParser(description='Train GNN with contrastive learning')
    # Dataset and model type arguments
    parser.add_argument('--dataset_type', type=str, default='single_epoch',
                        choices=['single_epoch', 'temporal'],
                        help='Dataset type: single_epoch or temporal')
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['gcn', 'gat', 'gin', 'temporal_gcn', 'temporal_gat', 'temporal_gin'],
                        help='Model type: gcn, gat, gin, temporal_gcn, temporal_gat, or temporal_gin')
    parser.add_argument('--n_temporal_steps', type=int, default=3,
                        help='Number of temporal steps (for temporal dataset/model)')
    # Graph construction arguments
    parser.add_argument('--k', type=int, default=10,
                        help='Number of top correlations per node')
    # Model architecture arguments
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for NT-Xent loss')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (based on avg_sep_ratio)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    return TrainConfig(
        dataset_type=args.dataset_type,
        model_type=args.model_type,
        n_temporal_steps=args.n_temporal_steps,
        k=args.k,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        temperature=args.temperature,
        patience=args.patience,
        seed=args.seed
    )


def main():
    """Main entry point for command line usage."""
    config = parse_args()
    return train(config)


if __name__ == '__main__':
    main()
