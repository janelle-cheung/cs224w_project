"""
Train GNN Encoder with Contrastive Learning.

Uses NT-Xent loss to learn embeddings where epochs from the same patient
are pulled together and epochs from different patients are pushed apart.

Validation uses silhouette scores to measure clustering quality:
- Stage silhouette: how well embeddings cluster by sleep stage
- Patient silhouette per stage: how well patients separate within each stage

Usage:
    python scripts/training/train_contrastive.py --k 10 --batch_size 32 --epochs 100
"""

import argparse
import csv
import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from pathlib import Path
from sklearn.metrics import silhouette_score
import sys

sys.path.append('src')

from eeg_dataset import EEGGraphDataset, split_by_patient, SPECTRAL_DIR
from eeg_model import GNNEncoder
from contrastive_loss import NTXentLoss
from config import DATA_DIR, MODELS_DIR, LOGS_DIR, set_seed, DEVICE

STAGE_NAMES = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}


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


@torch.no_grad()
def compute_validation_metrics(model, loader, criterion, device):
    """
    Compute validation metrics including silhouette scores.

    Returns:
        dict with keys: val_loss, stage_sil, patient_sil_W, patient_sil_N1,
                        patient_sil_N2, patient_sil_N3, patient_sil_R, avg_patient_sil
    """
    model.eval()

    # Extract all embeddings
    embeddings, stage_labels, patient_ids = extract_embeddings(model, loader, device)

    # Convert patient IDs to numeric
    unique_patients = sorted(set(patient_ids))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_labels = np.array([patient_to_idx[p] for p in patient_ids])

    # Compute validation loss
    total_loss = 0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        emb = model(batch)
        loss = criterion(emb, batch.y)
        total_loss += loss.item()
        n_batches += 1
    val_loss = total_loss / n_batches

    # Stage silhouette (all epochs, stages as labels)
    if len(np.unique(stage_labels)) > 1:
        stage_sil = silhouette_score(embeddings, stage_labels)
    else:
        stage_sil = 0.0

    # Patient silhouette per stage
    patient_sils = {}
    for stage_idx, stage_name in STAGE_NAMES.items():
        mask = stage_labels == stage_idx
        if mask.sum() < 2:
            patient_sils[stage_name] = 0.0
            continue

        stage_emb = embeddings[mask]
        stage_patients = patient_labels[mask]

        # Need at least 2 unique patients
        if len(np.unique(stage_patients)) < 2:
            patient_sils[stage_name] = 0.0
        else:
            patient_sils[stage_name] = silhouette_score(stage_emb, stage_patients)

    # Average patient silhouette across stages
    valid_sils = [v for v in patient_sils.values() if v != 0.0]
    avg_patient_sil = np.mean(valid_sils) if valid_sils else 0.0

    return {
        'val_loss': val_loss,
        'stage_sil': stage_sil,
        'patient_sil_W': patient_sils['W'],
        'patient_sil_N1': patient_sils['N1'],
        'patient_sil_N2': patient_sils['N2'],
        'patient_sil_N3': patient_sils['N3'],
        'patient_sil_R': patient_sils['R'],
        'avg_patient_sil': avg_patient_sil
    }


def main():
    parser = argparse.ArgumentParser(description='Train GNN with contrastive learning')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of top correlations per node')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
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
                        help='Early stopping patience (based on avg_patient_sil)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 70)
    print("Contrastive Learning Training Configuration")
    print("=" * 70)
    print(f"Top-k edges per node: {args.k}")
    print(f"GNN layers: {args.n_layers}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"Temperature: {args.temperature}")
    print(f"Patience: {args.patience}")
    print(f"Device: {DEVICE}")
    print("=" * 70 + "\n")

    # Get all patient IDs
    spectral_files = sorted(SPECTRAL_DIR.glob("P*_spectral_features.npy"))
    all_patient_ids = [f.name.split('_')[0] for f in spectral_files]

    # Generate run name for organized saving
    run_name = f"gcn_k{args.k}_e{args.embedding_dim}_t{args.temperature:.2f}_lr{args.lr:.0e}"

    # Create output directories
    model_dir = MODELS_DIR / 'gcn_contrastive'
    model_dir.mkdir(exist_ok=True, parents=True)

    log_dir = LOGS_DIR / 'training'
    log_dir.mkdir(exist_ok=True, parents=True)

    emb_dir = DATA_DIR / 'graph_embeddings' / run_name
    emb_dir.mkdir(exist_ok=True, parents=True)

    model_path = model_dir / f"{run_name}.pt"
    csv_path = log_dir / f"{run_name}.csv"

    print(f"Total patients: {len(all_patient_ids)}\n")

    # Split: 70/14/16
    train_ids, val_ids, test_ids = split_by_patient(
        all_patient_ids,
        train_ratio=0.70,
        val_ratio=0.14,
        test_ratio=0.16,
        seed=args.seed
    )

    print(f"Train patients ({len(train_ids)}): {train_ids}")
    print(f"Val patients ({len(val_ids)}): {val_ids}")
    print(f"Test patients ({len(test_ids)}): {test_ids}\n")

    # Create datasets
    print("Loading datasets...")
    train_dataset = EEGGraphDataset(train_ids, k=args.k)
    val_dataset = EEGGraphDataset(val_ids, k=args.k)
    test_dataset = EEGGraphDataset(test_ids, k=args.k)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val: {len(val_dataset)} graphs")
    print(f"  Test: {len(test_dataset)} graphs\n")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = GNNEncoder(
        input_dim=5,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pooling='mean'
    ).to(DEVICE)

    print(f"Model: {model.get_num_params():,} parameters\n")

    # Loss and optimizer
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # CSV logging setup
    csv_columns = ['epoch', 'train_loss', 'val_loss', 'stage_sil',
                   'patient_sil_W', 'patient_sil_N1', 'patient_sil_N2',
                   'patient_sil_N3', 'patient_sil_R', 'avg_patient_sil', 'lr']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # Training loop
    best_avg_patient_sil = -float('inf')
    patience_counter = 0

    print("Starting training...")
    header = f"{'Ep':<4} {'TrLoss':<8} {'VLoss':<8} {'StgSil':<8} " \
             f"{'W':<7} {'N1':<7} {'N2':<7} {'N3':<7} {'R':<7} {'AvgPat':<8} {'Best':<4}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate with silhouette metrics
        metrics = compute_validation_metrics(model, val_loader, criterion, DEVICE)

        # Update learning rate based on avg_patient_sil
        scheduler.step(metrics['avg_patient_sil'])
        current_lr = optimizer.param_groups[0]['lr']

        # Check if best
        is_best = metrics['avg_patient_sil'] > best_avg_patient_sil
        best_marker = "*" if is_best else ""

        # Print progress
        print(f"{epoch:<4} {train_loss:<8.4f} {metrics['val_loss']:<8.4f} "
              f"{metrics['stage_sil']:<8.4f} {metrics['patient_sil_W']:<7.4f} "
              f"{metrics['patient_sil_N1']:<7.4f} {metrics['patient_sil_N2']:<7.4f} "
              f"{metrics['patient_sil_N3']:<7.4f} {metrics['patient_sil_R']:<7.4f} "
              f"{metrics['avg_patient_sil']:<8.4f} {best_marker:<4}")

        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': metrics['val_loss'],
                'stage_sil': metrics['stage_sil'],
                'patient_sil_W': metrics['patient_sil_W'],
                'patient_sil_N1': metrics['patient_sil_N1'],
                'patient_sil_N2': metrics['patient_sil_N2'],
                'patient_sil_N3': metrics['patient_sil_N3'],
                'patient_sil_R': metrics['patient_sil_R'],
                'avg_patient_sil': metrics['avg_patient_sil'],
                'lr': current_lr
            })

        # Save best model (based on avg_patient_sil)
        if is_best:
            best_avg_patient_sil = metrics['avg_patient_sil']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'metrics': metrics,
                'run_name': run_name,
                'args': vars(args)
            }, model_path)
        else:
            patience_counter += 1

        # Early stopping based on avg_patient_sil
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement in avg_patient_sil for {args.patience} epochs)")
            break

        # Stop if learning rate too small
        if current_lr < 1e-6:
            print(f"\nStopping: learning rate too small ({current_lr})")
            break

    print("\n" + "=" * 70)
    print(f"Training complete! Best avg_patient_sil: {best_avg_patient_sil:.6f}")
    print(f"Model saved to: {model_path}")
    print(f"CSV log saved to: {csv_path}")

    # Load best model for final evaluation
    print("\n" + "=" * 70)
    print("Evaluating best model on test set...")
    print("=" * 70)

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = compute_validation_metrics(model, test_loader, criterion, DEVICE)
    print(f"\nTest metrics:")
    print(f"  Loss: {test_metrics['val_loss']:.6f}")
    print(f"  Stage silhouette: {test_metrics['stage_sil']:.6f}")
    print(f"  Patient silhouettes: W={test_metrics['patient_sil_W']:.4f}, "
          f"N1={test_metrics['patient_sil_N1']:.4f}, N2={test_metrics['patient_sil_N2']:.4f}, "
          f"N3={test_metrics['patient_sil_N3']:.4f}, R={test_metrics['patient_sil_R']:.4f}")
    print(f"  Avg patient silhouette: {test_metrics['avg_patient_sil']:.6f}")

    # Extract and save embeddings
    print("\n" + "=" * 70)
    print("Extracting embeddings...")
    print("=" * 70)

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

    print(f"\nEmbeddings saved to: {emb_dir}")
    print(f"  Train: {train_emb.shape}")
    print(f"  Val: {val_emb.shape}")
    print(f"  Test: {test_emb.shape}")

    print("\n" + "=" * 70)
    print("All done!")
    print("=" * 70)

    # Return metrics for sweep script
    return {
        'best_avg_patient_sil': best_avg_patient_sil,
        'test_metrics': test_metrics,
        'run_name': run_name
    }


if __name__ == '__main__':
    main()
