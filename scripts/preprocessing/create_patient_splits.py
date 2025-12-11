#!/usr/bin/env python3
"""
Create canonical patient train/val/test splits.

This script generates the OFFICIAL patient splits that ALL methods should use.
Run this ONCE to create the split files, then all training/evaluation scripts
should load from these files to ensure consistent comparisons.

Output files in data/patient_splits/:
    - patient_splits.json      : Complete split info with metadata (human-readable JSON)
    - train_patients.txt       : One patient ID per line
    - val_patients.txt         : One patient ID per line
    - test_patients.txt        : One patient ID per line
    - {split}_labels.npy       : Sleep stage labels for each split (epoch_idx, label)
    - {split}_patient_ids.npy  : Patient ID for each epoch in split

Usage:
    python scripts/preprocessing/create_patient_splits.py

    # To regenerate with a different seed:
    python scripts/preprocessing/create_patient_splits.py --seed 123
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import DATA_DIR, SEED, STAGE_NAMES


def create_patient_splits(
    all_patient_ids: list,
    train_ratio: float = 0.70,
    val_ratio: float = 0.14,
    test_ratio: float = 0.16,
    seed: int = SEED
) -> dict:
    """
    Split patient IDs into train/val/test sets.

    Args:
        all_patient_ids: List of all patient IDs (e.g., ['P01', 'P02', ...])
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.14)
        test_ratio: Fraction for testing (default: 0.16)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with split information
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Sort patient IDs to ensure consistent starting point
    sorted_ids = sorted(all_patient_ids)
    n_patients = len(sorted_ids)

    # Shuffle with seed
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(sorted_ids).tolist()

    # Calculate split sizes
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    # Remaining go to test (handles rounding)

    # Split
    train_ids = sorted(shuffled_ids[:n_train])
    val_ids = sorted(shuffled_ids[n_train:n_train + n_val])
    test_ids = sorted(shuffled_ids[n_train + n_val:])

    return {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
        'metadata': {
            'seed': seed,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'n_patients': n_patients,
            'n_train': len(train_ids),
            'n_val': len(val_ids),
            'n_test': len(test_ids),
            'created_at': datetime.now().isoformat(),
        }
    }


def load_split_labels(patient_ids: list, labels_dir: Path) -> tuple:
    """
    Load and aggregate sleep stage labels for a list of patients.

    Args:
        patient_ids: List of patient IDs
        labels_dir: Directory containing {patient_id}_labels.npy files

    Returns:
        labels: np.ndarray of shape (n_epochs,) with sleep stage labels (0-4)
        patient_ids_expanded: np.ndarray of shape (n_epochs,) with patient ID for each epoch
        stage_counts: dict mapping stage name to count
    """
    all_labels = []
    all_patient_ids = []

    for patient_id in patient_ids:
        label_file = labels_dir / f'{patient_id}_labels.npy'
        if not label_file.exists():
            print(f"  Warning: {label_file} not found, skipping")
            continue

        # Labels are (n_epochs, 2) with columns [epoch_idx, stage_label]
        labels = np.load(label_file)

        # Filter out invalid labels (L = -1)
        valid_mask = labels[:, 1] >= 0
        valid_labels = labels[valid_mask, 1].astype(np.int32)

        all_labels.append(valid_labels)
        all_patient_ids.extend([patient_id] * len(valid_labels))

    if not all_labels:
        return np.array([]), np.array([]), {}

    labels = np.concatenate(all_labels)
    patient_ids_expanded = np.array(all_patient_ids)

    # Count stages
    stage_counts = {}
    for i, name in enumerate(STAGE_NAMES):
        stage_counts[name] = int(np.sum(labels == i))

    return labels, patient_ids_expanded, stage_counts


def main():
    parser = argparse.ArgumentParser(description='Create patient train/val/test splits')
    parser.add_argument('--seed', type=int, default=SEED,
                        help=f'Random seed (default: {SEED})')
    parser.add_argument('--train-ratio', type=float, default=0.70,
                        help='Training set ratio (default: 0.70)')
    parser.add_argument('--val-ratio', type=float, default=0.14,
                        help='Validation set ratio (default: 0.14)')
    parser.add_argument('--test-ratio', type=float, default=0.16,
                        help='Test set ratio (default: 0.16)')
    args = parser.parse_args()

    # Find all patients from spectral features directory
    spectral_dir = DATA_DIR / 'processed' / 'spectral_features'
    labels_dir = DATA_DIR / 'processed' / 'sleep_stage_labels'
    patient_files = sorted(spectral_dir.glob('P*_spectral_features.npy'))
    all_patient_ids = [f.stem.split('_')[0] for f in patient_files]

    if not all_patient_ids:
        raise FileNotFoundError(f"No patient files found in {spectral_dir}")

    print(f"Found {len(all_patient_ids)} patients: {all_patient_ids}")
    print(f"\nSplit parameters:")
    print(f"  Seed: {args.seed}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Val ratio: {args.val_ratio}")
    print(f"  Test ratio: {args.test_ratio}")

    # Create splits
    splits = create_patient_splits(
        all_patient_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    # Create output directory
    splits_dir = DATA_DIR / 'patient_splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Save labels and patient IDs for each split
    print("\nAggregating labels per split...")
    split_stats = {}
    for split_name in ['train', 'val', 'test']:
        patient_ids = splits[split_name]
        labels, patient_ids_expanded, stage_counts = load_split_labels(patient_ids, labels_dir)

        # Save labels and patient IDs
        np.save(splits_dir / f'{split_name}_labels.npy', labels)
        np.save(splits_dir / f'{split_name}_patient_ids.npy', patient_ids_expanded)

        split_stats[split_name] = {
            'n_epochs': len(labels),
            'stage_counts': stage_counts
        }
        print(f"  {split_name}: {len(labels)} epochs")
        for stage, count in stage_counts.items():
            pct = count / len(labels) * 100 if len(labels) > 0 else 0
            print(f"    {stage}: {count} ({pct:.1f}%)")

    # Add split stats to metadata
    splits['metadata']['split_stats'] = split_stats

    # Save complete JSON (human-readable with all metadata)
    json_path = splits_dir / 'patient_splits.json'
    with open(json_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save individual text files (one patient ID per line, easy to read)
    for split_name in ['train', 'val', 'test']:
        txt_path = splits_dir / f'{split_name}_patients.txt'
        with open(txt_path, 'w') as f:
            for patient_id in splits[split_name]:
                f.write(f"{patient_id}\n")
        print(f"Saved: {txt_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PATIENT SPLITS CREATED")
    print("=" * 60)
    print(f"\nTrain ({len(splits['train'])} patients, {split_stats['train']['n_epochs']} epochs):")
    print(f"  {', '.join(splits['train'])}")
    print(f"\nVal ({len(splits['val'])} patients, {split_stats['val']['n_epochs']} epochs):")
    print(f"  {', '.join(splits['val'])}")
    print(f"\nTest ({len(splits['test'])} patients, {split_stats['test']['n_epochs']} epochs):")
    print(f"  {', '.join(splits['test'])}")
    print("\n" + "=" * 60)
    print("ALL SCRIPTS SHOULD NOW USE THESE SPLITS!")
    print(f"Load with: load_patient_splits() from src/config.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
