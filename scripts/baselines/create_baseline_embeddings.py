"""
Create baseline embeddings (power/connectivity) aligned with patient_splits.

IMPORTANT: PCA and StandardScaler are fit ONLY on train data to prevent data leakage.
The same fitted transforms are then applied to val and test data.

Usage:
    python scripts/baselines/create_baseline_embeddings.py --feature_type power --n_components 64
    python scripts/baselines/create_baseline_embeddings.py --feature_type connectivity --n_components 64
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.append('src')
from config import DATA_DIR, LOGS_DIR, load_patient_splits, N_CHANNELS

SPECTRAL_DIR = DATA_DIR / 'processed' / 'spectral_features'
LABELS_DIR = DATA_DIR / 'processed' / 'sleep_stage_labels'


def compute_connectivity_features(spectral_features: np.ndarray) -> np.ndarray:
    """
    Compute connectivity features from spectral features.

    Args:
        spectral_features: (num_epochs, 83, 5) band power per channel

    Returns:
        connectivity: (num_epochs, 3403) upper triangle of correlation matrix
    """
    num_epochs = spectral_features.shape[0]

    # Upper triangle indices (excluding diagonal)
    triu_idx = np.triu_indices(N_CHANNELS, k=1)
    n_features = len(triu_idx[0])  # 83*82/2 = 3403

    connectivity = np.zeros((num_epochs, n_features), dtype=np.float32)

    for epoch_idx in range(num_epochs):
        # Shape: (83, 5) - channels x bands
        epoch_data = spectral_features[epoch_idx]

        # Compute correlation matrix: (83, 83)
        corr_matrix = np.corrcoef(epoch_data)

        # Handle NaN (can occur if a channel has zero variance)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Extract upper triangle
        connectivity[epoch_idx] = corr_matrix[triu_idx]

    return connectivity


def load_patient_features(patient_ids: list, feature_type: str) -> tuple:
    """
    Load and concatenate features for a list of patients.
    Only includes epochs with valid labels (stage >= 0).

    Args:
        patient_ids: List of patient IDs
        feature_type: 'power' or 'connectivity'

    Returns:
        features: np.ndarray of shape (n_epochs, n_features)
        n_epochs_per_patient: dict mapping patient_id to number of valid epochs
    """
    all_features = []
    n_epochs_per_patient = {}

    for patient_id in patient_ids:
        # Load spectral features
        spectral_file = SPECTRAL_DIR / f'{patient_id}_spectral_features.npy'
        if not spectral_file.exists():
            print(f"  Warning: {spectral_file} not found, skipping {patient_id}")
            continue

        spectral = np.load(spectral_file)

        # Load labels to find valid epochs
        label_file = LABELS_DIR / f'{patient_id}_labels.npy'
        if not label_file.exists():
            print(f"  Warning: {label_file} not found, skipping {patient_id}")
            continue

        labels = np.load(label_file)

        # Only keep epochs with valid labels (stage >= 0)
        valid_mask = labels[:, 1] >= 0

        # Align features with labels
        n_epochs = min(len(spectral), len(valid_mask))
        spectral = spectral[:n_epochs]
        valid_mask = valid_mask[:n_epochs]

        valid_spectral = spectral[valid_mask]

        # Compute features based on type
        if feature_type == 'power':
            # Flatten spectral features: (n_epochs, 83, 5) -> (n_epochs, 415)
            features = valid_spectral.reshape(len(valid_spectral), -1)
        elif feature_type == 'connectivity':
            # Compute correlation matrix upper triangle
            features = compute_connectivity_features(valid_spectral)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        all_features.append(features)
        n_epochs_per_patient[patient_id] = len(features)

    if not all_features:
        return np.array([]), {}

    return np.vstack(all_features), n_epochs_per_patient


def main():
    parser = argparse.ArgumentParser(description='Create baseline embeddings with proper train/val/test split')
    parser.add_argument('--feature_type', type=str, required=True,
                        choices=['power', 'connectivity'],
                        help='Type of baseline features')
    parser.add_argument('--n_components', type=int, required=True,
                        help='Number of PCA components')
    args = parser.parse_args()

    feature_type = args.feature_type
    n_components = args.n_components

    # Output directory
    output_dir = DATA_DIR / 'baseline_embeddings' / f'{feature_type}{n_components}'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / 'baselines' / f'{feature_type}{n_components}_no_leakage.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Load patient splits
    splits = load_patient_splits()

    print(f"Creating baseline embeddings: {feature_type}{n_components}")
    print(f"Output: {output_dir}")
    print(f"\nIMPORTANT: PCA fitted on TRAIN data only (no data leakage)\n")

    log_lines = [
        f"Baseline Embeddings - {feature_type}{n_components}",
        f"PCA fitted on TRAIN data only (no data leakage)",
        "",
    ]

    # Step 1: Load TRAIN features and fit PCA
    print("Loading TRAIN features...")
    train_features, train_counts = load_patient_features(splits['train'], feature_type)
    print(f"  Train: {train_features.shape[0]} epochs from {len(train_counts)} patients")
    log_lines.append(f"Train: {train_features.shape[0]} epochs from {len(train_counts)} patients")

    # Fit StandardScaler on train data
    print("\nFitting StandardScaler on train data...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    # Fit PCA on train data
    print(f"Fitting PCA ({n_components} components) on train data...")
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_scaled)

    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained: {variance_explained:.1f}%")
    log_lines.append(f"PCA variance explained (train): {variance_explained:.2f}%")
    log_lines.append("")

    # Save train embeddings
    np.save(output_dir / 'train_embeddings.npy', train_pca.astype(np.float32))
    print(f"  Saved train_embeddings.npy: {train_pca.shape}")

    # Step 2: Transform VAL and TEST using fitted scaler and PCA
    for split_name in ['val', 'test']:
        print(f"\nTransforming {split_name.upper()} features...")
        features, counts = load_patient_features(splits[split_name], feature_type)
        print(f"  {split_name}: {features.shape[0]} epochs from {len(counts)} patients")
        log_lines.append(f"{split_name.capitalize()}: {features.shape[0]} epochs from {len(counts)} patients")

        # Transform using fitted scaler and PCA (NOT fit_transform!)
        scaled = scaler.transform(features)
        pca_features = pca.transform(scaled)

        # Save
        np.save(output_dir / f'{split_name}_embeddings.npy', pca_features.astype(np.float32))
        print(f"  Saved {split_name}_embeddings.npy: {pca_features.shape}")

    # Verify alignment with patient_splits labels
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    patient_splits_dir = DATA_DIR / 'patient_splits'
    all_aligned = True
    for split_name in ['train', 'val', 'test']:
        emb_file = output_dir / f'{split_name}_embeddings.npy'
        labels_file = patient_splits_dir / f'{split_name}_labels.npy'

        emb = np.load(emb_file)
        labels = np.load(labels_file)

        if emb.shape[0] == labels.shape[0]:
            print(f"  {split_name}: ALIGNED ({emb.shape[0]} samples)")
            log_lines.append(f"{split_name}: ALIGNED ({emb.shape[0]} samples)")
        else:
            print(f"  {split_name}: MISMATCH! embeddings={emb.shape[0]}, labels={labels.shape[0]}")
            log_lines.append(f"{split_name}: MISMATCH! embeddings={emb.shape[0]}, labels={labels.shape[0]}")
            all_aligned = False

    # Write log
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"\nLog saved to {log_file}")

    if all_aligned:
        print(f"\nDone! Baseline embeddings saved to {output_dir}")
    else:
        print(f"\nWARNING: Some splits are not aligned with labels!")


if __name__ == '__main__':
    main()
