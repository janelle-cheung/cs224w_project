"""
Baseline 2: Connectivity (correlation matrix) + PCA
Input: spectral features (num_epochs, 83, 5) per patient
Process: correlation matrix per epoch -> upper triangle -> PCA
Output:
  - Raw upper triangle: data/correlation_matrices/P##_corr.npy (num_epochs, 3403)
  - PCA-reduced: data/connectivity_{n}/P##_connectivity_{n}.npy (num_epochs, n)

Usage:
    python scripts/baselines/create_connectivity_baseline.py --n_components 64
    python scripts/baselines/create_connectivity_baseline.py --n_components 32
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.append('src')

from config import DATA_DIR, LOGS_DIR, N_CHANNELS

SPECTRAL_DIR = DATA_DIR / 'processed' / 'spectral_features'


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


def main():
    parser = argparse.ArgumentParser(description='Connectivity baseline with PCA')
    parser.add_argument('--n_components', type=int, required=True,
                        help='Number of PCA components')
    args = parser.parse_args()

    n_components = args.n_components

    # Output directories
    corr_dir = DATA_DIR / 'processed' / 'correlation_matrices'
    pca_dir = DATA_DIR / 'processed' / f'connectivity_{n_components}'
    corr_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / 'baselines' / f'pca_variance_connectivity{n_components}.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    spectral_files = sorted(SPECTRAL_DIR.glob("P*_spectral_features.npy"))

    print(f"Processing {len(spectral_files)} patients...")
    print(f"Pipeline: (epochs, 83, 5) -> corr matrix (83, 83) -> upper tri (3403) -> PCA ({n_components}D)")
    print(f"Raw correlation output: {corr_dir}")
    print(f"PCA output: {pca_dir}\n")

    log_lines = [
        f"Connectivity Baseline - {n_components}D",
        f"Input: spectral features (num_epochs, 83 channels, 5 bands)",
        f"Process: correlation matrix -> upper triangle (3403D) -> PCA ({n_components}D)",
        "",
        "Variance Explained per Patient:",
        "-" * 40,
    ]

    for spectral_file in spectral_files:
        patient_id = spectral_file.name.split('_')[0]

        try:
            # Load spectral features: (num_epochs, 83, 5)
            features = np.load(spectral_file)
            n_epochs = features.shape[0]

            # Compute connectivity features: (num_epochs, 3403)
            connectivity = compute_connectivity_features(features)

            # Save raw correlation matrix (upper triangle)
            corr_file = corr_dir / f'{patient_id}_corr.npy'
            np.save(corr_file, connectivity)

            # Standardize before PCA
            scaler = StandardScaler()
            connectivity_scaled = scaler.fit_transform(connectivity)

            # Apply PCA
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(connectivity_scaled)

            # Save PCA features
            pca_file = pca_dir / f'{patient_id}_connectivity_{n_components}.npy'
            np.save(pca_file, features_pca.astype(np.float32))

            variance_explained = pca.explained_variance_ratio_.sum() * 100
            print(f"  {patient_id}: {features.shape} -> corr -> {connectivity.shape} -> {features_pca.shape} ({variance_explained:.1f}% variance)")
            log_lines.append(f"{patient_id}: {variance_explained:.2f}%")

        except Exception as e:
            print(f"  {patient_id}: ERROR - {e}")
            log_lines.append(f"{patient_id}: ERROR - {e}")

    # Write log file
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"\nDone!")
    print(f"  Raw correlation matrices saved to {corr_dir}")
    print(f"  PCA features saved to {pca_dir}")
    print(f"  Variance log saved to {log_file}")


if __name__ == '__main__':
    main()
