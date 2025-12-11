"""
Baseline 1: Raw spectral power concatenated with PCA reduction.
Input: spectral features (num_epochs, 83, 5) per patient
Output: PCA-reduced features (num_epochs, n_components) per patient

Usage:
    python scripts/create_pca_baseline.py --n_components 64
    python scripts/create_pca_baseline.py --n_components 32
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.append('src')

from config import DATA_DIR, LOGS_DIR

SPECTRAL_DIR = DATA_DIR / 'processed' / 'spectral_features'


def main():
    parser = argparse.ArgumentParser(description='PCA baseline for spectral features')
    parser.add_argument('--n_components', type=int, required=True,
                        help='Number of PCA components')
    args = parser.parse_args()

    n_components = args.n_components
    output_dir = DATA_DIR / 'processed' / f'power{n_components}'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / 'baselines' / f'pca_variance_power{n_components}.txt'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    spectral_files = sorted(SPECTRAL_DIR.glob("P*_spectral_features.npy"))

    print(f"Processing {len(spectral_files)} patients...")
    print(f"PCA: 415D -> {n_components}D")
    print(f"Output directory: {output_dir}\n")

    log_lines = [
        f"PCA Baseline - {n_components}D",
        f"Input: spectral features (num_epochs, 83 channels, 5 bands) -> flattened to 415D",
        f"Output: PCA-reduced features ({n_components}D)",
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

            # Flatten to (num_epochs, 415)
            features_flat = features.reshape(n_epochs, -1)

            # Standardize before PCA
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_flat)

            # Apply PCA
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)

            # Save
            output_file = output_dir / f'{patient_id}_power{n_components}.npy'
            np.save(output_file, features_pca.astype(np.float32))

            variance_explained = pca.explained_variance_ratio_.sum() * 100
            print(f"  {patient_id}: {features.shape} -> {features_pca.shape} ({variance_explained:.1f}% variance)")
            log_lines.append(f"{patient_id}: {variance_explained:.2f}%")

        except Exception as e:
            print(f"  {patient_id}: ERROR - {e}")
            log_lines.append(f"{patient_id}: ERROR - {e}")

    # Write log file
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"\nDone! PCA features saved to {output_dir}")
    print(f"Variance log saved to {log_file}")


if __name__ == '__main__':
    main()
