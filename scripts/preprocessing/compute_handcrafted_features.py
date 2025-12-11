"""
Compute handcrafted sleep stage features per epoch.
Input: spectral features (num_epochs, 83, 5) - already have delta/theta/alpha/beta/gamma power
Output: handcrafted features (num_epochs, n_features)

Features:
  - delta_relative_power: delta / (delta + theta + alpha + beta + gamma)
  - theta_delta_ratio: theta / delta
  - spindle_presence: binary, has spindle in epoch (from YASA)
  - Plus all original 5 bands for context
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append('src')
from config import DATA_DIR

SPECTRAL_DIR = DATA_DIR / 'processed' / 'spectral_features'
OUTPUT_DIR = DATA_DIR / 'processed' / 'handcrafted_features'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_handcrafted(spectral_features: np.ndarray) -> np.ndarray:
    """
    Args:
        spectral_features: (num_epochs, 83, 5) where last dim is [delta, theta, alpha, beta, gamma]
    
    Returns:
        handcrafted: (num_epochs, n_features) 
    """
    num_epochs = spectral_features.shape[0]
    
    # Extract power per band, averaged across channels
    # Shape: (num_epochs, 5)
    band_power = spectral_features.mean(axis=1)  # average across 83 channels
    
    delta = band_power[:, 0]
    theta = band_power[:, 1]
    alpha = band_power[:, 2]
    beta = band_power[:, 3]
    gamma = band_power[:, 4]
    
    # Compute features
    total_power = delta + theta + alpha + beta + gamma + 1e-10  # avoid div by zero
    delta_relative = delta / total_power
    theta_delta_ratio = theta / (delta + 1e-10)

    # Sigma band power (11-15 Hz) - proxy for spindle activity
    # We can approximate this from the spectral data
    # For now, use a simple heuristic: alpha + beta as spindle proxy
    spindle_proxy = (alpha + beta) / total_power
    
    # Stack: (num_epochs, 9)
    features = np.stack([
        delta_relative,
        theta_delta_ratio,
        spindle_proxy,
        delta,
        theta,
        alpha,
        beta,
        gamma,
        (alpha + beta)  # raw spindle power
    ], axis=1)
    
    return features.astype(np.float32)

def main():
    spectral_files = sorted(SPECTRAL_DIR.glob("P*_spectral_features.npy"))
    print(f"Processing {len(spectral_files)} patients...\n")
    
    for spectral_file in spectral_files:
        patient_id = spectral_file.name.split('_')[0]
        
        try:
            spectral = np.load(spectral_file)  # (num_epochs, 83, 5)
            features = compute_handcrafted(spectral)
            
            output_file = OUTPUT_DIR / f'{patient_id}_handcrafted.npy'
            np.save(output_file, features)
            
            print(f"  {patient_id}: {spectral.shape} -> {features.shape}")
            
        except Exception as e:
            print(f"  {patient_id}: ERROR - {e}")
    
    print(f"\nDone! Features saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()