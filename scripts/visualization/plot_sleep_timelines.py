"""
Plot stacked timeline of sleep stages for all patients.
Each patient is a horizontal row, showing sleep stage progression over epochs.

Usage:
    python scripts/plot_sleep_timelines.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from matplotlib.patches import Patch

sys.path.append('src')

from config import DATA_DIR, STAGE_NAMES, FIGURES_DIR

LABELS_DIR = DATA_DIR / 'processed' / 'sleep_stage_labels'

def main():
    # Load all patient labels
    label_files = sorted(LABELS_DIR.glob("P*_labels.npy"))
    print(f"Found {len(label_files)} patients")

    patient_data = []
    patient_ids = []
    max_epochs = 0

    for label_file in label_files:
        patient_id = label_file.name.split('_')[0]
        labels = np.load(label_file)  # (num_epochs, 2) - [epoch_idx, label]

        # Extract label column and filter out -1
        stage_labels = labels[:, 1]
        valid_mask = stage_labels >= 0
        stage_labels_valid = stage_labels[valid_mask]

        patient_data.append(stage_labels_valid)
        patient_ids.append(patient_id)
        max_epochs = max(max_epochs, len(stage_labels_valid))

        print(f"  {patient_id}: {len(stage_labels_valid)} valid epochs")

    # Create 2D array with padding (use -1 for invalid/padding)
    n_patients = len(patient_data)
    timeline_matrix = np.full((n_patients, max_epochs), -1, dtype=int)

    for i, stages in enumerate(patient_data):
        timeline_matrix[i, :len(stages)] = stages

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Define colormap: 0-4 for sleep stages, -1 for invalid
    # Use discrete colors for each stage
    cmap = plt.colormaps.get_cmap('viridis').resampled(5)

    # Mask invalid regions
    masked_data = np.ma.masked_where(timeline_matrix == -1, timeline_matrix)

    # Plot timeline
    im = ax.imshow(masked_data, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=0, vmax=4)

    # Set axis labels
    ax.set_xlabel('Epoch', fontsize=15, labelpad=10)
    ax.set_ylabel('Patient', fontsize=15, labelpad=20)
    ax.set_title('Sleep Stage Timeline Per Patient', fontsize=18, pad=20)

    # Set y-ticks to patient IDs
    ax.set_yticks(range(n_patients))
    ax.set_yticklabels(patient_ids, fontsize=8)

    # Add colorbar with centered ticks
    cbar = plt.colorbar(im, ax=ax, ticks=[0.4, 1.2, 2, 2.8, 3.6])
    cbar.set_label('Sleep Stage', rotation=270, labelpad=30, fontsize=15)
    cbar.ax.set_yticklabels(STAGE_NAMES, fontsize=15)

    plt.tight_layout()

    # Save figure
    output_file = FIGURES_DIR / 'eda'/ 'sleep_stage_timelines.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")

    plt.close()


if __name__ == '__main__':
    main()
