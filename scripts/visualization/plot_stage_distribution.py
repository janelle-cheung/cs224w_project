"""
Plot distribution of sleep stages for each patient.
Shows the proportion of time spent in each sleep stage as a stacked bar chart.

Usage:
    python scripts/plot_stage_distribution.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append('src')

from config import DATA_DIR, STAGE_NAMES, FIGURES_DIR

LABELS_DIR = DATA_DIR / 'processed' / 'sleep_stage_labels'


def main():
    # Load all patient labels
    label_files = sorted(LABELS_DIR.glob("P*_labels.npy"))
    print(f"Found {len(label_files)} patients")

    patient_ids = []
    stage_distributions = []  # List of dicts: {stage: count}

    for label_file in label_files:
        patient_id = label_file.name.split('_')[0]
        labels = np.load(label_file)  # (num_epochs, 2) - [epoch_idx, label]

        # Extract label column and filter out -1
        stage_labels = labels[:, 1]
        valid_mask = stage_labels >= 0
        stage_labels_valid = stage_labels[valid_mask]

        # Count each stage
        stage_counts = {i: np.sum(stage_labels_valid == i) for i in range(5)}
        total_epochs = len(stage_labels_valid)
        stage_proportions = {i: stage_counts[i] / total_epochs * 100 for i in range(5)}

        patient_ids.append(patient_id)
        stage_distributions.append(stage_proportions)

        print(f"  {patient_id}: {total_epochs} epochs | "
              f"W={stage_proportions[0]:.1f}% N1={stage_proportions[1]:.1f}% "
              f"N2={stage_proportions[2]:.1f}% N3={stage_proportions[3]:.1f}% "
              f"R={stage_proportions[4]:.1f}%")

    # Define colors for each stage (using a distinct palette)
    colors = ['#FDB462', '#80B1D3', "#76D497", '#BEBADA', '#FB8072']

    # === FIGURE 1: Stacked horizontal bar chart ===
    fig1, ax1 = plt.subplots(figsize=(12, 10))

    n_patients = len(patient_ids)
    y_pos = np.arange(n_patients)

    # Build stacked bars
    left = np.zeros(n_patients)
    for stage_idx in range(5):
        widths = [dist[stage_idx] for dist in stage_distributions]
        ax1.barh(y_pos, widths, left=left, color=colors[stage_idx],
                 label=f'{STAGE_NAMES[stage_idx]}', edgecolor='white', linewidth=0.5)
        left += widths

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(patient_ids, fontsize=10)
    ax1.set_xlabel('Percentage (%)', fontsize=13)
    ax1.set_ylabel('Patient', fontsize=13)
    ax1.set_title('Sleep Stage Distribution per Patient (Stacked)', fontsize=15, pad=15)
    ax1.set_xlim(0, 100)
    ax1.legend(loc='upper right', fontsize=11, title='Sleep Stage')
    ax1.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save figure 1
    output_file1 = FIGURES_DIR / 'sleep_stage_distribution_stacked.png'
    plt.savefig(output_file1, dpi=200, bbox_inches='tight')
    print(f"\nStacked plot saved to {output_file1}")
    plt.close()

    # === FIGURE 2: Grouped bar chart ===
    fig2, ax2 = plt.subplots(figsize=(15, 8))

    x = np.arange(n_patients) * 2.1  # Add spacing between patient groups
    width = 0.35 # Wider bars

    for stage_idx in range(5):
        proportions = [dist[stage_idx] for dist in stage_distributions]
        offset = (stage_idx - 2) * width  # Center the bars
        ax2.bar(x + offset, proportions, width, color=colors[stage_idx],
                label=STAGE_NAMES[stage_idx], edgecolor='black', linewidth=0.8, alpha=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(patient_ids, rotation=90, fontsize=10)
    ax2.set_xlabel('Patient', fontsize=13)
    ax2.set_ylabel('Percentage (%)', fontsize=13)
    ax2.set_title('Sleep Stage Distribution per Patient (Grouped)', fontsize=15, pad=15)
    ax2.legend(loc='upper right', fontsize=11, title='Sleep Stage')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, max([max(d.values()) for d in stage_distributions]) * 1.1)

    plt.tight_layout()

    # Save figure 2
    output_file2 = FIGURES_DIR / 'sleep_stage_distribution_grouped.png'
    plt.savefig(output_file2, dpi=200, bbox_inches='tight')
    print(f"Grouped plot saved to {output_file2}")
    plt.close()

    # === Create summary statistics figure ===
    fig3, ax3 = plt.subplots(figsize=(10, 7))

    # Compute mean and std for each stage
    stage_means = []
    stage_stds = []
    for stage_idx in range(5):
        proportions = [dist[stage_idx] for dist in stage_distributions]
        stage_means.append(np.mean(proportions))
        stage_stds.append(np.std(proportions))

    # Average distribution
    ax3.bar(range(5), stage_means, color=colors, edgecolor='black', linewidth=1.5,
            yerr=stage_stds, capsize=5, width=0.6)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(STAGE_NAMES, fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=13)
    ax3.set_title('Average Sleep Stage Distribution\n(Mean ± SD across patients)',
                  fontsize=15, pad=15)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, max(stage_means) * 1.3)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(stage_means, stage_stds)):
        ax3.text(i, mean + std + 1, f'{mean:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save summary figure
    output_file3 = FIGURES_DIR / 'eda' / 'sleep_stage_summary.png'
    plt.savefig(output_file3, dpi=200, bbox_inches='tight')
    print(f"Summary plot saved to {output_file3}")

    plt.close('all')

    # Print aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    print(f"{'Stage':<10} {'Mean (%)':<12} {'Std (%)':<12} {'Min (%)':<12} {'Max (%)':<12}")
    print("-"*60)
    for stage_idx in range(5):
        proportions = [dist[stage_idx] for dist in stage_distributions]
        print(f"{STAGE_NAMES[stage_idx]:<10} {np.mean(proportions):<12.2f} "
              f"{np.std(proportions):<12.2f} {np.min(proportions):<12.2f} "
              f"{np.max(proportions):<12.2f}")


if __name__ == '__main__':
    main()
