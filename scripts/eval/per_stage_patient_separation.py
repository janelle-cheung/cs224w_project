"""
Per-Stage Patient Separation Analysis
Check if embeddings separate patients WITHIN each sleep stage.
This is the real test for individual fingerprints.

Usage:
    python scripts/diagnostic_per_patient_stage_clustering.py                    # val only (default)
    python scripts/diagnostic_per_patient_stage_clustering.py --split train      # train only
    python scripts/diagnostic_per_patient_stage_clustering.py --split val        # val only
    python scripts/diagnostic_per_patient_stage_clustering.py --split train_val  # train + val combined
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

sys.path.append('src')
from config import DATA_DIR, LOGS_DIR, FIGURES_DIR


def load_embeddings(embeddings_dir: Path, split: str):
    """
    Load embeddings, labels, and patient IDs for specified split(s).

    Args:
        embeddings_dir: Directory containing embedding files
        split: One of 'train', 'val', 'test', or 'all'

    Returns:
        emb, labels, patient_ids as numpy arrays
    """
    if split == 'train_val':
        splits_to_load = ['train', 'val']
    else:
        splits_to_load = [split]

    emb_list = []
    labels_list = []
    patient_ids_list = []

    for s in splits_to_load:
        emb_file = embeddings_dir / f'{s}_embeddings.npy'
        labels_file = embeddings_dir / f'{s}_labels.npy'
        patient_file = embeddings_dir / f'{s}_patient_ids.npy'

        if emb_file.exists():
            emb_list.append(np.load(emb_file))
            labels_list.append(np.load(labels_file))
            patient_ids_list.append(np.load(patient_file, allow_pickle=True))
        else:
            print(f"Warning: {emb_file} not found, skipping")

    if not emb_list:
        raise FileNotFoundError(f"No embedding files found for split(s): {splits_to_load}")

    emb = np.concatenate(emb_list)
    labels = np.concatenate(labels_list)
    patient_ids = np.concatenate(patient_ids_list)

    return emb, labels, patient_ids


def analyze_patient_separation_per_stage(embeddings_dir, split='val'):
    """
    For each sleep stage, compute:
    - Intra-patient distances (same patient, same stage, different epochs)
    - Inter-patient distances (different patients, same stage)
    - Silhouette score for patient clustering within that stage
    """
    embeddings_dir = Path(embeddings_dir)

    # Load embeddings and labels for specified split(s)
    emb, labels, patient_ids = load_embeddings(embeddings_dir, split)
    
    # Convert patient IDs to numeric
    unique_patients = sorted(set(patient_ids))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_numeric = np.array([patient_to_idx[p] for p in patient_ids])
    
    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}

    print("=" * 70)
    print("PER-STAGE PATIENT SEPARATION ANALYSIS")
    print("=" * 70)
    print(f"Split: {split}")
    print(f"Total embeddings: {len(emb)}")
    print(f"Unique patients: {len(unique_patients)}\n")
    
    results = {}
    
    for stage_idx in range(5):
        stage_name = stage_names[stage_idx]
        mask = labels == stage_idx
        
        if mask.sum() < 2:
            print(f"{stage_name}: Skipped (< 2 epochs)")
            continue
        
        stage_emb = emb[mask]
        stage_patients = patient_numeric[mask]
        
        # Need at least 2 unique patients
        if len(np.unique(stage_patients)) < 2:
            print(f"{stage_name}: Skipped (< 2 unique patients)")
            continue
        
        # Compute all pairwise distances
        distances = pdist(stage_emb)
        dist_matrix = squareform(distances)
        
        # Separate intra vs inter
        intra_dists = []
        inter_dists = []
        
        for i in range(len(stage_emb)):
            for j in range(i + 1, len(stage_emb)):
                d = dist_matrix[i, j]
                if stage_patients[i] == stage_patients[j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)
        
        intra_dists = np.array(intra_dists)
        inter_dists = np.array(inter_dists)
        
        # Silhouette for patient clustering in this stage
        try:
            pat_sil = silhouette_score(stage_emb, stage_patients)
        except:
            pat_sil = 0.0
        
        # Separation ratio
        sep_ratio = np.mean(inter_dists) / np.mean(intra_dists) if len(intra_dists) > 0 else 0
        
        results[stage_name] = {
            'n_epochs': len(stage_emb),
            'n_patients': len(np.unique(stage_patients)),
            'intra_mean': np.mean(intra_dists),
            'intra_std': np.std(intra_dists),
            'inter_mean': np.mean(inter_dists),
            'inter_std': np.std(inter_dists),
            'separation_ratio': sep_ratio,
            'silhouette': pat_sil
        }
        
        print(f"{stage_name}:")
        print(f"  Epochs: {len(stage_emb)}, Patients: {len(np.unique(stage_patients))}")
        print(f"  Intra-patient dist:  {np.mean(intra_dists):.4f} ± {np.std(intra_dists):.4f}")
        print(f"  Inter-patient dist:  {np.mean(inter_dists):.4f} ± {np.std(inter_dists):.4f}")
        print(f"  Separation ratio (inter/intra): {sep_ratio:.3f}")
        print(f"  Patient silhouette: {pat_sil:.4f}")
        
        if sep_ratio > 1.2:
            print(f"  ✅ GOOD: Patients separated")
        else:
            print(f"  ❌ BAD: Patients overlapping")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    good_stages = [s for s, r in results.items() if r['separation_ratio'] > 1.2]
    bad_stages = [s for s, r in results.items() if r['separation_ratio'] <= 1.2]
    
    if good_stages:
        print(f"✅ Stages with good patient separation: {', '.join(good_stages)}")
    if bad_stages:
        print(f"❌ Stages with poor patient separation: {', '.join(bad_stages)}")
    
    avg_ratio = np.mean([r['separation_ratio'] for r in results.values()])
    print(f"\nAverage separation ratio: {avg_ratio:.3f}")
    
    if avg_ratio > 1.2:
        print("→ Model has patient fingerprints")
    else:
        print("→ Model does NOT have patient fingerprints")
    
    return results


def plot_patient_separation(results: dict, output_name: str, split: str):
    """
    Plot per-stage patient separation metrics.
    Creates a 2-panel figure with:
    - Left: Intra vs Inter patient distances per stage (bar chart with error bars)
    - Right: Separation ratio and silhouette score per stage
    """
    if not results:
        print("No results to plot")
        return

    stages = list(results.keys())
    n_stages = len(stages)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(n_stages)
    width = 0.35

    # Left panel: Intra vs Inter distances
    ax = axes[0]
    intra_means = [results[s]['intra_mean'] for s in stages]
    intra_stds = [results[s]['intra_std'] for s in stages]
    inter_means = [results[s]['inter_mean'] for s in stages]
    inter_stds = [results[s]['inter_std'] for s in stages]

    bars1 = ax.bar(x - width/2, intra_means, width, yerr=intra_stds,
                   label='Intra-patient', color='tab:blue', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, inter_means, width, yerr=inter_stds,
                   label='Inter-patient', color='tab:orange', alpha=0.8, capsize=3)

    ax.set_xlabel('Sleep Stage')
    ax.set_ylabel('Distance')
    ax.set_title('Intra vs Inter Patient Distances')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Right panel: Separation ratio and silhouette (dual y-axis)
    ax1 = axes[1]
    sep_ratios = [results[s]['separation_ratio'] for s in stages]
    silhouettes = [results[s]['silhouette'] for s in stages]

    color1 = 'tab:green'
    ax1.set_xlabel('Sleep Stage')
    ax1.set_ylabel('Separation Ratio (inter/intra)', color=color1)
    bars3 = ax1.bar(x - width/2, sep_ratios, width, label='Sep. Ratio',
                    color=color1, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=1.2, color=color1, linestyle='--', alpha=0.5, label='Threshold (1.2)')

    ax2 = ax1.twinx()
    color2 = 'tab:purple'
    ax2.set_ylabel('Silhouette Score', color=color2)
    bars4 = ax2.bar(x + width/2, silhouettes, width, label='Silhouette',
                    color=color2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_title('Patient Separation Metrics')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.suptitle(f'Per-Stage Patient Separation: {output_name} ({split})', fontsize=12)
    plt.tight_layout()

    # Save figure
    fig_dir = FIGURES_DIR / 'diagnostics'
    fig_dir.mkdir(exist_ok=True, parents=True)
    plot_file = fig_dir / f'patient_separation_{output_name}_{split}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to {plot_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Per-stage patient separation analysis')
    parser.add_argument('--embeddings_dir', type=str, default=None,
                        help='Directory containing embedding files (default: temporal GCN)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'train_val'],
                        help='Which split(s) to analyze (default: val)')
    args = parser.parse_args()

    # Default to temporal GCN embeddings
    if args.embeddings_dir is None:
        embeddings_dir = DATA_DIR / 'graph_embeddings' / 'temporal_gcn_temporal_t3_k10_e64_temp0.50_lr1e-03'
    else:
        embeddings_dir = Path(args.embeddings_dir)

    if not embeddings_dir.exists():
        print(f"Embeddings dir not found: {embeddings_dir}")
        sys.exit(1)

    results = analyze_patient_separation_per_stage(embeddings_dir, split=args.split)

    # Generate plot
    output_name = embeddings_dir.name
    plot_patient_separation(results, output_name, args.split)