"""
Per-Stage Patient Separation Analysis
Check if embeddings separate patients WITHIN each sleep stage.
This is the real test for individual fingerprints.

Usage:
    python scripts/eval/per_stage_patient_separation.py --folder path/to/embeddings --splits val
    python scripts/eval/per_stage_patient_separation.py --folder path/to/embeddings --splits train val test
    python scripts/eval/per_stage_patient_separation.py --folder path/to/embeddings --splits train val
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


def load_embeddings(embeddings_dir: Path, splits: list):
    """
    Load embeddings from embeddings_dir, and labels/patient_ids.

    - patient_ids: ALWAYS from patient_splits
    - labels: from embeddings_dir first, fallback to patient_splits (for baselines)

    Args:
        embeddings_dir: Directory containing embedding files ({split}_embeddings.npy)
        splits: List of splits to load, e.g. ['train'], ['val'], ['train', 'val', 'test']

    Returns:
        emb, labels, patient_ids as numpy arrays

    Raises:
        FileNotFoundError: If any requested split's files are missing
    """
    patient_splits_dir = DATA_DIR / 'patient_splits'

    emb_list = []
    labels_list = []
    patient_ids_list = []

    for s in splits:
        emb_file = embeddings_dir / f'{s}_embeddings.npy'

        if not emb_file.exists():
            raise FileNotFoundError(f"Missing required file: {emb_file}")

        emb = np.load(emb_file)
        emb_list.append(emb)

        # Labels: check embeddings_dir first, then patient_splits (for baselines)
        labels_file = embeddings_dir / f'{s}_labels.npy'
        if not labels_file.exists():
            labels_file = patient_splits_dir / f'{s}_labels.npy'
        if not labels_file.exists():
            raise FileNotFoundError(f"Missing labels file for {s}")

        # Patient IDs: check embeddings_dir first (newer runs), then patient_splits
        patient_file = embeddings_dir / f'{s}_patient_ids.npy'
        if not patient_file.exists():
            patient_file = patient_splits_dir / f'{s}_patient_ids.npy'
        if not patient_file.exists():
            raise FileNotFoundError(f"Missing patient_ids file for {s}")

        labels = np.load(labels_file)
        pids = np.load(patient_file, allow_pickle=True)

        # Verify sizes match
        if len(labels) != len(emb):
            raise ValueError(f"Labels size mismatch for {s}: emb={len(emb)}, labels={len(labels)}")
        if len(pids) != len(emb):
            raise ValueError(f"Patient IDs size mismatch for {s}: emb={len(emb)}, pids={len(pids)}")

        labels_list.append(labels)
        patient_ids_list.append(pids)

    emb = np.concatenate(emb_list)
    labels = np.concatenate(labels_list)
    patient_ids = np.concatenate(patient_ids_list)

    return emb, labels, patient_ids


def compute_separation_ratio(dist_matrix, patient_labels):
    """
    Compute separation ratio (inter/intra) given a distance matrix and patient labels.
    """
    intra_dists = []
    inter_dists = []

    n = len(patient_labels)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if patient_labels[i] == patient_labels[j]:
                intra_dists.append(d)
            else:
                inter_dists.append(d)

    if len(intra_dists) == 0:
        return 0.0

    return np.mean(inter_dists) / np.mean(intra_dists)


def permutation_test_per_stage(stage_emb, stage_patients, n_permutations=1000):
    """
    Run permutation test for a single stage.
    Shuffles patient labels within the stage and computes null distribution of separation ratios.

    Returns:
        null_ratios: array of separation ratios under null hypothesis
    """
    # Precompute distance matrix (this is the expensive part, only do once)
    dist_matrix = squareform(pdist(stage_emb))

    null_ratios = []
    for _ in range(n_permutations):
        # Shuffle patient labels
        shuffled_patients = np.random.permutation(stage_patients)
        ratio = compute_separation_ratio(dist_matrix, shuffled_patients)
        null_ratios.append(ratio)

    return np.array(null_ratios)


def analyze_patient_separation_per_stage(embeddings_dir, splits: list, n_permutations=1000):
    """
    For each sleep stage, compute:
    - Intra-patient distances (same patient, same stage, different epochs)
    - Inter-patient distances (different patients, same stage)
    - Silhouette score for patient clustering within that stage
    - Null hypothesis threshold via permutation test

    Args:
        embeddings_dir: Directory containing embedding files
        splits: List of splits to analyze, e.g. ['train'], ['val'], ['train', 'val', 'test']
        n_permutations: Number of permutations for null hypothesis test
    """
    embeddings_dir = Path(embeddings_dir)

    # Load embeddings and labels for specified split(s)
    emb, labels, patient_ids = load_embeddings(embeddings_dir, splits)

    split_str = '+'.join(splits)
    
    # Convert patient IDs to numeric
    unique_patients = sorted(set(patient_ids))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_numeric = np.array([patient_to_idx[p] for p in patient_ids])
    
    stage_names = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}

    print("=" * 70)
    print("PER-STAGE PATIENT SEPARATION ANALYSIS")
    print("=" * 70)
    print(f"Split(s): {split_str}")
    print(f"Total embeddings: {len(emb)}")
    print(f"Unique patients: {len(unique_patients)}")
    print(f"Permutation tests: {n_permutations} iterations\n")

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

        # Permutation test for null hypothesis
        print(f"{stage_name}: Running permutation test...")
        null_ratios = permutation_test_per_stage(stage_emb, stage_patients, n_permutations)
        null_mean = np.mean(null_ratios)
        null_std = np.std(null_ratios)
        null_95 = np.percentile(null_ratios, 95)

        # p-value: fraction of null ratios >= observed ratio
        p_value = np.mean(null_ratios >= sep_ratio)

        results[stage_name] = {
            'n_epochs': len(stage_emb),
            'n_patients': len(np.unique(stage_patients)),
            'intra_mean': np.mean(intra_dists),
            'intra_std': np.std(intra_dists),
            'inter_mean': np.mean(inter_dists),
            'inter_std': np.std(inter_dists),
            'separation_ratio': sep_ratio,
            'silhouette': pat_sil,
            'null_mean': null_mean,
            'null_std': null_std,
            'null_95': null_95,
            'p_value': p_value
        }

        print(f"  Epochs: {len(stage_emb)}, Patients: {len(np.unique(stage_patients))}")
        print(f"  Intra-patient dist:  {np.mean(intra_dists):.4f} ± {np.std(intra_dists):.4f}")
        print(f"  Inter-patient dist:  {np.mean(inter_dists):.4f} ± {np.std(inter_dists):.4f}")
        print(f"  Separation ratio (inter/intra): {sep_ratio:.3f}")
        print(f"  Null hypothesis: {null_mean:.3f} ± {null_std:.3f} (95th: {null_95:.3f})")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Patient silhouette: {pat_sil:.4f}")

        if sep_ratio > null_95:
            print(f"  ✅ SIGNIFICANT: Patients separated (p < 0.05)")
        else:
            print(f"  ❌ NOT SIGNIFICANT: Patients overlapping")
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Use null hypothesis threshold (p < 0.05) instead of hardcoded 1.2
    sig_stages = [s for s, r in results.items() if r['separation_ratio'] > r['null_95']]
    nonsig_stages = [s for s, r in results.items() if r['separation_ratio'] <= r['null_95']]

    if sig_stages:
        print(f"✅ Stages with significant patient separation (p < 0.05): {', '.join(sig_stages)}")
    if nonsig_stages:
        print(f"❌ Stages without significant separation: {', '.join(nonsig_stages)}")

    avg_ratio = np.mean([r['separation_ratio'] for r in results.values()])
    avg_null = np.mean([r['null_mean'] for r in results.values()])
    print(f"\nAverage separation ratio: {avg_ratio:.3f} (null avg: {avg_null:.3f})")

    if len(sig_stages) >= len(results) / 2:
        print("→ Model has patient fingerprints (majority of stages significant)")
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

    # Right panel: Separation ratio vs null threshold
    ax1 = axes[1]
    sep_ratios = [results[s]['separation_ratio'] for s in stages]
    null_means = [results[s]['null_mean'] for s in stages]
    null_95s = [results[s]['null_95'] for s in stages]

    # Bar chart: observed separation ratio
    bars3 = ax1.bar(x, sep_ratios, width * 0.8, label='Observed Sep. Ratio',
                    color='tab:green', alpha=0.8)

    # Plot null mean as horizontal markers with error showing 95th percentile
    ax1.scatter(x, null_means, color='tab:red', marker='_', s=200, linewidths=2,
                label='Null Mean', zorder=5)

    # Plot 95th percentile as threshold line per stage
    for i, (xi, null_95) in enumerate(zip(x, null_95s)):
        ax1.hlines(y=null_95, xmin=xi - width/2, xmax=xi + width/2,
                   colors='tab:red', linestyles='--', alpha=0.7,
                   label='Null 95th %ile' if i == 0 else None)

    ax1.set_xlabel('Sleep Stage')
    ax1.set_ylabel('Separation Ratio (inter/intra)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_title('Separation Ratio vs Null Hypothesis')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

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
    parser.add_argument('--folder', type=str, required=True,
                        help='Directory containing embedding files')
    parser.add_argument('--splits', type=str, nargs='+', default=['val'],
                        choices=['train', 'val', 'test'],
                        help='Which split(s) to analyze (default: val). Can specify multiple.')
    args = parser.parse_args()

    embeddings_dir = Path(args.folder)

    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory not found: {embeddings_dir}")
        sys.exit(1)

    results = analyze_patient_separation_per_stage(embeddings_dir, splits=args.splits)

    # Generate plot
    output_name = embeddings_dir.name
    split_str = '+'.join(args.splits)
    plot_patient_separation(results, output_name, split_str)