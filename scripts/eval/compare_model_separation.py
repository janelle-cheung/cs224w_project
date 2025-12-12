"""
Compare Patient Separation Across All Models and Baselines.

Auto-discovers embedding folders from:
- data/baseline_embeddings/* (Power, Connectivity baselines)
- data/graph_embeddings/* (GNN model embeddings from hyperparam_search.py)

Computes per-stage patient separation ratios and generates comparison plots.

Usage:
    python scripts/eval/compare_model_separation.py
    python scripts/eval/compare_model_separation.py --split val
    python scripts/eval/compare_model_separation.py --split test
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import sys

sys.path.append('src')

from config import DATA_DIR, FIGURES_DIR, LOGS_DIR, STAGE_NAMES


def load_embeddings_from_folder(folder: Path, split: str) -> tuple:
    """
    Load embeddings from a folder for the specified split.

    Embeddings folders can be:
    - GNN models: data/graph_embeddings/{run_name}/ with {split}_embeddings.npy, {split}_labels.npy, {split}_patient_ids.npy
    - Baselines: data/baseline_embeddings/{name}/ with {split}_embeddings.npy (labels/patient_ids from patient_splits)

    Returns:
        embeddings: np.ndarray of shape (n_epochs, embedding_dim)
        labels: np.ndarray of shape (n_epochs,) with sleep stage labels
        patient_ids: np.ndarray of patient IDs for each epoch
    """
    patient_splits_dir = DATA_DIR / 'patient_splits'

    emb_file = folder / f'{split}_embeddings.npy'
    if not emb_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_file}")

    embeddings = np.load(emb_file)

    # Labels: check folder first, then patient_splits (for baselines)
    labels_file = folder / f'{split}_labels.npy'
    if not labels_file.exists():
        labels_file = patient_splits_dir / f'{split}_labels.npy'
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels not found for {folder.name}")

    # Patient IDs: check folder first, then patient_splits (for baselines)
    patient_file = folder / f'{split}_patient_ids.npy'
    if not patient_file.exists():
        patient_file = patient_splits_dir / f'{split}_patient_ids.npy'
    if not patient_file.exists():
        raise FileNotFoundError(f"Patient IDs not found for {folder.name}")

    labels = np.load(labels_file)
    patient_ids = np.load(patient_file, allow_pickle=True)

    # Verify sizes match
    if len(labels) != len(embeddings):
        raise ValueError(f"Size mismatch in {folder.name}: emb={len(embeddings)}, labels={len(labels)}")
    if len(patient_ids) != len(embeddings):
        raise ValueError(f"Size mismatch in {folder.name}: emb={len(embeddings)}, pids={len(patient_ids)}")

    return embeddings, labels, patient_ids


def get_display_name(folder: Path) -> str:
    """
    Generate a clean display name from folder path.

    Examples:
        gcn_single_epoch_k10_e64_temp0.20_lr1e-04 -> GCN
        temporal_gcn_temporal_t3_k10_e64_temp0.10_lr5e-04 -> T-GCN
        power64 -> Power
        connectivity64 -> Connectivity
    """
    name = folder.name.lower()

    # Baselines
    if 'power' in name:
        return 'Power'
    if 'connectivity' in name:
        return 'Connectivity'

    # GNN models - extract model type
    if 'temporal_gin' in name:
        return 'T-GIN'
    if 'temporal_gat' in name:
        return 'T-GAT'
    if 'temporal_gcn' in name:
        return 'T-GCN'
    if 'gin' in name:
        return 'GIN'
    if 'gat' in name:
        return 'GAT'
    if 'gcn' in name:
        return 'GCN'

    # Fallback: use folder name
    return folder.name


def compute_separation_ratio_per_stage(embeddings, labels, patient_ids, normalize=True) -> dict:
    """
    Compute patient separation ratio (inter/intra) for each sleep stage.

    Args:
        embeddings: np.ndarray of shape (n_epochs, embedding_dim)
        labels: np.ndarray of shape (n_epochs,) with sleep stage labels
        patient_ids: list/array of patient IDs
        normalize: If True, L2-normalize embeddings (should match training)

    Returns:
        dict mapping stage name to separation ratio
    """
    # L2-normalize embeddings to match contrastive training
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

    # Convert patient IDs to numeric
    unique_patients = sorted(set(patient_ids))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_numeric = np.array([patient_to_idx[p] for p in patient_ids])

    sep_ratios = {}

    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        mask = labels == stage_idx

        if mask.sum() < 2:
            sep_ratios[stage_name] = 0.0
            continue

        stage_emb = embeddings[mask]
        stage_patients = patient_numeric[mask]

        # Need at least 2 unique patients
        if len(np.unique(stage_patients)) < 2:
            sep_ratios[stage_name] = 0.0
            continue

        # Compute pairwise distances
        dist_matrix = squareform(pdist(stage_emb))

        # Separate intra vs inter patient distances
        same_patient = stage_patients[:, None] == stage_patients[None, :]
        upper_tri = np.triu(np.ones_like(same_patient, dtype=bool), k=1)

        intra_mask = same_patient & upper_tri
        inter_mask = ~same_patient & upper_tri

        intra_dists = dist_matrix[intra_mask]
        inter_dists = dist_matrix[inter_mask]

        if len(intra_dists) == 0:
            sep_ratios[stage_name] = 0.0
        else:
            sep_ratios[stage_name] = np.mean(inter_dists) / np.mean(intra_dists)

    return sep_ratios


def plot_comparison(all_results: dict, split: str, output_path: Path):
    """
    Create a grouped bar plot comparing separation ratios across all models.

    Args:
        all_results: dict mapping model name to {stage: sep_ratio}
        split: 'train', 'val', or 'test'
        output_path: Path to save the figure
    """
    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_stages = len(STAGE_NAMES)

    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Bar positions
    x = np.arange(n_stages)
    width = 0.8 / n_models  # Width of each bar

    # Colors for different model types
    colors = {
        'Power': '#7f7f7f',       # Gray (baseline)
        'Connectivity': '#9f9f9f', # Lighter gray (baseline)
        'GCN': '#1f77b4',         # Blue
        'GAT': '#ff7f0e',         # Orange
        'GIN': '#2ca02c',         # Green
        'T-GCN': '#d62728',       # Red
        'T-GAT': '#9467bd',       # Purple
        'T-GIN': '#8c564b',       # Brown
    }

    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        ratios = [all_results[model_name].get(stage, 0) for stage in STAGE_NAMES]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, ratios, width, label=model_name,
                      color=colors.get(model_name, f'C{i}'), alpha=0.85)

    # Add reference line at 1.0 (no separation)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='No separation')

    # Labels and formatting
    ax.set_xlabel('Sleep Stage', fontsize=12)
    ax.set_ylabel('Patient Separation Ratio (inter/intra)', fontsize=12)
    ax.set_title(f'Patient Separation Ratio by Sleep Stage ({split.upper()} set)\nHigher = Better Patient Fingerprinting', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES, fontsize=11)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_path}")


def discover_embedding_folders() -> list:
    """
    Auto-discover embedding folders based on:
    - data/baseline_embeddings/* (all baseline folders)
    - data/graph_embeddings/{name}/ where models/{name}.pt exists (best models only)

    Returns list of (folder_path, display_name) tuples, sorted with baselines first.
    """
    baseline_dir = DATA_DIR / 'baseline_embeddings'
    graph_dir = DATA_DIR / 'graph_embeddings'
    models_dir = DATA_DIR.parent / 'models'

    folders = []

    # Baselines first
    if baseline_dir.exists():
        for folder in sorted(baseline_dir.iterdir()):
            if folder.is_dir():
                folders.append((folder, get_display_name(folder)))

    # GNN models - only include if corresponding .pt file exists in models/
    if graph_dir.exists() and models_dir.exists():
        # Get set of model names from saved .pt files
        saved_models = {f.stem for f in models_dir.glob('*.pt')}

        for folder in sorted(graph_dir.iterdir()):
            if folder.is_dir() and folder.name in saved_models:
                folders.append((folder, get_display_name(folder)))

    return folders


def main():
    parser = argparse.ArgumentParser(description='Compare patient separation across models')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate (default: val)')
    args = parser.parse_args()

    split = args.split

    # Auto-discover embedding folders
    folders = discover_embedding_folders()

    if not folders:
        print("Error: No embedding folders found")
        print(f"Looked in: {DATA_DIR / 'baseline_embeddings'}, {DATA_DIR / 'graph_embeddings'}")
        return

    print("=" * 70)
    print(f"Comparing Patient Separation Across Models ({split.upper()} set)")
    print("=" * 70)
    print(f"Found {len(folders)} embedding folders")

    all_results = {}

    print("\nLoading embeddings...")
    for folder, display_name in folders:
        print(f"  {display_name}...", end=" ", flush=True)

        try:
            emb, labels, patients = load_embeddings_from_folder(folder, split)
            sep_ratios = compute_separation_ratio_per_stage(emb, labels, patients)
            all_results[display_name] = sep_ratios
            avg = np.mean([v for v in sep_ratios.values() if v > 0])
            print(f"avg={avg:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")

    if not all_results:
        print("\nNo embeddings loaded successfully")
        return

    # Print summary table
    print("\n" + "=" * 70)
    print("Per-Stage Separation Ratios")
    print("=" * 70)

    # Header
    header = f"{'Model':<15}" + "".join(f"{s:<10}" for s in STAGE_NAMES) + f"{'Avg':<10}"
    print(header)
    print("-" * len(header))

    # Data rows
    for model_name, ratios in all_results.items():
        row = f"{model_name:<15}"
        valid_ratios = []
        for stage in STAGE_NAMES:
            r = ratios.get(stage, 0)
            row += f"{r:<10.3f}"
            if r > 0:
                valid_ratios.append(r)
        avg = np.mean(valid_ratios) if valid_ratios else 0
        row += f"{avg:<10.3f}"
        print(row)

    # Generate plot
    output_path = FIGURES_DIR / 'eval' / f'model_comparison_separation_{split}.png'
    plot_comparison(all_results, split, output_path)

    # Save results to CSV
    csv_path = LOGS_DIR / 'eval' / f'model_comparison_separation_{split}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w') as f:
        f.write("model," + ",".join(STAGE_NAMES) + ",avg\n")
        for model_name, ratios in all_results.items():
            valid_ratios = [ratios.get(s, 0) for s in STAGE_NAMES]
            avg = np.mean([r for r in valid_ratios if r > 0]) if any(r > 0 for r in valid_ratios) else 0
            f.write(f"{model_name}," + ",".join(f"{r:.4f}" for r in valid_ratios) + f",{avg:.4f}\n")

    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()
