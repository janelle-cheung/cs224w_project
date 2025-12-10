"""
Clustering analysis for sleep stage embeddings.
Runs k-means clustering with k=1 to 10, computes BIC, finds elbow point,
and validates against ground truth labels.

Usage:
    python scripts/clustering_analysis.py --features_dir data/power64
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append('src')

from config import DATA_DIR, LOGS_DIR, FIGURES_DIR

LABELS_DIR = DATA_DIR / 'processed' / 'sleep_stage_labels'


def compute_bic(X: np.ndarray, kmeans: KMeans) -> float:
    """
    Compute Bayesian Information Criterion for k-means clustering.
    BIC = -2 * log_likelihood + k * log(n)
    """
    n_samples, n_features = X.shape
    n_clusters = kmeans.n_clusters

    # Compute variance
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Sum of squared distances to cluster centers
    ss_within = 0
    for k in range(n_clusters):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            ss_within += np.sum((cluster_points - cluster_centers[k]) ** 2)

    # Variance estimate
    variance = ss_within / (n_samples - n_clusters) if n_samples > n_clusters else 1e-10

    # Log likelihood (assuming Gaussian clusters)
    log_likelihood = -0.5 * n_samples * n_features * np.log(2 * np.pi * variance)
    log_likelihood -= 0.5 * ss_within / variance

    # Number of parameters: k*d (centers) + k-1 (mixing) + 1 (variance)
    n_params = n_clusters * n_features + n_clusters

    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return bic


def compute_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute clustering purity: fraction of samples correctly assigned
    to majority class in each cluster.
    """
    n_samples = len(labels_true)
    purity = 0

    for cluster_id in np.unique(labels_pred):
        cluster_mask = labels_pred == cluster_id
        if cluster_mask.sum() == 0:
            continue
        true_labels_in_cluster = labels_true[cluster_mask]
        most_common_count = np.bincount(true_labels_in_cluster.astype(int)).max()
        purity += most_common_count

    return purity / n_samples


def find_elbow(k_values: list, inertias: list) -> int:
    """
    Find elbow point using the kneedle algorithm (max perpendicular distance).
    """
    # Normalize to [0, 1]
    k_norm = (np.array(k_values) - k_values[0]) / (k_values[-1] - k_values[0])
    inertia_norm = (np.array(inertias) - min(inertias)) / (max(inertias) - min(inertias) + 1e-10)

    # Line from first to last point
    p1 = np.array([k_norm[0], inertia_norm[0]])
    p2 = np.array([k_norm[-1], inertia_norm[-1]])

    # Find point with max perpendicular distance to line
    max_dist = 0
    elbow_idx = 0

    for i in range(1, len(k_values) - 1):
        point = np.array([k_norm[i], inertia_norm[i]])
        # Perpendicular distance from point to line
        dist = np.abs(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)
        if dist > max_dist:
            max_dist = dist
            elbow_idx = i

    return k_values[elbow_idx]


def analyze_patient(features: np.ndarray, labels: np.ndarray, k_range: range) -> dict:
    """
    Run clustering analysis for a single patient.
    Returns dict with all metrics.
    """
    # Filter out invalid labels (L = -1)
    valid_mask = labels[:, 1] >= 0

    # Align features and labels by epoch index
    n_epochs = min(len(features), len(labels))
    features = features[:n_epochs]
    labels = labels[:n_epochs]
    valid_mask = valid_mask[:n_epochs]

    features_valid = features[valid_mask]
    labels_valid = labels[valid_mask, 1]

    results = {
        'n_epochs': len(features_valid),
        'n_stages': len(np.unique(labels_valid)),
        'k_values': [],
        'inertias': [],
        'bics': [],
        'silhouettes': [],
        'purities': [],
    }

    for k in k_range:
        if k > len(features_valid):
            break

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_valid)

        results['k_values'].append(k)
        results['inertias'].append(kmeans.inertia_)
        results['bics'].append(compute_bic(features_valid, kmeans))

        if k > 1:
            results['silhouettes'].append(silhouette_score(features_valid, kmeans.labels_))
            results['purities'].append(compute_purity(labels_valid, kmeans.labels_))
        else:
            results['silhouettes'].append(0)
            results['purities'].append(0)

    # Find elbow
    if len(results['k_values']) > 2:
        results['elbow_k'] = find_elbow(results['k_values'], results['inertias'])
    else:
        results['elbow_k'] = results['k_values'][0]

    # Best k by BIC (minimum)
    results['best_k_bic'] = results['k_values'][np.argmin(results['bics'])]

    return results


def main():
    parser = argparse.ArgumentParser(description='Clustering analysis for sleep embeddings')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing feature files')
    parser.add_argument('--k_max', type=int, default=10,
                        help='Maximum k for k-means')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Name for output log file (default: derived from features_dir)')
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = DATA_DIR / args.features_dir.replace('data/', '')

    k_range = range(1, args.k_max + 1)

    # Find feature files
    feature_files = sorted(features_dir.glob(f"*.npy"))
    print(f"Found {len(feature_files)} patients in {features_dir}")
    print(f"K range: {list(k_range)}\n")

    all_results = {}

    for feature_file in feature_files:
        patient_id = feature_file.name.split('_')[0]
        label_file = LABELS_DIR / f'{patient_id}_labels.npy'

        if not label_file.exists():
            print(f"  {patient_id}: WARNING - no labels found, skipping")
            continue

        features = np.load(feature_file)
        labels = np.load(label_file)

        results = analyze_patient(features, labels, k_range)
        all_results[patient_id] = results

        print(f"  {patient_id}: {results['n_epochs']} epochs, {results['n_stages']} stages | "
              f"elbow={results['elbow_k']}, best_bic={results['best_k_bic']}")

    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    # Average metrics per k
    k_values = list(k_range)
    avg_metrics = {k: {'bic': [], 'silhouette': [], 'purity': []} for k in k_values}

    for patient_id, results in all_results.items():
        for i, k in enumerate(results['k_values']):
            avg_metrics[k]['bic'].append(results['bics'][i])
            avg_metrics[k]['silhouette'].append(results['silhouettes'][i])
            avg_metrics[k]['purity'].append(results['purities'][i])

    print(f"\n{'k':>3} | {'BIC':>12} | {'Silhouette':>10} | {'Purity':>8}")
    print("-" * 45)

    for k in k_values:
        if avg_metrics[k]['bic']:
            avg_bic = np.mean(avg_metrics[k]['bic'])
            avg_sil = np.mean(avg_metrics[k]['silhouette'])
            avg_pur = np.mean(avg_metrics[k]['purity'])
            print(f"{k:>3} | {avg_bic:>12.1f} | {avg_sil:>10.3f} | {avg_pur:>8.3f}")

    # Elbow distribution
    elbows = [r['elbow_k'] for r in all_results.values()]
    print(f"\nElbow distribution: {dict(zip(*np.unique(elbows, return_counts=True)))}")

    # Best k by BIC distribution
    best_bics = [r['best_k_bic'] for r in all_results.values()]
    print(f"Best k (BIC) distribution: {dict(zip(*np.unique(best_bics, return_counts=True)))}")

    # Save detailed log
    output_name = args.output_name or features_dir.name
    log_dir = LOGS_DIR / 'analysis'
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f'clustering_{output_name}.txt'

    with open(log_file, 'w') as f:
        f.write(f"Clustering Analysis: {features_dir.name}\n")
        f.write(f"K range: 1-{args.k_max}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Patient Results:\n")
        f.write("-" * 60 + "\n")
        for patient_id, results in all_results.items():
            f.write(f"\n{patient_id}:\n")
            f.write(f"  Epochs: {results['n_epochs']}, Unique stages: {results['n_stages']}\n")
            f.write(f"  Elbow k: {results['elbow_k']}, Best BIC k: {results['best_k_bic']}\n")
            f.write(f"  k  | BIC         | Silhouette | Purity\n")
            for i, k in enumerate(results['k_values']):
                f.write(f"  {k:>2} | {results['bics'][i]:>11.1f} | {results['silhouettes'][i]:>10.3f} | {results['purities'][i]:>6.3f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Aggregate Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'k':>3} | {'Avg BIC':>12} | {'Avg Silhouette':>14} | {'Avg Purity':>10}\n")
        for k in k_values:
            if avg_metrics[k]['bic']:
                f.write(f"{k:>3} | {np.mean(avg_metrics[k]['bic']):>12.1f} | "
                        f"{np.mean(avg_metrics[k]['silhouette']):>14.3f} | "
                        f"{np.mean(avg_metrics[k]['purity']):>10.3f}\n")

        f.write(f"\nElbow distribution: {dict(zip(*np.unique(elbows, return_counts=True)))}\n")
        f.write(f"Best k (BIC) distribution: {dict(zip(*np.unique(best_bics, return_counts=True)))}\n")

    print(f"\nLog saved to {log_file}")

    # Create plots
    # plot_bic_curve(k_values, avg_metrics, elbows, output_name)
    # plot_individual_bic_curves(all_results, output_name)
    plot_individual_metrics(all_results, output_name)


def plot_bic_curve(k_values: list, avg_metrics: dict, elbows: list, output_name: str):
    """
    Plot average BIC curve across patients with elbow marked and purity annotated.
    """
    # Compute averages
    avg_bics = [np.mean(avg_metrics[k]['bic']) for k in k_values if avg_metrics[k]['bic']]
    avg_purities = [np.mean(avg_metrics[k]['purity']) for k in k_values if avg_metrics[k]['purity']]
    avg_silhouettes = [np.mean(avg_metrics[k]['silhouette']) for k in k_values if avg_metrics[k]['silhouette']]
    valid_k = [k for k in k_values if avg_metrics[k]['bic']]

    # Find most common elbow
    elbow_counts = dict(zip(*np.unique(elbows, return_counts=True)))
    most_common_elbow = max(elbow_counts, key=elbow_counts.get)
    elbow_idx = valid_k.index(most_common_elbow)
    elbow_bic = avg_bics[elbow_idx]
    elbow_purity = avg_purities[elbow_idx]
    elbow_silhouette = avg_silhouettes[elbow_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot BIC curve
    ax.plot(valid_k, avg_bics, 'b-o', linewidth=2, markersize=8, label='Avg BIC')

    # Mark elbow point
    ax.scatter([most_common_elbow], [elbow_bic], color='red', s=200, zorder=5,
               marker='*', label=f'Elbow (k={most_common_elbow})')

    # Annotate with silhouette and purity
    ax.annotate(f'k={most_common_elbow}\nSil={elbow_silhouette:.3f}\nPurity={elbow_purity:.3f}',
                xy=(most_common_elbow, elbow_bic),
                xytext=(most_common_elbow + 1, elbow_bic + (max(avg_bics) - min(avg_bics)) * 0.1),
                fontsize=11,
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('BIC', fontsize=12)
    ax.set_title(f'K-Means BIC Curve: {output_name}', fontsize=14)
    ax.set_xticks(valid_k)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_dir = FIGURES_DIR / 'clustering'
    fig_dir.mkdir(exist_ok=True, parents=True)
    plot_file = fig_dir / f'bic_avg_{output_name}.png'
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"Plot saved to {plot_file}")

def plot_individual_bic_curves(all_results: dict, output_name: str):
    """
    Plot individual BIC curves for all patients in a grid.
    """
    n_patients = len(all_results)
    n_cols = 5
    n_rows = (n_patients + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten() if n_patients > 1 else [axes]

    for idx, (patient_id, results) in enumerate(sorted(all_results.items())):
        ax = axes[idx]

        k_values = results['k_values']
        bics = results['bics']
        elbow_k = results['elbow_k']

        # Plot BIC curve
        ax.plot(k_values, bics, 'b-o', linewidth=1.5, markersize=5)

        # Mark elbow
        elbow_idx = k_values.index(elbow_k)
        ax.scatter([elbow_k], [bics[elbow_idx]], color='red', s=100,
                   marker='*', zorder=5)

        # Get purity at elbow
        purity = results['purities'][elbow_idx]

        ax.set_title(f'{patient_id} (k={elbow_k}, p={purity:.2f})', fontsize=10)
        ax.set_xlabel('k', fontsize=8)
        ax.set_ylabel('BIC', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for idx in range(n_patients, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Individual BIC Curves: {output_name}', fontsize=14, y=0.995)
    plt.tight_layout()

    fig_dir = FIGURES_DIR / 'clustering'
    fig_dir.mkdir(exist_ok=True, parents=True)
    plot_file = fig_dir / f'bic_all_{output_name}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Individual plot saved to {plot_file}")


def plot_individual_metrics(all_results: dict, output_name: str):
    """
    Plot BIC, silhouette, and purity on the same subplot for each patient.
    Each subplot has 3 y-axes, and the elbow point is marked with a star on BIC.
    """
    n_patients = len(all_results)
    n_cols = 5
    n_rows = (n_patients + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_patients > 1 else [axes]

    for idx, (patient_id, results) in enumerate(sorted(all_results.items())):
        ax1 = axes[idx]

        k_values = results['k_values']
        bics = results['bics']
        silhouettes = results['silhouettes']
        purities = results['purities']
        elbow_k = results['elbow_k']
        elbow_idx = k_values.index(elbow_k)

        # Normalize BIC for better visualization (scale to 0-1 range)
        bic_min, bic_max = min(bics), max(bics)
        bic_range = bic_max - bic_min if bic_max > bic_min else 1

        # Plot BIC (left y-axis, blue)
        color1 = 'tab:blue'
        ax1.set_xlabel('k', fontsize=8)
        ax1.set_ylabel('BIC', color=color1, fontsize=8)
        line1, = ax1.plot(k_values, bics, color=color1, marker='o', linewidth=1.5,
                          markersize=4, label='BIC')
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=7)
        ax1.tick_params(axis='x', labelsize=7)

        # Mark elbow on BIC curve
        ax1.scatter([elbow_k], [bics[elbow_idx]], color='red', s=150,
                    marker='*', zorder=10, label=f'Elbow (k={elbow_k})')

        # Create second y-axis for silhouette (right side, orange)
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Silhouette', color=color2, fontsize=8)
        line2, = ax2.plot(k_values, silhouettes, color=color2, marker='s',
                          linewidth=1.5, markersize=4, linestyle='--', label='Silhouette')
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=7)
        ax2.set_ylim(-0.2, 0.6)

        # Create third y-axis for purity (far right, green)
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 45))
        color3 = 'tab:green'
        ax3.set_ylabel('Purity', color=color3, fontsize=8)
        line3, = ax3.plot(k_values, purities, color=color3, marker='^',
                          linewidth=1.5, markersize=4, linestyle=':', label='Purity')
        ax3.tick_params(axis='y', labelcolor=color3, labelsize=7)
        ax3.set_ylim(0, 1)

        # Title with elbow info
        ax1.set_title(f'{patient_id} (elbow k={elbow_k})', fontsize=10)
        ax1.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_patients, len(axes)):
        axes[idx].axis('off')

    # Add shared legend
    fig.legend([line1, line2, line3], ['BIC', 'Silhouette', 'Purity'],
               loc='upper right', fontsize=10, bbox_to_anchor=(0.99, 0.99))

    fig.suptitle(f'Clustering Metrics per Patient: {output_name}', fontsize=14, y=1.0)
    plt.tight_layout()

    fig_dir = FIGURES_DIR / 'clustering'
    fig_dir.mkdir(exist_ok=True, parents=True)
    plot_file = fig_dir / f'metrics_per_patient_{output_name}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Metrics plot saved to {plot_file}")


if __name__ == '__main__':
    main()
