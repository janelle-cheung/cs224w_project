"""
Hyperparameter sweep for contrastive learning.

Runs multiple training runs with different hyperparameter combinations
and logs results to a summary file.

Usage:
    python scripts/training/sweep_hyperparams.py
    python scripts/training/sweep_hyperparams.py --k 10 20 --lr 0.001 0.01 --embedding_dim 32 64
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

sys.path.append('src')
from config import LOGS_DIR

TRAINING_LOGS_DIR = LOGS_DIR / 'training'

def run_training(k, lr, embedding_dim, temperature, epochs=100, patience=3):
    """Run a single training with given hyperparameters."""
    cmd = [
        sys.executable, 'scripts/training/train_contrastive.py',
        '--k', str(k),
        '--lr', str(lr),
        '--embedding_dim', str(embedding_dim),
        '--temperature', str(temperature),
        '--epochs', str(epochs),
        '--patience', str(patience)
    ]

    print(f"\n{'='*70}")
    print(f"Running: k={k}, lr={lr}, emb_dim={embedding_dim}, temp={temperature}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    return result.returncode == 0


def parse_csv_results(run_name):
    """Parse the CSV log to get best metrics."""
    csv_path = TRAINING_LOGS_DIR / f"{run_name}.csv"

    if not csv_path.exists():
        return None

    best_row = None
    best_avg_patient_sil = -float('inf')

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_sil = float(row['avg_patient_sil'])
            if avg_sil > best_avg_patient_sil:
                best_avg_patient_sil = avg_sil
                best_row = row

    return best_row


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for contrastive learning')
    parser.add_argument('--k', type=int, nargs='+', default=[10, 20, 30],
                        help='List of k values (top correlations per node)')
    parser.add_argument('--lr', type=float, nargs='+', default=[0.001, 0.01],
                        help='List of learning rates')
    parser.add_argument('--embedding_dim', type=int, nargs='+', default=[32, 64],
                        help='List of embedding dimensions')
    parser.add_argument('--temperature', type=float, nargs='+', default=[0.3, 0.5, 1.0],
                        help='List of temperature values')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs per run')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    args = parser.parse_args()

    # Generate all combinations
    combos = list(product(args.k, args.lr, args.embedding_dim, args.temperature))
    n_combos = len(combos)

    print("=" * 70)
    print("Hyperparameter Sweep Configuration")
    print("=" * 70)
    print(f"k values: {args.k}")
    print(f"Learning rates: {args.lr}")
    print(f"Embedding dims: {args.embedding_dim}")
    print(f"Temperatures: {args.temperature}")
    print(f"Total combinations: {n_combos}")
    print(f"Max epochs per run: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print("=" * 70 + "\n")

    # Setup summary log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = TRAINING_LOGS_DIR / f"sweep_summary_{timestamp}.csv"

    summary_columns = [
        'k', 'lr', 'embedding_dim', 'temperature', 'run_name',
        'best_epoch', 'train_loss', 'val_loss', 'stage_sil',
        'patient_sil_W', 'patient_sil_N1', 'patient_sil_N2',
        'patient_sil_N3', 'patient_sil_R', 'avg_patient_sil', 'status'
    ]

    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()

    print(f"Summary will be saved to: {summary_path}\n")

    # Track results
    results = []

    for i, (k, lr, emb_dim, temp) in enumerate(combos, 1):
        print(f"\n[{i}/{n_combos}] Starting run...")

        run_name = f"gcn_k{k}_e{emb_dim}_t{temp:.2f}_lr{lr:.0e}"

        success = run_training(k, lr, emb_dim, temp, args.epochs, args.patience)

        if success:
            best_row = parse_csv_results(run_name)
            if best_row:
                result = {
                    'k': k,
                    'lr': lr,
                    'embedding_dim': emb_dim,
                    'temperature': temp,
                    'run_name': run_name,
                    'best_epoch': best_row['epoch'],
                    'train_loss': best_row['train_loss'],
                    'val_loss': best_row['val_loss'],
                    'stage_sil': best_row['stage_sil'],
                    'patient_sil_W': best_row['patient_sil_W'],
                    'patient_sil_N1': best_row['patient_sil_N1'],
                    'patient_sil_N2': best_row['patient_sil_N2'],
                    'patient_sil_N3': best_row['patient_sil_N3'],
                    'patient_sil_R': best_row['patient_sil_R'],
                    'avg_patient_sil': best_row['avg_patient_sil'],
                    'status': 'success'
                }
            else:
                result = {
                    'k': k, 'lr': lr, 'embedding_dim': emb_dim, 'temperature': temp,
                    'run_name': run_name, 'status': 'no_results'
                }
        else:
            result = {
                'k': k, 'lr': lr, 'embedding_dim': emb_dim, 'temperature': temp,
                'run_name': run_name, 'status': 'failed'
            }

        results.append(result)

        # Append to summary CSV
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_columns)
            writer.writerow(result)

    # Print final summary
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE - SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r.get('status') == 'success']

    if successful:
        # Sort by avg_patient_sil
        successful.sort(key=lambda x: float(x.get('avg_patient_sil', 0)), reverse=True)

        print(f"\nTop 5 configurations by avg_patient_sil:\n")
        print(f"{'Rank':<6} {'k':<4} {'lr':<10} {'emb':<6} {'temp':<6} {'avg_pat_sil':<12} {'stage_sil':<10}")
        print("-" * 60)

        for i, r in enumerate(successful[:5], 1):
            print(f"{i:<6} {r['k']:<4} {r['lr']:<10} {r['embedding_dim']:<6} "
                  f"{r['temperature']:<6} {float(r['avg_patient_sil']):<12.4f} "
                  f"{float(r['stage_sil']):<10.4f}")

        best = successful[0]
        print(f"\nBest configuration:")
        print(f"  k={best['k']}, lr={best['lr']}, embedding_dim={best['embedding_dim']}, "
              f"temperature={best['temperature']}")
        print(f"  avg_patient_sil={float(best['avg_patient_sil']):.4f}")
        print(f"  Run name: {best['run_name']}")

    print(f"\nTotal runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len([r for r in results if r.get('status') != 'success'])}")
    print(f"\nFull summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
