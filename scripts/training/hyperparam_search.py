"""
Hyperparameter Grid Search for Contrastive Learning.

Reads search configuration from YAML file and runs experiments
for specified model types. Only saves the best model per model type.

Usage:
    # Run search for a specific model
    python scripts/training/hyperparam_search.py --model gcn
    python scripts/training/hyperparam_search.py --model temporal_gcn

    # Run search for all models
    python scripts/training/hyperparam_search.py --model all

    # Use custom config file
    python scripts/training/hyperparam_search.py --model gcn --config my_config.yaml

    # Dry run (show combinations without running)
    python scripts/training/hyperparam_search.py --model gcn --dry-run
"""

import argparse
import csv
import itertools
from datetime import datetime
from pathlib import Path

import yaml

import sys
sys.path.append('src')
sys.path.append('scripts/training')

from config import LOGS_DIR
from train_contrastive import train, TrainConfig


DEFAULT_CONFIG_PATH = Path('configs/hyperparam_search.yaml')


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_config(config: dict, model_name: str) -> dict:
    """
    Get configuration for a specific model, merging with defaults.

    Returns dict with keys: dataset_type, model_type, fixed, grid
    """
    if model_name not in config:
        available = [k for k in config.keys() if k != 'defaults']
        raise ValueError(f"Model '{model_name}' not found in config. Available: {available}")

    defaults = config.get('defaults', {})
    model_cfg = config[model_name]

    # Merge fixed params (model-specific overrides defaults)
    fixed = {**defaults.get('fixed', {}), **model_cfg.get('fixed', {})}

    # Use model-specific grid if provided, else default grid
    grid = model_cfg.get('grid', defaults.get('grid', {}))

    return {
        'dataset_type': model_cfg['dataset_type'],
        'model_type': model_cfg['model_type'],
        'fixed': fixed,
        'grid': grid,
    }


def get_all_model_names(config: dict) -> list:
    """Get list of all model names (excluding 'defaults')."""
    return [k for k in config.keys() if k != 'defaults']


def generate_combinations(grid: dict) -> list:
    """Generate all parameter combinations from a grid."""
    if not grid:
        return [{}]

    param_names = list(grid.keys())
    param_values = list(grid.values())
    combinations = list(itertools.product(*param_values))

    return [dict(zip(param_names, values)) for values in combinations]


def run_experiment(params: dict, dataset_type: str, model_type: str, fixed: dict, save_model: bool = False) -> dict:
    """
    Run a single training experiment with given parameters.

    Args:
        params: Hyperparameters from grid search
        dataset_type: 'single_epoch' or 'temporal'
        model_type: Model architecture name
        fixed: Fixed parameters not being searched
        save_model: Whether to save model and embeddings

    Returns:
        dict with results
    """
    config = TrainConfig(
        dataset_type=dataset_type,
        model_type=model_type,
        n_temporal_steps=fixed.get('n_temporal_steps', 3),
        temperature=params.get('temperature', 0.07),
        lr=params.get('lr', 1e-3),
        dropout=params.get('dropout', 0.3),
        epochs=fixed.get('epochs', 50),
        patience=fixed.get('patience', 10),
        batch_size=fixed.get('batch_size', 32),
        hidden_dim=fixed.get('hidden_dim', 128),
        embedding_dim=fixed.get('embedding_dim', 64),
        n_layers=fixed.get('n_layers', 3),
        k=fixed.get('k', 10),
        seed=fixed.get('seed', 42),
        verbose=False,
        save_model=save_model,
    )

    try:
        result = train(config)
        return {
            'status': 'success',
            'best_avg_sep_ratio': result['best_avg_sep_ratio'],
            'test_avg_sep_ratio': result['test_metrics']['avg_sep_ratio'],
            'test_stage_sil': result['test_metrics']['stage_sil'],
            'test_metrics': result['test_metrics'],
            'run_name': result['run_name'],
            'model_state': result['model_state'],
            'config': result['config'],
            **params
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e), **params}


def save_best_model(result: dict, models_dir: Path, data_dir: Path):
    """
    Save the best model and extract train/val/test embeddings.

    Args:
        result: Training result dict with model_state, config, metrics
        models_dir: Directory to save model checkpoint
        data_dir: Directory to save embeddings
    """
    import torch
    import numpy as np
    from torch_geometric.loader import DataLoader
    from train_contrastive import get_model, get_dataset_class, extract_embeddings, load_patient_splits
    from config import DEVICE

    config = result['config']
    model_state = result['model_state']
    run_name = result['run_name']

    # Create model and load state
    model = get_model(config.model_type, config).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    # Save model directly to models/ (not a subdirectory)
    model_path = models_dir / f"{run_name}.pt"

    torch.save({
        'model_state_dict': model_state,
        'run_name': run_name,
        'config': config.__dict__,
        'best_avg_sep_ratio': result['best_avg_sep_ratio'],
        'test_metrics': result['test_metrics'],
    }, model_path)

    # Save embeddings to data/graph_embeddings/{run_name}/ (includes hyperparam config)
    splits = load_patient_splits()
    DatasetClass = get_dataset_class(config.dataset_type)

    emb_dir = data_dir / 'graph_embeddings' / run_name
    emb_dir.mkdir(exist_ok=True, parents=True)

    for split_name in ['train', 'val', 'test']:
        if config.dataset_type == 'temporal':
            dataset = DatasetClass(splits[split_name], n_temporal_steps=config.n_temporal_steps, k=config.k, verbose=False)
        else:
            dataset = DatasetClass(splits[split_name], k=config.k, verbose=False)

        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        emb, labels, patients = extract_embeddings(model, loader, DEVICE)

        np.save(emb_dir / f'{split_name}_embeddings.npy', emb)
        np.save(emb_dir / f'{split_name}_labels.npy', labels)
        np.save(emb_dir / f'{split_name}_patient_ids.npy', np.array(patients))

    return model_path


def run_search_for_model(model_name: str, model_cfg: dict, results_dir: Path, dry_run: bool = False) -> dict:
    """
    Run hyperparameter search for a single model.

    Tracks best model in memory, saves only once at the end.

    Returns:
        dict with 'results' list and 'best' result (or None if dry_run)
    """
    dataset_type = model_cfg['dataset_type']
    model_type = model_cfg['model_type']
    fixed = model_cfg['fixed']
    grid = model_cfg['grid']

    combinations = generate_combinations(grid)

    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Model type: {model_type}")
    print(f"  Fixed params: {fixed}")
    print(f"  Search grid: {grid}")
    print(f"  Total combinations: {len(combinations)}")

    if dry_run:
        print("\n  Combinations:")
        for i, combo in enumerate(combinations, 1):
            print(f"    {i}. {combo}")
        return {'results': None, 'best': None}

    # Setup CSV logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = results_dir / f'search_{model_name}_{timestamp}.csv'

    # Determine CSV columns from grid params
    grid_params = list(grid.keys())
    columns = ['status'] + grid_params + ['best_avg_sep_ratio', 'test_avg_sep_ratio', 'test_stage_sil']

    results = []
    best_result = None  # Stores full result including model_state
    best_val_sep = -float('inf')

    # Print header
    header_parts = ['Idx'] + [p[:6] for p in grid_params] + ['val_sep', 'test_sep', 'best?']
    print(f"\n  {' '.join(f'{h:<10}' for h in header_parts)}")
    print("  " + "-" * (10 * len(header_parts)))

    # Run all experiments without saving to disk
    for i, params in enumerate(combinations, 1):
        result = run_experiment(params, dataset_type, model_type, fixed, save_model=False)
        results.append(result)

        val_sep = result.get('best_avg_sep_ratio', 0) or 0
        test_sep = result.get('test_avg_sep_ratio', 0) or 0

        # Check if new best - keep model state in memory
        is_new_best = (result.get('status') == 'success' and val_sep > best_val_sep)
        best_marker = ""

        if is_new_best:
            best_result = result  # Keep full result with model_state
            best_val_sep = val_sep
            best_marker = "*"

        # Print result row
        row_parts = [str(i)] + [f"{params[p]:.0e}" if isinstance(params[p], float) and params[p] < 0.01 else str(params[p]) for p in grid_params]
        row_parts += [f"{val_sep:.4f}", f"{test_sep:.4f}", best_marker]
        print(f"  {' '.join(f'{r:<10}' for r in row_parts)}")

        # Save intermediate results to CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

    # Save the best model to disk
    if best_result and best_result.get('model_state'):
        from config import MODELS_DIR, DATA_DIR
        model_path = save_best_model(best_result, MODELS_DIR, DATA_DIR)

        print(f"\n  Best configuration:")
        for p in grid_params:
            print(f"    {p}: {best_result[p]}")
        print(f"    val_sep: {best_result['best_avg_sep_ratio']:.4f}")
        print(f"    test_sep: {best_result['test_avg_sep_ratio']:.4f}")
        print(f"  Saved model: {model_path}")

    print(f"\n  Results CSV: {csv_path}")
    return {'results': results, 'best': best_result, 'csv_path': csv_path}


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for contrastive learning')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to search (e.g., gcn, temporal_gcn) or "all"')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH),
                        help='Path to YAML config file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show combinations without running experiments')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    config = load_config(config_path)

    # Setup results directory
    results_dir = LOGS_DIR / 'hyperparam_search'
    results_dir.mkdir(exist_ok=True, parents=True)

    print(f"Hyperparameter Search")
    print(f"Config: {config_path}")

    # Determine which models to run
    if args.model == 'all':
        model_names = get_all_model_names(config)
        print(f"Running search for all models: {model_names}")
    else:
        model_names = [args.model]

    # Run search for each model
    all_results = {}
    for model_name in model_names:
        try:
            model_cfg = get_model_config(config, model_name)
            search_output = run_search_for_model(model_name, model_cfg, results_dir, args.dry_run)
            all_results[model_name] = search_output
        except ValueError as e:
            print(f"Error: {e}")
            continue

    # Save summary CSV for multi-model runs
    if len(model_names) > 1 and not args.dry_run:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        summary_rows = []
        for model_name, output in all_results.items():
            best = output.get('best')
            if best:
                print(f"  {model_name}: val_sep={best['best_avg_sep_ratio']:.4f}, test_sep={best['test_avg_sep_ratio']:.4f}")
                summary_rows.append({
                    'model': model_name,
                    'temperature': best.get('temperature'),
                    'lr': best.get('lr'),
                    'dropout': best.get('dropout'),
                    'best_val_sep_ratio': best['best_avg_sep_ratio'],
                    'test_sep_ratio': best['test_avg_sep_ratio'],
                    'test_stage_sil': best.get('test_stage_sil'),
                })

        # Save summary CSV
        if summary_rows:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_path = results_dir / f'summary_all_models_{timestamp}.csv'
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"\n  Summary CSV: {summary_path}")


if __name__ == '__main__':
    main()
