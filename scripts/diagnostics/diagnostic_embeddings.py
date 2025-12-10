"""
Diagnostic: determine if embeddings cluster by sleep stage or patient identity.
Shows which signal dominates the learned representations.

Usage:
    python scripts/diagnostic_embeddings.py --embeddings_dir data/embeddings/__
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist

sys.path.append('src')

def load_embeddings_and_labels(embeddings_dir):
    """Load embeddings and labels from train/val/test splits."""
    embeddings_dir = Path(embeddings_dir)

    # Load embeddings and labels from each split
    all_emb = []
    all_labels = []
    all_patient_ids = []

    for split in ['train', 'val', 'test']:
        emb_file = embeddings_dir / f'{split}_embeddings.npy'
        labels_file = embeddings_dir / f'{split}_labels.npy'
        patient_file = embeddings_dir / f'{split}_patient_ids.npy'

        if emb_file.exists():
            all_emb.append(np.load(emb_file))
            all_labels.append(np.load(labels_file))
            all_patient_ids.append(np.load(patient_file, allow_pickle=True))

    all_emb = np.vstack(all_emb)
    stage_labels = np.concatenate(all_labels)
    patient_ids = np.concatenate(all_patient_ids)

    # Convert patient IDs (e.g., 'P01', 'P02') to numeric labels
    unique_patients = sorted(set(patient_ids))
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_labels = np.array([patient_to_idx[p] for p in patient_ids])

    return all_emb, patient_labels, stage_labels


def compute_stage_names(stage_idx):
    """Convert numeric stage to name."""
    stage_map = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R', -1: 'L'}
    return stage_map.get(stage_idx, 'Unknown')


def main():
    parser = argparse.ArgumentParser(description='Diagnostic: embeddings by stage vs patient')
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory with embeddings (e.g., data/contrastive_embeddings_k10)')
    args = parser.parse_args()
    
    print(f"Loading embeddings from {args.embeddings_dir}...\n")
    all_emb, patient_labels, stage_labels = load_embeddings_and_labels(args.embeddings_dir)
    
    print(f"Loaded {len(all_emb)} embeddings, {len(np.unique(patient_labels))} patients, {len(np.unique(stage_labels))} stages\n")
    
    # ===== GLOBAL SILHOUETTE =====
    print("="*60)
    print("GLOBAL SILHOUETTE SCORES")
    print("="*60)
    
    pat_sil = silhouette_score(all_emb, patient_labels)
    stage_sil = silhouette_score(all_emb, stage_labels)
    
    print(f"Patient silhouette: {pat_sil:.4f}")
    print(f"Stage silhouette:   {stage_sil:.4f}")
    print(f"Difference (stage - patient): {stage_sil - pat_sil:.4f}")
    
    if stage_sil > pat_sil:
        print(f"→ STAGE DOMINATES: embeddings cluster better by sleep stage")
    elif pat_sil > stage_sil:
        print(f"→ PATIENT DOMINATES: embeddings cluster better by patient identity")
    else:
        print(f"→ SIMILAR: both signals equally weak")
    
    # ===== WITHIN-STAGE PATIENT SEPARATION =====
    print("\n" + "="*60)
    print("PATIENT SEPARATION WITHIN EACH SLEEP STAGE")
    print("="*60)
    
    for stage_idx in sorted(np.unique(stage_labels)):
        if stage_idx == -1:  # Skip "L" (lights) label
            continue
        
        stage_name = compute_stage_names(stage_idx)
        stage_mask = stage_labels == stage_idx
        stage_emb = all_emb[stage_mask]
        stage_pat = patient_labels[stage_mask]
        
        n_epochs = stage_mask.sum()
        n_patients = len(np.unique(stage_pat))
        
        if n_patients < 2:
            print(f"{stage_name}: {n_epochs} epochs, {n_patients} patient - skipped")
            continue
        
        sil = silhouette_score(stage_emb, stage_pat)
        db = davies_bouldin_score(stage_emb, stage_pat)
        
        print(f"{stage_name}: silhouette={sil:>7.4f}, davies_bouldin={db:>6.2f}, "
              f"epochs={n_epochs:>5}, patients={n_patients}")
    
    # ===== DISTANCE DISTRIBUTION =====
    print("\n" + "="*60)
    print("EMBEDDING DISTANCE DISTRIBUTION")
    print("="*60)
    
    distances = pdist(all_emb)
    print(f"All pairwise distances:")
    print(f"  Mean:   {np.mean(distances):.4f}")
    print(f"  Std:    {np.std(distances):.4f}")
    print(f"  Min:    {np.min(distances):.4f}")
    print(f"  Max:    {np.max(distances):.4f}")
    print(f"  Median: {np.median(distances):.4f}")
    
    # ===== INTRA vs INTER DISTANCES =====
    print("\n" + "="*60)
    print("INTRA-PATIENT vs INTER-PATIENT DISTANCES")
    print("="*60)
    
    intra_dists = []
    inter_dists = []
    
    for pat_idx in np.unique(patient_labels):
        pat_mask = patient_labels == pat_idx
        pat_emb = all_emb[pat_mask]
        
        if len(pat_emb) > 1:
            # Intra-patient distances
            pat_dists = pdist(pat_emb)
            intra_dists.extend(pat_dists)
        
        # Inter-patient distances (vs other patients)
        other_mask = patient_labels != pat_idx
        if other_mask.sum() > 0:
            for other_pat in np.unique(patient_labels[other_mask]):
                other_emb = all_emb[patient_labels == other_pat]
                from scipy.spatial.distance import cdist
                inter_d = cdist(pat_emb, other_emb).flatten()
                inter_dists.extend(inter_d)
    
    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)
    
    print(f"Intra-patient distances:  mean={np.mean(intra_dists):.4f}, std={np.std(intra_dists):.4f}")
    print(f"Inter-patient distances:  mean={np.mean(inter_dists):.4f}, std={np.std(inter_dists):.4f}")
    print(f"Separation ratio (inter/intra): {np.mean(inter_dists) / np.mean(intra_dists):.3f}")
    
    if np.mean(inter_dists) > np.mean(intra_dists):
        print(f"→ GOOD: patients separated (inter > intra)")
    else:
        print(f"→ BAD: patients overlapping (inter ≤ intra)")
    
    # ===== SUMMARY =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if stage_sil > pat_sil and stage_sil > 0.3:
        print("✗ EMBEDDINGS CLUSTER BY SLEEP STAGE (not patient identity)")
        print("  → Contrastive loss failed to prioritize patient separation")
        print("  → Next: add anti-stage regularization or switch to multi-epoch windows")
    elif pat_sil > 0.3:
        print("✓ EMBEDDINGS CLUSTER BY PATIENT (good fingerprinting)")
        print("  → Contrastive loss working as intended")
    else:
        print("✗ EMBEDDINGS DON'T CLUSTER BY EITHER (representations weak overall)")
        print("  → Model may need more training or different architecture")


if __name__ == '__main__':
    main()