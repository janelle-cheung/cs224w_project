"""
Create clean sleep stage label files for each patient.
Output: data/sleep_stage_labels/{patient_id}.npy
Each file contains array of shape (n_epochs, 2) with columns [epoch_idx, sleep_stage]
"""

import numpy as np
import sys
from pathlib import Path

sys.path.append('src')

from config import RAW_DATA_DIR, DATA_DIR, SLEEP_STAGES

OUTPUT_DIR = DATA_DIR / 'sleep_stage_labels'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_txt_file(txt_file: Path) -> np.ndarray:
    """Parse sleep stage txt file and return array of (epoch_idx, stage_label)."""
    labels = []
    with open(txt_file, 'r') as f:
        for epoch_idx, line in enumerate(f):
            parts = line.strip().split('\t')
            stage_str = parts[0]
            stage_label = SLEEP_STAGES[stage_str]
            labels.append([epoch_idx, stage_label])

    return np.array(labels, dtype=np.int16)

def main():
    txt_files = sorted(RAW_DATA_DIR.glob("EPCTL*/*.txt"))

    print(f"Processing {len(txt_files)} patients...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for txt_file in txt_files:
        patient_id = txt_file.parent.name
        try:
            labels = process_txt_file(txt_file)
            output_file = OUTPUT_DIR / f'{patient_id}.npy'
            np.save(output_file, labels)

            # Summary stats
            n_epochs = len(labels)
            valid_mask = labels[:, 1] >= 0  # Exclude L (-1)
            n_valid = valid_mask.sum()
            print(f"  {patient_id}: {n_epochs} epochs ({n_valid} valid, {n_epochs - n_valid} excluded)")

        except Exception as e:
            print(f"  {patient_id}: ERROR - {e}")

    print(f"\nDone! Labels saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
