"""
Diagnostic script to verify temporal alignment between EDF recordings and sleep stage labels.
Checks duration, epoch counts, and label coverage for each patient.
"""

import mne
import numpy as np
import pandas as pd

# Import from config
import sys
# Add src to path to import config
sys.path.append('src')
from config import RAW_DATA_DIR, EPOCH_LENGTH, SAMPLING_RATE

mne.set_log_level('WARNING')

def parse_label_file(txt_path):
    """Parse sleep stage label file and return stage labels with timing."""
    stages = []
    start_times = []
    durations = []
    
    with open(txt_path, 'r') as f:
        content = f.read().strip()
        # Split by whitespace and process in groups of 3
        tokens = content.split()
        for i in range(0, len(tokens), 3):
            if i+2 < len(tokens):
                stage = tokens[i]
                start = int(tokens[i+1])
                duration = int(tokens[i+2])
                stages.append(stage)
                start_times.append(start)
                durations.append(duration)
    
    return stages, start_times, durations

def check_patient_alignment(patient_dir):
    """Check temporal alignment for a single patient."""
    patient_id = patient_dir.name
    
    # Find EDF file (naming varies)
    edf_files = list(patient_dir.glob("*.edf"))
    if not edf_files:
        return None
    edf_path = edf_files[0]
    
    # Find label file (should be .txt)
    txt_files = list(patient_dir.glob("*.txt"))
    if not txt_files:
        return None
    txt_path = txt_files[0]
    
    # Load EDF to get recording duration
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    edf_duration_sec = raw.n_times / raw.info['sfreq']
    edf_duration_min = edf_duration_sec / 60
    edf_duration_hr = edf_duration_min / 60
    
    # Parse label file
    stages, start_times, durations = parse_label_file(txt_path)
    
    # Calculate label coverage
    label_start_sec = start_times[0]
    label_end_sec = start_times[-1] + durations[-1]
    label_duration_sec = label_end_sec - label_start_sec
    label_duration_min = label_duration_sec / 60
    label_duration_hr = label_duration_min / 60
    
    # Count epochs
    num_epochs_labels = len(stages)
    num_epochs_edf = int(edf_duration_sec / EPOCH_LENGTH)
    
    # Check if all epochs are 30 seconds
    all_30sec = all(d == 30 for d in durations)
    
    # Check for gaps
    expected_starts = [start_times[0] + i*30 for i in range(len(stages))]
    has_gaps = start_times != expected_starts
    
    return {
        'patient_id': patient_id,
        'edf_duration_hr': edf_duration_hr,
        'edf_duration_min': edf_duration_min,
        'label_duration_hr': label_duration_hr,
        'label_duration_min': label_duration_min,
        'label_start_sec': label_start_sec,
        'label_end_sec': label_end_sec,
        'num_epochs_edf': num_epochs_edf,
        'num_epochs_labels': num_epochs_labels,
        'all_30sec_epochs': all_30sec,
        'has_gaps': has_gaps,
        'edf_file': edf_path.name,
        'txt_file': txt_path.name,
        'stage_counts': pd.Series(stages).value_counts().to_dict()
    }

def main():
    # Find all patient directories
    patient_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() 
                          if d.is_dir() and d.name.startswith('EPCTL')])
    
    print(f"Found {len(patient_dirs)} patient directories\n")
    print("="*100)
    
    results = []
    for patient_dir in patient_dirs:
        result = check_patient_alignment(patient_dir)
        if result:
            results.append(result)
    
    # Print detailed results
    for r in results:
        print(f"\n{r['patient_id']}")
        print(f"  EDF file: {r['edf_file']}")
        print(f"  Label file: {r['txt_file']}")
        print(f"  EDF duration: {r['edf_duration_hr']:.2f} hours ({r['edf_duration_min']:.1f} min)")
        print(f"  Label coverage: {r['label_duration_hr']:.2f} hours ({r['label_duration_min']:.1f} min)")
        print(f"  Label start: {r['label_start_sec']}s, end: {r['label_end_sec']}s")
        print(f"  Epochs in EDF (30s): {r['num_epochs_edf']}")
        print(f"  Epochs in labels: {r['num_epochs_labels']}")
        print(f"  All epochs 30sec: {r['all_30sec_epochs']}")
        print(f"  Has gaps in labels: {r['has_gaps']}")
        print(f"  Stage distribution: {r['stage_counts']}")
        
        # Check for misalignment
        if abs(r['num_epochs_edf'] - r['num_epochs_labels']) > 1:
            print(f"  ⚠️  WARNING: Epoch count mismatch!")
        if r['label_start_sec'] > 60:
            print(f"  ⚠️  WARNING: Labels don't start at beginning of recording!")
        if r['has_gaps']:
            print(f"  ⚠️  WARNING: Gaps detected in label sequence!")
    
    # Summary statistics
    print("\n" + "="*100)
    print("\nSUMMARY STATISTICS")
    print("="*100)
    
    edf_durations = [r['edf_duration_hr'] for r in results]
    label_durations = [r['label_duration_hr'] for r in results]
    epoch_counts = [r['num_epochs_labels'] for r in results]
    
    print(f"\nEDF durations (hours):")
    print(f"  Mean: {np.mean(edf_durations):.2f} ± {np.std(edf_durations):.2f}")
    print(f"  Range: [{np.min(edf_durations):.2f}, {np.max(edf_durations):.2f}]")
    
    print(f"\nLabel durations (hours):")
    print(f"  Mean: {np.mean(label_durations):.2f} ± {np.std(label_durations):.2f}")
    print(f"  Range: [{np.min(label_durations):.2f}, {np.max(label_durations):.2f}]")
    
    print(f"\nNumber of labeled epochs:")
    print(f"  Mean: {np.mean(epoch_counts):.0f} ± {np.std(epoch_counts):.0f}")
    print(f"  Range: [{np.min(epoch_counts)}, {np.max(epoch_counts)}]")
    
    # Check for issues
    issues = []
    for r in results:
        if abs(r['num_epochs_edf'] - r['num_epochs_labels']) > 1:
            issues.append(f"{r['patient_id']}: Epoch mismatch")
        if r['label_start_sec'] > 60:
            issues.append(f"{r['patient_id']}: Late label start")
        if r['has_gaps']:
            issues.append(f"{r['patient_id']}: Label gaps")
    
    if issues:
        print("\n\nISSUES DETECTED:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("\n\n✅ No major alignment issues detected!")

if __name__ == "__main__":
    main()