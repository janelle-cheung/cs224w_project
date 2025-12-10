"""
Preprocess EEG data: compute spectral power per frequency band for each patient
"""
import numpy as np
import mne
import sys

# Add src to path to import config
sys.path.append('src')
from config import *

mne.set_log_level('WARNING')

# ==================== SETUP OUTPUT DIRECTORY ====================

FEATURES_DIR = DATA_DIR / 'spectral_features'
FEATURES_DIR.mkdir(exist_ok=True)
print(f"Saving features to: {FEATURES_DIR}")

# ==================== HELPER FUNCTIONS ====================

def compute_band_power(raw, band_name, fmin, fmax, epoch_length=30, chunk_size=300):
    """
    Compute average power in chunks to avoid memory issues
    
    Args:
        chunk_size: number of epochs to process at once (default 300 = 2.5 hours)
    """
    print(f"  Filtering {band_name} band ({fmin}-{fmax} Hz)...")
    
    sfreq = raw.info['sfreq']
    samples_per_epoch = int(epoch_length * sfreq)
    n_channels = len(raw.ch_names)
    n_epochs = int(raw.n_times / samples_per_epoch)
    
    # Preallocate output
    power = np.zeros((n_epochs, n_channels))
    
    # Process in chunks
    n_chunks = (n_epochs + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_epoch = chunk_idx * chunk_size
        end_epoch = min(start_epoch + chunk_size, n_epochs)
        n_epochs_chunk = end_epoch - start_epoch
        
        start_sample = start_epoch * samples_per_epoch
        end_sample = end_epoch * samples_per_epoch
        
        print(f"    Chunk {chunk_idx+1}/{n_chunks}: epochs {start_epoch}-{end_epoch}")
        
        # Load and filter chunk
        raw_chunk = raw.copy().crop(tmin=start_sample/sfreq, tmax=(end_sample-1)/sfreq)
        raw_chunk.load_data()
        raw_chunk.filter(fmin, fmax, fir_design='firwin', verbose=False)
        
        # Reshape and compute power
        data = raw_chunk.get_data()
        data_epochs = data.reshape(n_channels, n_epochs_chunk, samples_per_epoch)
        power[start_epoch:end_epoch, :] = np.var(data_epochs, axis=2).T
        
        del raw_chunk, data, data_epochs
    
    return power


def preprocess_subject(edf_path):
    """
    Process one subject: compute spectral power for all bands
    
    Args:
        edf_path: path to EDF file
        
    Returns:
        features: array of shape (n_epochs, n_channels, n_bands)
    """
    subject_id = edf_path.parent.name  # e.g., EPCTL01
    print(f"\n{'='*60}")
    print(f"Processing {subject_id}")
    print(f"{'='*60}")
    
    # Load EDF metadata first (lazy loading)
    print("Loading EDF file (lazy)...")
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    
    n_epochs = int(raw.n_times / (EPOCH_LENGTH * raw.info['sfreq']))
    print(f"  Duration: {raw.times[-1]/3600:.1f} hours ({raw.times[-1]:.1f} seconds)")
    print(f"  Num epochs: {n_epochs}")
    
    # Select only EEG channels that match canonical channels
    eeg_channels = []
    for ch in raw.ch_names:
        normalized_name = normalize_channel_name(ch)
        if normalized_name in CANONICAL_CHANNELS:
            eeg_channels.append(ch)
            
    print(f"  EEG channels: {len(eeg_channels)}/{len(CANONICAL_CHANNELS)}")
    
    # Error handling: Check if all canonical channels are present
    if len(eeg_channels) != len(CANONICAL_CHANNELS):
        missing_channels = set(CANONICAL_CHANNELS) - set(eeg_channels)
        raise Exception(
            f"ERROR: Patient {subject_id} is missing {len(missing_channels)} channels!\n"
            f"  Expected {len(CANONICAL_CHANNELS)} channels, found {len(eeg_channels)} channels\n"
            f"  Missing: {sorted(list(missing_channels))[:10]}..."  # Show first 10
        )
    
    # Pick only EEG channels (still lazy)
    raw.pick_channels(eeg_channels)
    
    # Compute power for each band
    band_powers = []
    for band_name in BAND_NAMES:
        fmin, fmax = FREQ_BANDS[band_name]
        power = compute_band_power(raw, band_name, fmin, fmax)
        band_powers.append(power)
    
    # Stack into (n_epochs, n_channels, n_bands)
    features = np.stack(band_powers, axis=2)
    
    print(f"\nFinal feature shape: {features.shape}")
    print(f"  (n_epochs={features.shape[0]}, n_channels={features.shape[1]}, n_bands={features.shape[2]})")
    
    # Validation: Check if shape matches expected dimensions
    total_duration = raw.times[-1]  # seconds
    expected_n_epochs = int(total_duration // EPOCH_LENGTH)
    expected_n_channels = len(eeg_channels)
    expected_n_bands = len(BAND_NAMES)
    
    print(f"\nValidation:")
    print(f"  Total duration: {total_duration:.1f} seconds ({total_duration/3600:.2f} hours)")
    print(f"  Expected epochs: {expected_n_epochs} (duration / {EPOCH_LENGTH}s)")
    print(f"  Actual epochs: {features.shape[0]}")
    print(f"  Expected channels: {expected_n_channels}")
    print(f"  Actual channels: {features.shape[1]}")
    print(f"  Expected bands: {expected_n_bands}")
    print(f"  Actual bands: {features.shape[2]}")
    
    # Sanity checks
    assert features.shape[0] == expected_n_epochs, f"Epoch mismatch! Expected {expected_n_epochs}, got {features.shape[0]}"
    assert features.shape[1] == expected_n_channels, f"Channel mismatch! Expected {expected_n_channels}, got {features.shape[1]}"
    assert features.shape[2] == expected_n_bands, f"Band mismatch! Expected {expected_n_bands}, got {features.shape[2]}"
    
    print(f"  ✓ All dimension checks passed!")
    
    return features


def normalize_channel_name(ch_name):
    """
    Normalize channel names by removing common suffixes ("C5", "C5-Ref", "C5-")
    and capitalizing all letters.
    
    Args:
        ch_name: channel name string
        
    Returns:
        normalized channel name (e.g., "C5-Ref" -> "C5")
    """
    ch_name = ch_name.strip().upper()
    if ch_name.endswith('-REF'):
        return ch_name[:-4]
    elif ch_name.endswith('-'):
        return ch_name[:-1]
    return ch_name

# ==================== MAIN PROCESSING LOOP ====================

def main():
    # Get all EDF files
    edf_files = sorted(RAW_DATA_DIR.glob("EPCTL*/*.edf"))
    print(f"Found {len(edf_files)} EDF files\n")
    
    if len(edf_files) == 0:
        print("ERROR: No EDF files found!")
        return
    
    # Process each subject
    for i, edf_path in enumerate(edf_files, 1):
        subject_id = edf_path.parent.name  # e.g., EPCTL01
        
        # Extract patient number (e.g., EPCTL01 -> P01)
        patient_num = subject_id.replace('EPCTL', 'P')
        output_file = FEATURES_DIR / f"{patient_num}_spectral_features.npy"
        
        # Skip if already processed
        if output_file.exists():
            print(f"\n[{i}/{len(edf_files)}] {subject_id}: Already processed, skipping...")
            continue
        
        print(f"\n[{i}/{len(edf_files)}] Processing {subject_id}...")
        
        try:
            features = preprocess_subject(edf_path)
            
            if features is not None:
                # Save features
                np.save(output_file, features)
                print(f"  ✓ Saved to {output_file.name}")
            else:
                print(f"  ✗ Failed to process {subject_id}")
                
        except Exception as e:
            print(f"  ✗ ERROR processing {subject_id}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()