from pathlib import Path
import torch
import random
import numpy as np

# ==================== HYPERPARAMETERS ====================

SEED = 42
DURATION = 30  # seconds

# GNN parameters
EMBEDDING_DIM = 64
N_GNN_LAYERS = 3
LEARNING_RATE = 0.001

# EEG parameters
# EEG unit is Volts in MNE
SAMPLING_RATE = 1000  # Hz
EPOCH_LENGTH = 30  # seconds
N_AUX_CHANNELS = 10  # EOG, EMG, etc.

# Frequency bands (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 50.0)
}
BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Sleep stage labels
SLEEP_STAGES = {
    'W': 0,   # Wake
    'N1': 1,  # NREM Stage 1
    'N2': 2,  # NREM Stage 2
    'N3': 3,  # NREM Stage 3 (Slow Wave Sleep)
    'R': 4,   # REM
    'L': -1,  # Lights off / Unknown (exclude from analysis)
}
STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'R']  # Ordered by numeric label (excludes L)
STAGE_NAMES_MAP = {i: name for i, name in enumerate(STAGE_NAMES)}  # {0: 'W', 1: 'N1', ...}

def set_seed(seed=SEED):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== PATHS & DIRECTORIES ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories - scripts should create subdirs as needed
DATA_DIR = PROJECT_ROOT / 'data'
FIGURES_DIR = PROJECT_ROOT / 'figures'
LOGS_DIR = PROJECT_ROOT / 'logs'
MODELS_DIR = PROJECT_ROOT / 'models'

# Raw data location
RAW_DATA_DIR = DATA_DIR / 'osfstorage-archive'

# Create main directories if they don't exist
for directory in (DATA_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

if not RAW_DATA_DIR.exists():
    raise FileNotFoundError(f"No data! Expected data at: {RAW_DATA_DIR}")

# ==================== EEG MONTAGE ====================
N_CHANNELS = 83

def load_canonical_channels():
    channels = []
    positions = []
    
    POS_FILE = RAW_DATA_DIR / 'Co-registered average positions.pos'

    with open(POS_FILE, 'r') as f:
        N_CHANNELS = int(f.readline().strip()) # Redundant but just to be sure
        for line in f:
            parts = line.strip().split()
            ch_name = parts[1]
            coords = [float(parts[2]), float(parts[3]), float(parts[4])]
            channels.append(ch_name.upper())
            positions.append(coords)
    
    return channels, np.array(positions)

CANONICAL_CHANNELS, CANONICAL_POSITIONS = load_canonical_channels()

# ==================== DEVICE ====================

def get_device():
    if torch.cuda.is_available():
        print("\nCUDA available, using GPU \n")
        return torch.device('cuda')
    else:
        print("\nCUDA not available, using CPU \n")
        return torch.device('cpu')

DEVICE = get_device()

# ==================== PATIENT SPLITS ====================

def load_patient_splits():
    """
    Load canonical patient train/val/test splits.

    Returns:
        dict with keys 'train', 'val', 'test' containing lists of patient IDs,
        plus 'metadata' with split info.
    """
    import json
    splits_file = DATA_DIR / 'patient_splits' / 'patient_splits.json'

    if not splits_file.exists():
        raise FileNotFoundError(
            f"Patient splits not found at {splits_file}. "
            "Run 'python scripts/preprocessing/create_patient_splits.py' first."
        )

    with open(splits_file, 'r') as f:
        return json.load(f)


def load_split_labels(split: str):
    """
    Load sleep stage labels and patient IDs for a specific split.

    Args:
        split: One of 'train', 'val', 'test'

    Returns:
        labels: np.ndarray of shape (n_epochs,) with sleep stage labels (0-4)
        patient_ids: np.ndarray of shape (n_epochs,) with patient ID for each epoch
    """
    splits_dir = DATA_DIR / 'patient_splits'

    labels_file = splits_dir / f'{split}_labels.npy'
    patient_ids_file = splits_dir / f'{split}_patient_ids.npy'

    if not labels_file.exists():
        raise FileNotFoundError(
            f"Split labels not found at {labels_file}. "
            "Run 'python scripts/preprocessing/create_patient_splits.py' first."
        )

    labels = np.load(labels_file)
    patient_ids = np.load(patient_ids_file, allow_pickle=True)

    return labels, patient_ids