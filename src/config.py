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

# Directories
DATA_DIR = PROJECT_ROOT / 'data'
FIGURES_DIR = PROJECT_ROOT / 'figures'
RESULTS_DIR = FIGURES_DIR / 'results'
LOGS_DIR = PROJECT_ROOT / 'logs'
MODELS_DIR = PROJECT_ROOT / 'models'
DIRECTORIES = (DATA_DIR, FIGURES_DIR, RESULTS_DIR, LOGS_DIR, MODELS_DIR)

# Create directories if they don't exist
for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

RAW_DATA_DIR = DATA_DIR / 'osfstorage-archive'

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