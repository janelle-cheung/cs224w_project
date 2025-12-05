from pathlib import Path
import torch
import random
import numpy as np

# ==================== HYPERPARAMETERS ====================

SEED = 42
DURATION = 30  # seconds

# GNN parameters
EMBEDDING_DIM = 64
NUM_GNN_LAYERS = 3
LEARNING_RATE = 0.001

# EEG parameters
# EEG unit is Volts in MNE
SAMPLING_RATE = 1000  # Hz
EPOCH_LENGTH = 30  # seconds
NUM_AUX_CHANNELS = 10  # EOG, EMG, etc.

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
DATA_DIR = PROJECT_ROOT/'data'
FIGURES_DIR = PROJECT_ROOT/'figures'
LOGS_DIR = PROJECT_ROOT/'logs'
MODELS_DIR = PROJECT_ROOT/'models'
DIRECTORIES = (DATA_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR)

# Create directories if they don't exist
for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

if not (DATA_DIR / 'osfstorage-archive').exists():
    raise FileNotFoundError(f"No data! 'osfstorage-archive' was not found in {DATA_DIR}")

DATA_DIR = DATA_DIR / 'osfstorage-archive'

# ==================== EEG MONTAGE ====================
NUM_CHANNELS = 83

def load_canonical_channels():
    channels = []
    positions = []
    
    POS_FILE = DATA_DIR / 'Co-registered average positions.pos'

    with open(POS_FILE, 'r') as f:
        NUM_CHANNELS = int(f.readline().strip()) # Redundant but just to be sure
        for line in f:
            parts = line.strip().split()
            ch_name = parts[1]
            coords = [float(parts[2]), float(parts[3]), float(parts[4])]
            channels.append(ch_name)
            positions.append(coords)
    
    return channels, np.array(positions)

CANONICAL_CHANNELS, CANONICAL_POSITIONS = load_canonical_channels()

# ==================== DEVICE ====================

def get_device():
    if torch.cuda.is_available():
        print("\n CUDA available, using GPU \n")
        return torch.device('cuda')
    else:
        print("\n CUDA not available, using CPU \n")
        return torch.device('cpu')

DEVICE = get_device()