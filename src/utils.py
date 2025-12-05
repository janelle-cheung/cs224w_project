from config import *
import h5py
import mne

mne.set_log_level('WARNING')
edf_files = sorted(DATA_DIR.glob("EPCTL*/*.edf"))