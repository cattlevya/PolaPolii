"""
Konfigurasi Umum Proyek Pengenalan Pola
Mobile Price Classification
"""
from pathlib import Path

# Direktori Utama
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR
OUTPUTS_DIR = BASE_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# File Data
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

# Konfigurasi Model & Data
RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.2
TARGET_COL = "price_range"
ID_COL = "id"

CLASS_NAMES = ["0 (Low)", "1 (Medium)", "2 (High)", "3 (Very High)"]

def init_output_dirs():
    """Membuat direktori output jika belum ada."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
