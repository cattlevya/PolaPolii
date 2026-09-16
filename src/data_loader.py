"""
Modul DataLoader untuk memuat dan membagi dataset Mobile Price Classification
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TRAIN_FILE, TEST_FILE, TARGET_COL, ID_COL, RANDOM_STATE, TEST_SPLIT_SIZE

def load_train_data() -> pd.DataFrame:
    """Memuat data training dari file train.csv."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"File {TRAIN_FILE} tidak ditemukan!")
    return pd.read_csv(TRAIN_FILE)

def load_test_data() -> pd.DataFrame:
    """Memuat data testing dari file test.csv."""
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"File {TEST_FILE} tidak ditemukan!")
    return pd.read_csv(TEST_FILE)

def get_train_val_data(test_size: float = TEST_SPLIT_SIZE, random_state: int = RANDOM_STATE):
    """
    Membagi train.csv menjadi data latih dan data validasi (Stratified Split).
    
    Returns:
        X_train (pd.DataFrame): Fitur data latih
        X_val (pd.DataFrame): Fitur data validasi
        y_train (pd.Series): Label target data latih
        y_val (pd.Series): Label target data validasi
        feature_names (list): Daftar nama kolom fitur
    """
    df = load_train_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_val, y_train, y_val, feature_names

def get_full_train_data():
    """
    Mendapatkan seluruh data train.csv untuk melatih model sebelum memprediksi test.csv.
    
    Returns:
        X (pd.DataFrame): Seluruh fitur dari train.csv
        y (pd.Series): Seluruh target dari train.csv
        feature_names (list): Daftar nama kolom fitur
    """
    df = load_train_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y, list(X.columns)

def get_test_features():
    """
    Mendapatkan data test.csv siap inferensi.
    
    Returns:
        test_df (pd.DataFrame): DataFrame asli test.csv
        X_test (pd.DataFrame): Fitur untuk prediksi (tanpa kolom 'id')
        ids (pd.Series): Kolom ID pengujian
    """
    test_df = load_test_data()
    if ID_COL in test_df.columns:
        ids = test_df[ID_COL]
        X_test = test_df.drop(columns=[ID_COL])
    else:
        ids = pd.Series(range(1, len(test_df) + 1), name="id")
        X_test = test_df.copy()
    return test_df, X_test, ids
