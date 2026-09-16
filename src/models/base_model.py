"""
Kelas Dasar (Base Class) untuk Seluruh Model Pengenalan Pola
"""
from abc import ABC, abstractmethod
import time
import pandas as pd
from pathlib import Path
from src.config import PREDICTIONS_DIR, ID_COL
from src.evaluation import calculate_metrics, print_detailed_report, plot_confusion_matrix

class BasePatternModel(ABC):
    """
    Kelas abstrak penyedia antarmuka seragam (unified interface)
    untuk seluruh algoritma klasifikasi pengenalan pola.
    """

    def __init__(self, name: str, **hyperparameters):
        self.name = name
        self.hyperparameters = hyperparameters
        self.model = self._build_model()
        self.train_time = 0.0

    @abstractmethod
    def _build_model(self):
        """Membangun objek estimator / pipeline scikit-learn."""
        pass

    def fit(self, X_train, y_train):
        """Melatih model pada data latih serta mencatat waktu komputasi."""
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
        return self

    def predict(self, X):
        """Melakukan prediksi label kelas."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Mendapatkan probabilitas prediksi tiap kelas jika didukung."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X_val, y_val, verbose: bool = True, save_cm: bool = True) -> dict:
        """
        Mengevaluasi performa model terhadap data validasi.
        
        Returns:
            dict metrik performa (Akurasi, F1, Precision, Recall, Train Time).
        """
        y_pred = self.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        metrics["train_time"] = self.train_time
        metrics["model_name"] = self.name

        if verbose:
            print_detailed_report(y_val, y_pred, self.name)
            print(f"Training Time: {self.train_time:.4f} detik")

        if save_cm:
            cm_path = plot_confusion_matrix(y_val, y_pred, self.name)
            if verbose:
                print(f"Confusion Matrix tersimpan di: {cm_path}")

        return metrics

    def predict_test(self, X_test, ids) -> pd.DataFrame:
        """
        Melakukan inferensi pada data test.csv dan mengembalikan DataFrame berformat [id, price_range].
        """
        preds = self.predict(X_test)
        result_df = pd.DataFrame({
            ID_COL: ids,
            "predicted_price_range": preds
        })
        return result_df

    def save_test_predictions(self, X_test, ids, filename: str = None) -> Path:
        """
        Menyimpan hasil prediksi data test.csv ke format file CSV.
        """
        if filename is None:
            clean_name = self.name.lower().replace(" ", "_").replace("-", "_")
            filename = f"test_pred_{clean_name}.csv"

        out_path = PREDICTIONS_DIR / filename
        pred_df = self.predict_test(X_test, ids)
        pred_df.to_csv(out_path, index=False)
        return out_path
