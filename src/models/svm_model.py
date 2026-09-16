"""
Implementasi Model Support Vector Machine (SVM) untuk Pengenalan Pola
"""
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.base_model import BasePatternModel
from src.config import RANDOM_STATE

class SVMPatternModel(BasePatternModel):
    """
    Model Support Vector Machine (SVM).
    
    Catatan Pengenalan Pola:
    SVM mencari hyperplane pemisah optimal dengan memaksimalkan margin (jarak)
    antara hyperplane dengan titik-titik data terdekat (support vectors).
    Untuk data non-linear, SVM menggunakan fungsi kernel (misalnya RBF - Radial Basis Function)
    untuk memetakan data ke ruang berdimensi lebih tinggi.
    Standarisasi fitur (StandardScaler) sangat penting agar perhitungan jarak kernel
    tidak terdistorsi oleh fitur berskala besar.
    """

    def __init__(self, C: float = 1.0, kernel: str = "linear", gamma: str = "scale", scale_features: bool = False, probability: bool = False, random_state: int = RANDOM_STATE, **kwargs):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.scale_features = scale_features
        self.probability = probability
        self.random_state = random_state
        super().__init__(
            name="Support Vector Machine",
            C=C,
            kernel=kernel,
            gamma=gamma,
            scale_features=scale_features,
            probability=probability,
            random_state=random_state,
            **kwargs
        )

    def _build_model(self):
        svc = SVC(
            C=self.C,
            kernel=self.kernel,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state
        )
        if self.scale_features:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("svm", svc)
            ])
        return svc

