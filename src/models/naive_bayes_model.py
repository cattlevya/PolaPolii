"""
Implementasi Model Naive Bayes (Gaussian NB) untuk Pengenalan Pola
"""
from sklearn.naive_bayes import GaussianNB
from src.models.base_model import BasePatternModel

class NaiveBayesPatternModel(BasePatternModel):
    """
    Model Naive Bayes (Gaussian Naive Bayes).
    
    Catatan Pengenalan Pola:
    Naive Bayes menerapkan Teorema Bayes dengan asumsi independensi bersyarat 
    (conditional independence) antar fitur terhadap label kelas:
        P(C | X) ∝ P(C) * ∏ P(x_i | C)
    Varian GaussianNB mengasumsikan bahwa fitur-fitur kontinu terdistribusi normal (Gaussian)
    pada setiap kelas target. Sangat efisien dalam waktu komputasi pelatihan dan inferensi.
    """

    def __init__(self, var_smoothing: float = 1e-9, **kwargs):
        self.var_smoothing = var_smoothing
        super().__init__(
            name="Naive Bayes",
            var_smoothing=var_smoothing,
            **kwargs
        )

    def _build_model(self):
        return GaussianNB(var_smoothing=self.var_smoothing)
