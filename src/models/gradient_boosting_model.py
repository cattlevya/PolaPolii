"""
Implementasi Model Gradient Boosting untuk Pengenalan Pola
"""
from sklearn.ensemble import GradientBoostingClassifier
from src.models.base_model import BasePatternModel
from src.config import RANDOM_STATE

class GradientBoostingPatternModel(BasePatternModel):
    """
    Model Gradient Boosting Classifier.
    
    Catatan Pengenalan Pola:
    Gradient Boosting adalah teknik ensemble boosting yang melatih estimator secara sekuensial.
    Setiap pohon baru dilatih untuk memprediksi residual (kesalahan) dari kombinasi pohon-pohon
    sebelumnya menggunakan pendekatan optimasi gradien (gradient descent) pada fungsi kerugian (loss function).
    Metode ini sangat andal menangani hubungan non-linear yang kompleks dan fitur berinteraksi tinggi.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, random_state: int = RANDOM_STATE, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        super().__init__(
            name="Gradient Boosting",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def _build_model(self):
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
