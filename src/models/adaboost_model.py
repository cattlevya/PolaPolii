"""
Implementasi Model AdaBoost (Adaptive Boosting) untuk Pengenalan Pola
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from src.models.base_model import BasePatternModel
from src.config import RANDOM_STATE

class AdaBoostPatternModel(BasePatternModel):
    """
    Model AdaBoost (Adaptive Boosting) Classifier.
    
    Catatan Pengenalan Pola:
    AdaBoost bekerja dengan melatih rangkaian base learner (biasanya decision stump / shallow tree).
    Pada setiap iterasi, data sampel yang salah diklasifikasikan akan diberikan bobot (weight) lebih besar,
    sehingga model berikutnya memusatkan perhatian pada sampel-sampel yang sulit dipelajari tersebut.
    Hasil akhir ditentukan melalui pemungutan suara berbobot (weighted voting) dari seluruh learner.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.5, max_depth: int = 3, random_state: int = RANDOM_STATE, **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        super().__init__(
            name="AdaBoost",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

    def _build_model(self):
        base_estimator = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        try:
            # scikit-learn >= 1.2 menggunakan parameter 'estimator'
            return AdaBoostClassifier(
                estimator=base_estimator,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                algorithm="SAMME",
                random_state=self.random_state
            )
        except TypeError:
            # Kompatibilitas mundur jika versi scikit-learn lama
            return AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                algorithm="SAMME",
                random_state=self.random_state
            )

