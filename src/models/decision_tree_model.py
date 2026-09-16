"""
Implementasi Model Decision Tree (Pohon Keputusan) untuk Pengenalan Pola
"""
from sklearn.tree import DecisionTreeClassifier
from src.models.base_model import BasePatternModel
from src.config import RANDOM_STATE

class DecisionTreePatternModel(BasePatternModel):
    """
    Model Decision Tree Classifier.
    
    Catatan Pengenalan Pola:
    Decision Tree membagi ruang fitur secara ortogonal menggunakan aturan biner.
    Kriteria pembagian (split) dievaluasi menggunakan Gini Impurity atau Entropi (Information Gain).
    Kelebihan utama metode ini adalah kemampuan interpretasi struktur keputusan dan
    ketahanannya terhadap perbedaan skala nilai fitur.
    """

    def __init__(self, criterion: str = "gini", max_depth: int = 10, min_samples_split: int = 5, random_state: int = RANDOM_STATE, **kwargs):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        super().__init__(
            name="Decision Tree",
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            **kwargs
        )

    def _build_model(self):
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
