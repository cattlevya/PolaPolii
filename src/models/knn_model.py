"""
Implementasi Model K-Nearest Neighbors (KNN) untuk Pengenalan Pola
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.models.base_model import BasePatternModel

class KNNPatternModel(BasePatternModel):
    """
    Model K-Nearest Neighbors (KNN).
    
    Catatan Pengenalan Pola:
    KNN mengklasifikasikan sampel baru berdasarkan kedekatan jarak (misal Euclidean)
    dengan k tetangga terdekat di ruang fitur. Karena fitur seperti 'ram' bernilai ribuan
    sedangkan 'm_dep' bernilai pecahan 0-1, standarisasi fitur (StandardScaler) 
    adalah prasyarat mutlak agar seluruh dimensi fitur berkontribusi secara proporsional.
    """

    def __init__(self, n_neighbors: int = 9, weights: str = "distance", metric: str = "euclidean", scale_features: bool = False, **kwargs):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.scale_features = scale_features
        super().__init__(
            name="K-Nearest Neighbors",
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            scale_features=scale_features,
            **kwargs
        )

    def _build_model(self):
        knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )
        if self.scale_features:
            return Pipeline([
                ("scaler", StandardScaler()),
                ("knn", knn)
            ])
        return knn

