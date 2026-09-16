"""
Registry dan Factory untuk Model Pengenalan Pola
"""
from typing import Dict, Type
from src.models.base_model import BasePatternModel
from src.models.knn_model import KNNPatternModel
from src.models.decision_tree_model import DecisionTreePatternModel
from src.models.naive_bayes_model import NaiveBayesPatternModel
from src.models.svm_model import SVMPatternModel
from src.models.gradient_boosting_model import GradientBoostingPatternModel
from src.models.adaboost_model import AdaBoostPatternModel

MODEL_REGISTRY: Dict[str, Type[BasePatternModel]] = {
    "knn": KNNPatternModel,
    "decision_tree": DecisionTreePatternModel,
    "naive_bayes": NaiveBayesPatternModel,
    "svm": SVMPatternModel,
    "gradient_boosting": GradientBoostingPatternModel,
    "adaboost": AdaBoostPatternModel,
}

MODEL_DISPLAY_NAMES = {
    "knn": "K-Nearest Neighbors (KNN)",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes (Gaussian NB)",
    "svm": "Support Vector Machine (SVM)",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "AdaBoost",
}

def get_model(model_key: str, **kwargs) -> BasePatternModel:
    """
    Mengembalikan instance model berdasarkan kunci registrasi.
    
    Pilihan model_key:
        - 'knn'
        - 'decision_tree'
        - 'naive_bayes'
        - 'svm'
        - 'gradient_boosting'
        - 'adaboost'
    """
    normalized_key = model_key.lower().strip()
    if normalized_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_key}' tidak ditemukan. Pilihan yang tersedia: {available}")
    
    model_class = MODEL_REGISTRY[normalized_key]
    return model_class(**kwargs)

def get_all_models() -> Dict[str, BasePatternModel]:
    """
    Mengembalikan dictionary seluruh instance model yang siap digunakan.
    """
    return {key: get_model(key) for key in MODEL_REGISTRY.keys()}

__all__ = [
    "BasePatternModel",
    "KNNPatternModel",
    "DecisionTreePatternModel",
    "NaiveBayesPatternModel",
    "SVMPatternModel",
    "GradientBoostingPatternModel",
    "AdaBoostPatternModel",
    "MODEL_REGISTRY",
    "MODEL_DISPLAY_NAMES",
    "get_model",
    "get_all_models",
]
