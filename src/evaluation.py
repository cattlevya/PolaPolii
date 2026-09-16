"""
Modul Evaluasi & Visualisasi Performa Model Pengenalan Pola
"""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend untuk penyimpanan gambar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.config import CLASS_NAMES, FIGURES_DIR

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Menghitung sekumpulan metrik evaluasi klasifikasi multikelas.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def print_detailed_report(y_true, y_pred, model_name: str):
    """
    Mencetak Classification Report scikit-learn secara lengkap ke konsol.
    """
    print(f"\n==================== EVALUATION REPORT: {model_name} ====================")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

def plot_confusion_matrix(y_true, y_pred, model_name: str, filename: str = None) -> str:
    """
    Membuat dan menyimpan visualisasi Confusion Matrix dalam format PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True
    )
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if filename is None:
        clean_name = model_name.lower().replace(" ", "_").replace("-", "_")
        filename = f"confusion_matrix_{clean_name}.png"

    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    return str(save_path)

def plot_model_comparison(comparison_df: pd.DataFrame, filename: str = "model_comparison.png") -> str:
    """
    Membuat grafik batang perbandingan performa antar model.
    """
    plt.figure(figsize=(11, 6))
    
    # Menyiapkan data untuk barplot
    df_plot = comparison_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "F1-Score (Macro)"],
        var_name="Metric",
        value_name="Score"
    )
    
    ax = sns.barplot(
        data=df_plot,
        x="Model",
        y="Score",
        hue="Metric",
        palette=["#2b5c8f", "#d95f02"]
    )
    
    plt.title("Perbandingan Performa Model Pengenalan Pola", fontsize=15, fontweight="bold", pad=15)
    plt.ylabel("Nilai Skor (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Metode / Algoritma", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Tambahkan angka skor di atas tiap batang
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2%}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300)
    plt.close()
    return str(save_path)
