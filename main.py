"""
Script Utama: Pelatihan, Evaluasi, dan Pengujian Seluruh Model Pengenalan Pola
"""
import sys
import warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import pandas as pd
from tabulate import tabulate
from src.config import init_output_dirs, OUTPUTS_DIR, PREDICTIONS_DIR, FIGURES_DIR, ID_COL
from src.data_loader import get_train_val_data, get_test_features
from src.evaluation import plot_model_comparison
from src.models import get_all_models, MODEL_DISPLAY_NAMES

def main():
    print("=" * 80)
    print("      TUGAS PENGENALAN POLA: MULTI-MODEL CLASSIFICATION & BENCHMARK")
    print("                   Dataset: Mobile Price Classification")
    print("=" * 80)

    # 1. Inisialisasi folder output
    init_output_dirs()
    print("[1/5] Memuat dan membagi dataset (Train: 80%, Val: 20% Stratified)...")
    X_train, X_val, y_train, y_val, feature_names = get_train_val_data()
    test_df, X_test, test_ids = get_test_features()
    
    print(f"      - Fitur terdeteksi: {len(feature_names)} fitur")
    print(f"      - Data Latih (Train): {X_train.shape[0]} baris")
    print(f"      - Data Validasi (Val): {X_val.shape[0]} baris")
    print(f"      - Data Uji (test.csv): {X_test.shape[0]} baris")

    # 2. Inisialisasi model
    models = get_all_models()
    print(f"\n[2/5] Menyiapkan {len(models)} model algoritma:")
    for key, disp_name in MODEL_DISPLAY_NAMES.items():
        print(f"      * [{key}] {disp_name}")


    # 3. Pelatihan & Evaluasi tiap model
    print("\n[3/5] Memulai proses Pelatihan (Training) dan Evaluasi (Validation)...")
    results_list = []
    all_predictions_df = pd.DataFrame({ID_COL: test_ids})

    for key, model in models.items():
        disp_name = MODEL_DISPLAY_NAMES[key]
        print("-" * 80)
        print(f"--> Melatih Model: {disp_name}...")
        
        # Fit model
        model.fit(X_train, y_train)

        # Evaluasi
        metrics = model.evaluate(X_val, y_val, verbose=False, save_cm=True)
        metrics["Model Key"] = key
        metrics["Model"] = disp_name
        results_list.append(metrics)

        # Prediksi test.csv
        test_pred_df = model.predict_test(X_test, test_ids)
        pred_filename = f"test_pred_{key}.csv"
        test_pred_df.to_csv(PREDICTIONS_DIR / pred_filename, index=False)
        all_predictions_df[f"pred_{key}"] = test_pred_df["predicted_price_range"]

        print(f"    [OK] Akurasi: {metrics['accuracy']:.2%} | F1-Score: {metrics['f1_macro']:.2%} | Waktu: {metrics['train_time']:.4f}s")
        print(f"    [OK] Prediksi test.csv tersimpan di: outputs/predictions/{pred_filename}")

    # Simpan file gabungan seluruh prediksi
    all_pred_path = PREDICTIONS_DIR / "all_models_test_predictions.csv"
    all_predictions_df.to_csv(all_pred_path, index=False)
    print(f"\n[4/5] Hasil gabungan prediksi test.csv tersimpan di: {all_pred_path}")

    # 4. Ringkasan & Komparasi Performa
    print("\n[5/5] Membangun Ringkasan Evaluasi & Grafik Perbandingan...")
    summary_data = []
    for r in results_list:
        summary_data.append({
            "Model": r["Model"],
            "Accuracy": r["accuracy"],
            "Precision (Macro)": r["precision_macro"],
            "Recall (Macro)": r["recall_macro"],
            "F1-Score (Macro)": r["f1_macro"],
            "Train Time (s)": round(r["train_time"], 4)
        })

    summary_df = pd.DataFrame(summary_data)
    # Urutkan berdasarkan akurasi tertinggi
    summary_df = summary_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    # Simpan ke CSV
    csv_summary_path = OUTPUTS_DIR / "model_comparison.csv"
    summary_df.to_csv(csv_summary_path, index=False)

    # Buat grafik perbandingan
    chart_path = plot_model_comparison(summary_df)

    # Cetak tabel berformat rapi
    formatted_rows = []
    for idx, row in summary_df.iterrows():
        formatted_rows.append([
            idx + 1,
            row["Model"],
            f"{row['Accuracy']:.2%}",
            f"{row['Precision (Macro)']:.2%}",
            f"{row['Recall (Macro)']:.2%}",
            f"{row['F1-Score (Macro)']:.2%}",
            f"{row['Train Time (s)']:.4f}"
        ])

    headers = ["Rank", "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Time (s)"]
    print("\n" + "=" * 80)
    print("                     TABEL PERBANDINGAN PERFORMA MODEL")
    print("=" * 80)
    print(tabulate(formatted_rows, headers=headers, tablefmt="grid"))


    print("\n" + "=" * 80)
    print("                             RINGKASAN SELESAI")
    print("=" * 80)
    print(f"1. Tabel perbandingan disimpan di      : {csv_summary_path}")
    print(f"2. Grafik perbandingan disimpan di     : {chart_path}")
    print(f"3. Confusion Matrix per model ada di   : {FIGURES_DIR}")
    print(f"4. File prediksi data test ada di      : {PREDICTIONS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
