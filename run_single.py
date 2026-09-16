"""
CLI Runner untuk Melatih dan Menguji Satu Metode Spesifik Pengenalan Pola
"""
import sys
import warnings
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from src.config import init_output_dirs, PREDICTIONS_DIR, FIGURES_DIR
from src.data_loader import get_train_val_data, get_test_features
from src.models import get_model, MODEL_REGISTRY, MODEL_DISPLAY_NAMES

def parse_args():
    parser = argparse.ArgumentParser(
        description="Jalankan satu algoritma klasifikasi Pengenalan Pola secara independen."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Nama metode: {', '.join(MODEL_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=9,
        help="Jumlah tetangga untuk KNN (default: 9)"
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Parameter regularisasi C untuk SVM (default: 1.0)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="linear",
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="Jenis kernel untuk SVM (default: linear)"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Gunakan StandardScaler untuk fitur (opsional untuk KNN/SVM)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Kedalaman pohon maksimal untuk Decision Tree / Boosting"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Jumlah estimator untuk Gradient Boosting / AdaBoost (default: 100)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    init_output_dirs()

    # Siapkan argumen kustom sesuai model
    kwargs = {}
    if args.model == "knn":
        kwargs["n_neighbors"] = args.k
        kwargs["scale_features"] = args.scale
    elif args.model == "svm":
        kwargs["C"] = args.c
        kwargs["kernel"] = args.kernel
        kwargs["scale_features"] = args.scale
    elif args.model == "decision_tree" and args.max_depth is not None:
        kwargs["max_depth"] = args.max_depth
    elif args.model in ["gradient_boosting", "adaboost"]:
        kwargs["n_estimators"] = args.n_estimators
        if args.max_depth is not None:
            kwargs["max_depth"] = args.max_depth

    model = get_model(args.model, **kwargs)
    disp_name = MODEL_DISPLAY_NAMES[args.model]

    print("=" * 80)
    print(f"       EKSEKUSI TUNGGAL METODE: {disp_name.upper()}")
    print("=" * 80)

    # Memuat data
    print("[1/4] Memuat data latihan (train) dan validasi...")
    X_train, X_val, y_train, y_val, feature_names = get_train_val_data()
    test_df, X_test, test_ids = get_test_features()

    print(f"      - Data Latih   : {len(X_train)} sampel")
    print(f"      - Data Validasi: {len(X_val)} sampel")
    print(f"      - Data Uji Test: {len(X_test)} sampel")

    # Training
    print(f"\n[2/4] Melatih model {disp_name}...")
    model.fit(X_train, y_train)
    print(f"      Pelatihan selesai dalam {model.train_time:.4f} detik.")

    # Evaluasi
    print("\n[3/4] Evaluasi Performa pada Data Validasi...")
    metrics = model.evaluate(X_val, y_val, verbose=True, save_cm=True)

    # Prediksi test.csv
    print("\n[4/4] Memprediksi kelas label pada data test.csv...")
    out_csv = model.save_test_predictions(X_test, test_ids)
    print(f"      Hasil prediksi test.csv berhasil disimpan ke: {out_csv}")

    # Tampilkan preview 5 baris pertama prediksi
    sample_preds = model.predict_test(X_test, test_ids).head(5)
    print("\nPreview 5 Prediksi Teratas:")
    print(sample_preds.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"Ringkasan: Akurasi = {metrics['accuracy']:.2%} | F1-Score = {metrics['f1_macro']:.2%}")
    print("=" * 80)

if __name__ == "__main__":
    main()
