# Tugas Pengenalan Pola — Mobile Price Classification

Implementasi 6 algoritma klasifikasi **Machine Learning** untuk tugas mata kuliah Pengenalan Pola.  
Dataset: [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

---

## Quick Start

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan seluruh model sekaligus (benchmark)
python main.py

# 3. Jalankan satu metode spesifik
python run_single.py --model knn
python run_single.py --model decision_tree
python run_single.py --model naive_bayes
python run_single.py --model svm
python run_single.py --model gradient_boosting
python run_single.py --model adaboost
```

---

## Hasil Performa (Validation Set — Stratified 80:20)

| Rank | Metode                    | Accuracy | F1-Score |
|------|---------------------------|----------|----------|
| 1    | Support Vector Machine    | **98.25%** | **98.25%** |
| 2    | K-Nearest Neighbors       | 95.75%   | 95.72%   |
| 3    | AdaBoost                  | 93.25%   | 93.28%   |
| 4    | Gradient Boosting         | 91.25%   | 91.23%   |
| 5    | Decision Tree             | 84.50%   | 84.54%   |
| 6    | Naive Bayes (Gaussian NB) | 81.00%   | 81.05%   |

---

## Struktur Proyek

```
Tugas/
├── src/
│   ├── config.py               # Konfigurasi path & konstanta
│   ├── data_loader.py          # Utilitas pemuatan dan pembagian data
│   ├── evaluation.py           # Metrik & visualisasi evaluasi
│   └── models/
│       ├── __init__.py         # Model registry & factory
│       ├── base_model.py       # Abstract base class
│       ├── knn_model.py        # K-Nearest Neighbors
│       ├── decision_tree_model.py
│       ├── naive_bayes_model.py
│       ├── svm_model.py        # Support Vector Machine
│       ├── gradient_boosting_model.py
│       └── adaboost_model.py
├── outputs/
│   ├── predictions/            # Prediksi test.csv per model
│   └── figures/                # Confusion Matrix & grafik perbandingan
├── train.csv                   # Dataset pelatihan (2000 sampel, 20 fitur)
├── test.csv                    # Dataset pengujian (1000 sampel)
├── main.py                     # Runner semua model sekaligus
├── run_single.py               # Runner satu metode dengan argumen CLI
├── requirements.txt
├── README.md
└── PANDUAN_PENGGUNAAN.md       # Dokumentasi lengkap & panduan akademis
```

---

## Dokumentasi Lengkap

Lihat **[PANDUAN_PENGGUNAAN.md](PANDUAN_PENGGUNAAN.md)** untuk:
- Penjelasan teori & formulasi matematika setiap metode
- Panduan menjalankan dan menginterpretasikan hasil
- Analisis perbandingan metode pada dataset ini
