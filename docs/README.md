# Dokumentasi Kode — Tugas Pengenalan Pola
## Penjelasan Baris per Baris Setiap Metode Klasifikasi

Folder ini berisi file Markdown terpisah untuk setiap algoritma yang diimplementasikan.

---

## Daftar File

| No. | File | Metode | Akurasi |
|-----|------|--------|---------|
| 1 | [01_knn.md](01_knn.md) | K-Nearest Neighbors (KNN) | 95.75% |
| 2 | [02_decision_tree.md](02_decision_tree.md) | Decision Tree | 84.50% |
| 3 | [03_naive_bayes.md](03_naive_bayes.md) | Naive Bayes (Gaussian NB) | 81.00% |
| 4 | [04_svm.md](04_svm.md) | Support Vector Machine | **98.25%** |
| 5 | [05_gradient_boosting.md](05_gradient_boosting.md) | Gradient Boosting | 91.25% |
| 6 | [06_adaboost.md](06_adaboost.md) | AdaBoost | 93.25% |

---

## Setiap File Berisi

- **Penjelasan per baris** dari kode implementasi model
- **Cara kerja algoritma** saat `fit()` dan `predict()`
- **Penjelasan hyperparameter** beserta efeknya
- **Alur eksekusi** dari pemanggilan hingga output prediksi
- **Perbandingan dan analisis** khusus untuk dataset Mobile Price Classification

---

## Cara Membaca

```
src/models/<nama>_model.py   ←  kode implementasi
docs/<no>_<nama>.md          ←  penjelasan baris per baris
```

Baca kode dan file markdown secara berdampingan untuk pemahaman penuh.
