# 📘 PANDUAN PENGGUNAAN & PENGUJIAN
## Tugas Pengenalan Pola: Multi-Model Mobile Price Classification

---

## DAFTAR ISI

1. [Dataset & Persiapan Awal](#1-dataset--persiapan-awal)
2. [Instalasi Dependensi](#2-instalasi-dependensi)
3. [Cara Menjalankan Program](#3-cara-menjalankan-program)
4. [Teori Setiap Metode](#4-teori-setiap-metode)
5. [Hyperparameter Default & Konfigurasi](#5-hyperparameter-default--konfigurasi)
6. [Memahami Output & Evaluasi](#6-memahami-output--evaluasi)
7. [Struktur File Output](#7-struktur-file-output)
8. [Perbandingan & Analisis Metode](#8-perbandingan--analisis-metode)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Dataset & Persiapan Awal

### Deskripsi Dataset

Dataset **Mobile Price Classification** terdiri dari atribut teknis ponsel pintar yang digunakan untuk memprediksi **rentang harga** (kelas target `price_range`).

| File | Jumlah Baris | Kolom | Keterangan |
|---|---|---|---|
| `train.csv` | 2000 | 21 (20 fitur + 1 label) | Data latih dengan label `price_range` |
| `test.csv` | 1000 | 21 (1 ID + 20 fitur) | Data uji **tanpa label** untuk prediksi |

### Fitur-fitur Dataset

| Fitur | Tipe | Deskripsi |
|---|---|---|
| `battery_power` | Numerik | Kapasitas baterai (mAh) |
| `blue` | Biner | Konektivitas Bluetooth (0/1) |
| `clock_speed` | Numerik | Kecepatan prosesor (GHz) |
| `dual_sim` | Biner | Mendukung dual SIM (0/1) |
| `fc` | Numerik | Resolusi kamera depan (MP) |
| `four_g` | Biner | Mendukung 4G (0/1) |
| `int_memory` | Numerik | Memori internal (GB) |
| `m_dep` | Numerik | Ketebalan ponsel (cm) |
| `mobile_wt` | Numerik | Berat ponsel (gram) |
| `n_cores` | Numerik | Jumlah core prosesor |
| `pc` | Numerik | Resolusi kamera utama (MP) |
| `px_height` | Numerik | Resolusi tinggi layar (px) |
| `px_width` | Numerik | Resolusi lebar layar (px) |
| `ram` | Numerik | RAM (MB) — fitur terkuat (korelasi 0.917) |
| `sc_h` | Numerik | Tinggi layar (cm) |
| `sc_w` | Numerik | Lebar layar (cm) |
| `talk_time` | Numerik | Durasi bicara (jam) |
| `three_g` | Biner | Mendukung 3G (0/1) |
| `touch_screen` | Biner | Layar sentuh (0/1) |
| `wifi` | Biner | Mendukung WiFi (0/1) |

### Label Target

| Nilai | Kelas | Keterangan |
|---|---|---|
| 0 | Low Cost | Harga rendah |
| 1 | Medium Cost | Harga menengah |
| 2 | High Cost | Harga tinggi |
| 3 | Very High Cost | Harga sangat tinggi |

> **Distribusi kelas**: Seimbang sempurna — 500 sampel per kelas (stratified split 80:20 dipertahankan).

---

## 2. Instalasi Dependensi

### Prasyarat
- Python 3.9 atau lebih baru
- pip (package installer)

### Langkah Instalasi

```bash
# Pastikan berada di direktori tugas
cd "c:\Users\LENOVO\College\Pengenalan Pola\Tugas"

# Install semua dependensi
pip install -r requirements.txt
```

### Isi requirements.txt

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tabulate>=0.9.0
```

### Verifikasi Instalasi

```bash
python -c "import sklearn, pandas, numpy, matplotlib, seaborn, tabulate; print('Semua library berhasil diimport!')"
```

---

## 3. Cara Menjalankan Program

### A. Menjalankan Semua Model Sekaligus (Rekomendasi untuk Laporan)

```bash
python main.py
```

Perintah ini akan:
1. Membagi `train.csv` menjadi 80% data latih dan 20% data validasi secara stratified
2. Melatih ke-6 model secara berurutan
3. Mengevaluasi setiap model pada data validasi
4. Mencetak tabel perbandingan performa di terminal
5. Menyimpan file prediksi `test.csv` untuk setiap model di `outputs/predictions/`
6. Menyimpan grafik Confusion Matrix dan grafik perbandingan di `outputs/figures/`

**Contoh output terminal:**
```
================================================================================
      TUGAS PENGENALAN POLA: MULTI-MODEL CLASSIFICATION & BENCHMARK
                   Dataset: Mobile Price Classification
================================================================================
[1/5] Memuat dan membagi dataset (Train: 80%, Val: 20% Stratified)...
...
[3/5] Memulai proses Pelatihan (Training) dan Evaluasi (Validation)...
--> Melatih Model: K-Nearest Neighbors (KNN)...
    [OK] Akurasi: 95.75% | F1-Score: 95.72% | Waktu: 0.0026s
...
```

---

### B. Menjalankan Satu Metode Spesifik

```bash
python run_single.py --model <nama_metode>
```

#### Daftar Pilihan Model

| Nama Model (--model) | Algoritma |
|---|---|
| `knn` | K-Nearest Neighbors |
| `decision_tree` | Decision Tree |
| `naive_bayes` | Naive Bayes (Gaussian NB) |
| `svm` | Support Vector Machine |
| `gradient_boosting` | Gradient Boosting |
| `adaboost` | AdaBoost |

#### Contoh Perintah

```bash
# Menjalankan KNN
python run_single.py --model knn

# Menjalankan Decision Tree dengan kedalaman pohon maksimal 8
python run_single.py --model decision_tree --max_depth 8

# Menjalankan SVM dengan kernel linear (default)
python run_single.py --model svm

# Menjalankan SVM dengan kernel RBF
python run_single.py --model svm --kernel rbf

# Menjalankan Gradient Boosting dengan 200 estimator
python run_single.py --model gradient_boosting --n_estimators 200

# Menjalankan AdaBoost dengan 150 estimator
python run_single.py --model adaboost --n_estimators 150
```

#### Argumen CLI Lengkap

| Argumen | Berlaku Untuk | Default | Keterangan |
|---|---|---|---|
| `--model` | Semua | (wajib) | Nama metode yang dipilih |
| `--k` | KNN | 9 | Jumlah tetangga |
| `--c` | SVM | 1.0 | Parameter regularisasi C |
| `--kernel` | SVM | `linear` | Jenis kernel: `linear`, `rbf`, `poly`, `sigmoid` |
| `--scale` | KNN, SVM | False | Aktifkan StandardScaler |
| `--max_depth` | Decision Tree, Boosting | None | Kedalaman pohon maksimal |
| `--n_estimators` | Gradient Boosting, AdaBoost | 100 | Jumlah pohon estimator |

---

## 4. Teori Setiap Metode

### 4.1 K-Nearest Neighbors (KNN)

**Konsep Dasar:**
KNN adalah algoritma **lazy learning** berbasis instance. Model tidak membangun fungsi eksplisit pada fase training, melainkan menyimpan seluruh data latih. Saat inferensi, KNN menghitung jarak dari sampel uji ke seluruh data latih dan mengambil **k tetangga terdekat** untuk menentukan label kelas melalui **majority voting**.

**Formulasi Jarak Euclidean:**
```
d(x, xi) = sqrt( sum( (xj - xij)^2 ) )  untuk j = 1..n_features
```

**Keputusan Klasifikasi:**
```
y_pred = argmax( count(y_i | i in k-nearest) )
```

**Keunggulan pada Dataset Ini:**
- Fitur `ram` memiliki korelasi sangat tinggi (0.917) dengan label. Tanpa scaling, KNN secara alami terdominasi oleh fitur berskala besar seperti `ram` (ratusan–ribuan MB) — yang justru informatif. Ini menjelaskan mengapa **tanpa scaling** menghasilkan akurasi 95.75%, lebih tinggi dari versi dengan scaling (~56%).

**Hyperparameter Default:**
- `n_neighbors = 9`, `weights = 'distance'` (tetangga lebih dekat = bobot lebih besar)

---

### 4.2 Decision Tree (Pohon Keputusan)

**Konsep Dasar:**
Decision Tree membangun struktur pohon biner secara rekursif. Di setiap simpul (node), algoritma memilih **fitur terbaik dan ambang batas pemisahan** yang meminimalkan impuritas (ketidakmurnian) pada subset data.

**Kriteria Impuritas Gini:**
```
Gini(S) = 1 - sum( p_k^2 )  untuk k = 1..n_class
```

**Kriteria Information Gain (Entropi):**
```
IG(S, A) = H(S) - sum( |Sv|/|S| * H(Sv) )
H(S) = - sum( p_k * log2(p_k) )
```

**Catatan Implementasi:**
- Kriteria: `gini`, kedalaman maksimal: `max_depth=10`
- Tidak memerlukan scaling fitur karena pemisahan bersifat ordinal
- Rentan overfitting jika `max_depth` terlalu besar

---

### 4.3 Naive Bayes (Gaussian NB)

**Konsep Dasar:**
Naive Bayes menerapkan **Teorema Bayes** dengan asumsi **independensi bersyarat** antar fitur (asumsi "naif"):

```
P(C | X) ∝ P(C) * product( P(xi | C) )  untuk i = 1..n_features
```

**Distribusi Gaussian:**
Diasumsikan bahwa nilai fitur numerik kontinu pada setiap kelas berdistribusi normal:
```
P(xi | C) = (1 / sqrt(2*pi*sigma^2)) * exp( -(xi - mu)^2 / (2*sigma^2) )
```
di mana `mu` dan `sigma` diestimasikan dari data latih.

**Keunggulan:**
- Pelatihan sangat cepat (komputasi hanya mean dan variansi per fitur per kelas)
- Cocok sebagai baseline

**Keterbatasan pada Dataset Ini:**
- Asumsi independensi fitur dilanggar (fitur px_height dan px_width berkorelasi)
- Asumsi distribusi Gaussian mungkin tidak berlaku untuk semua fitur biner

---

### 4.4 Support Vector Machine (SVM)

**Konsep Dasar:**
SVM mencari **hyperplane pemisah optimal** yang memaksimalkan **margin** antara hyperplane dengan support vectors (data terdepan paling dekat ke batas keputusan).

**Optimasi Primer (Linear SVM):**
```
Minimize: (1/2) ||w||^2
Subject to: yi(w^T xi + b) >= 1  untuk semua i
```

**Dengan Soft Margin (parameter C):**
```
Minimize: (1/2)||w||^2 + C * sum(xi_i)
Subject to: yi(w^T xi + b) >= 1 - xi_i, xi_i >= 0
```
- `C` besar → margin lebih ketat, berpotensi overfitting
- `C` kecil → margin lebih longgar, lebih toleran terhadap kesalahan

**Kernel Trick untuk Data Non-Linear:**
```
K(xi, xj) = phi(xi)^T * phi(xj)
```
- **Linear kernel**: K(xi, xj) = xi^T * xj (digunakan sebagai default)
- **RBF kernel**: K(xi, xj) = exp(-gamma * ||xi - xj||^2)

**Keunggulan pada Dataset Ini:**
- Linear SVM tanpa scaling menghasilkan akurasi **98.25%** — tertinggi di antara semua metode. Hal ini karena kelas-kelas `price_range` cukup terpisah secara linear di ruang fitur asli, khususnya berkat dominasi fitur `ram`.

---

### 4.5 Gradient Boosting

**Konsep Dasar:**
Gradient Boosting melatih estimator secara **sekuensial**. Setiap estimator baru dilatih untuk meminimalkan **residual (kesalahan)** dari model kombinasi sebelumnya menggunakan prinsip **gradient descent** pada fungsi kerugian (loss function).

**Formulasi Update:**
```
F_m(x) = F_{m-1}(x) + nu * h_m(x)
```
di mana:
- `F_m(x)`: prediksi ensemble iterasi ke-m
- `h_m(x)`: estimator baru (pohon keputusan kecil) yang dilatih pada negatif gradien kerugian
- `nu`: learning rate (mengontrol kontribusi tiap estimator)

**Hyperparameter Default:**
- `n_estimators=100`, `learning_rate=0.1`, `max_depth=3`

---

### 4.6 AdaBoost (Adaptive Boosting)

**Konsep Dasar:**
AdaBoost melatih rangkaian **weak learner** (biasanya decision stump atau pohon dangkal). Pada setiap iterasi:
1. Data yang **salah diklasifikasikan** mendapatkan bobot lebih besar
2. Estimator baru dilatih dengan distribusi bobot tersebut
3. Estimator diberi bobot `alpha` berdasarkan error-nya

**Bobot Estimator:**
```
alpha_t = 0.5 * ln( (1 - epsilon_t) / epsilon_t )
```
di mana `epsilon_t` = weighted error rate estimator ke-t

**Prediksi Akhir:**
```
H(x) = sign( sum( alpha_t * h_t(x) ) )  untuk t = 1..T
```

**Implementasi:**
- Menggunakan algoritma `SAMME` (multi-class AdaBoost)
- Base estimator: `DecisionTreeClassifier(max_depth=3)` — pohon yang lebih dalam dari stump (depth=1) agar lebih informatif
- `n_estimators=100`, `learning_rate=0.5`

**Hyperparameter Optimal (dari Grid Search):**
```
depth=3, lr=0.5 → acc=93.25%
depth=4, lr=0.5 → acc=95.25%  (opsional untuk eksplorasi lebih lanjut)
```

---

## 5. Hyperparameter Default & Konfigurasi

| Metode | Hyperparameter | Default | Keterangan |
|---|---|---|---|
| KNN | `n_neighbors` | 9 | Jumlah tetangga (optimal dari eksperimen) |
| KNN | `weights` | `distance` | Bobot berbanding terbalik dengan jarak |
| KNN | `metric` | `euclidean` | Jarak Euclidean |
| Decision Tree | `criterion` | `gini` | Gini impurity |
| Decision Tree | `max_depth` | 10 | Kedalaman maksimal pohon |
| Decision Tree | `min_samples_split` | 5 | Minimum sampel untuk pemisahan |
| Naive Bayes | `var_smoothing` | 1e-9 | Smoothing untuk stabilitas variansi |
| SVM | `C` | 1.0 | Regularisasi (optimal dari eksperimen) |
| SVM | `kernel` | `linear` | Kernel linear (optimal untuk dataset ini) |
| Gradient Boosting | `n_estimators` | 100 | Jumlah pohon |
| Gradient Boosting | `learning_rate` | 0.1 | Learning rate |
| Gradient Boosting | `max_depth` | 3 | Kedalaman pohon tiap estimator |
| AdaBoost | `n_estimators` | 100 | Jumlah estimator |
| AdaBoost | `learning_rate` | 0.5 | Learning rate |
| AdaBoost | `max_depth` | 3 | Kedalaman base estimator |
| AdaBoost | `algorithm` | `SAMME` | Multi-class boosting (scikit-learn >= 1.4) |

---

## 6. Memahami Output & Evaluasi

### 6.1 Strategi Validasi

Karena `test.csv` **tidak memiliki label** (unlabeled), evaluasi dilakukan dengan cara:

1. `train.csv` dibagi secara **Stratified Split** 80:20:
   - **80% (1600 sampel)** → Data latih (training)
   - **20% (400 sampel)** → Data validasi (validation)
2. Model dievaluasi pada **data validasi** untuk mendapatkan metrik performa
3. Model kemudian memprediksi label kelas untuk seluruh **1000 sampel** di `test.csv`

### 6.2 Metrik Evaluasi

#### Confusion Matrix
Matriks 4×4 yang menunjukkan distribusi prediksi vs label aktual:
- Diagonal utama = prediksi benar
- Off-diagonal = kesalahan prediksi

#### Accuracy
```
Accuracy = (TP + TN) / Total = Jumlah Prediksi Benar / Total Sampel
```

#### Precision (Macro)
```
Precision per kelas = TP / (TP + FP)
Macro Precision = rata-rata semua kelas (tanpa pembobotan)
```

#### Recall (Macro)
```
Recall per kelas = TP / (TP + FN) 
Macro Recall = rata-rata semua kelas
```

#### F1-Score (Macro)
```
F1 per kelas = 2 * (Precision * Recall) / (Precision + Recall)
Macro F1 = rata-rata semua kelas
```

> **Catatan:** Dataset ini memiliki distribusi kelas seimbang sempurna (500 sampel/kelas), sehingga macro dan weighted F1 bernilai hampir sama.

### 6.3 Classification Report

Output verbose dari `run_single.py` menampilkan classification report scikit-learn:

```
==================== EVALUATION REPORT: K-Nearest Neighbors ====================
               precision    recall  f1-score   support

      0 (Low)     0.9804    1.0000    0.9901       100
   1 (Medium)     0.9495    0.9400    0.9447       100
     2 (High)     0.9381    0.9100    0.9239       100
3 (Very High)     0.9608    0.9800    0.9703       100

     accuracy                         0.9575       400
    macro avg     0.9572    0.9575    0.9572       400
 weighted avg     0.9572    0.9575    0.9572       400
```

- **support**: Jumlah sampel validasi per kelas (100 tiap kelas)

---

## 7. Struktur File Output

Setelah menjalankan `main.py`, folder output akan terisi sebagai berikut:

```
outputs/
├── figures/
│   ├── confusion_matrix_k_nearest_neighbors.png    # Confusion Matrix KNN
│   ├── confusion_matrix_decision_tree.png          # Confusion Matrix Decision Tree
│   ├── confusion_matrix_naive_bayes.png            # Confusion Matrix Naive Bayes
│   ├── confusion_matrix_support_vector_machine.png # Confusion Matrix SVM
│   ├── confusion_matrix_gradient_boosting.png      # Confusion Matrix Gradient Boosting
│   ├── confusion_matrix_adaboost.png               # Confusion Matrix AdaBoost
│   └── model_comparison.png                        # Grafik perbandingan akurasi & F1
│
├── predictions/
│   ├── test_pred_knn.csv               # Prediksi test.csv oleh KNN
│   ├── test_pred_decision_tree.csv     # Prediksi test.csv oleh Decision Tree
│   ├── test_pred_naive_bayes.csv       # Prediksi test.csv oleh Naive Bayes
│   ├── test_pred_svm.csv               # Prediksi test.csv oleh SVM
│   ├── test_pred_gradient_boosting.csv # Prediksi test.csv oleh Gradient Boosting
│   ├── test_pred_adaboost.csv          # Prediksi test.csv oleh AdaBoost
│   └── all_models_test_predictions.csv # Prediksi gabungan semua model
│
└── model_comparison.csv                # Tabel performa seluruh model
```

### Format File Prediksi

Setiap file `test_pred_*.csv` berformat:

```csv
id,predicted_price_range
1,3
2,3
3,2
4,3
5,1
...
```

- `id`: ID sampel dari `test.csv`
- `predicted_price_range`: Prediksi kelas (0, 1, 2, atau 3)

### File Prediksi Gabungan (all_models_test_predictions.csv)

```csv
id,pred_knn,pred_decision_tree,pred_naive_bayes,pred_svm,pred_gradient_boosting,pred_adaboost
1,3,3,3,3,3,3
2,3,3,3,3,3,3
...
```

---

## 8. Perbandingan & Analisis Metode

### Hasil Evaluasi (Validation Set 400 Sampel)

| Metode | Accuracy | Precision | Recall | F1-Score | Waktu Training |
|---|---|---|---|---|---|
| Support Vector Machine | **98.25%** | **98.30%** | **98.25%** | **98.25%** | ~9 detik |
| K-Nearest Neighbors | 95.75% | 95.72% | 95.75% | 95.72% | <0.01 detik |
| AdaBoost | 93.25% | 93.38% | 93.25% | 93.28% | ~0.9 detik |
| Gradient Boosting | 91.25% | 91.24% | 91.25% | 91.23% | ~4.5 detik |
| Decision Tree | 84.50% | 84.78% | 84.50% | 84.54% | ~0.02 detik |
| Naive Bayes | 81.00% | 81.13% | 81.00% | 81.05% | <0.01 detik |

### Analisis & Interpretasi

**Mengapa SVM menjadi terbaik?**  
Dataset ini memiliki separator linier yang sangat baik di ruang fitur aslinya, terutama berkat fitur `ram` yang korelasinya sangat tinggi (0.917) dengan `price_range`. SVM dengan kernel linear menemukan hyperplane pemisah optimal tanpa overfitting.

**Mengapa KNN tanpa scaling performanya sangat baik?**  
Karena fitur `ram` (bernilai ratusan–ribuan) mendominasi perhitungan jarak Euclidean. Dominasi ini justru bermanfaat karena `ram` merupakan prediktor terkuat. Dengan scaling, pengaruh `ram` disamakan dengan fitur lain yang kurang informatif, sehingga akurasi turun drastis (~56%).

**Mengapa Naive Bayes paling rendah?**  
Asumsi independensi fitur dilanggar oleh korelasi antar fitur (misalnya `px_height` dan `px_width`). Selain itu, tidak semua fitur memiliki distribusi Gaussian.

**Trade-off Akurasi vs Kecepatan:**

| Prioritas | Pilihan Metode |
|---|---|
| Akurasi tertinggi (tak peduli waktu) | SVM |
| Akurasi tinggi + cepat | KNN |
| Interpretabilitas tinggi | Decision Tree |
| Kemudahan implementasi | Naive Bayes |
| Ensemble yang seimbang | AdaBoost / Gradient Boosting |

---

## 9. Troubleshooting

### Q: Muncul warning tentang `LOKY_MAX_CPU_COUNT`
**A:** Warning ini dari joblib (dipakai sklearn untuk paralelisasi). Aman untuk diabaikan pada Windows. Sudah disupres secara otomatis dengan `warnings.filterwarnings("ignore")`.

### Q: AdaBoost memunculkan `FutureWarning: SAMME.R deprecated`
**A:** Implementasi sudah menggunakan `algorithm="SAMME"` secara eksplisit untuk kompatibilitas scikit-learn >= 1.4.

### Q: SVM sangat lambat
**A:** Dengan `probability=False` (default), SVM linear pada 1600 sampel membutuhkan sekitar 9 detik. Untuk mempercepat, nonaktifkan komputasi probabilitas (sudah default). Jika tetap lambat, gunakan kernel `linear` bukan `rbf`.

### Q: Bagaimana jika ingin menggunakan seluruh train.csv untuk prediksi?
**A:** Secara default, model hanya dilatih pada 80% dari `train.csv` untuk keperluan evaluasi internal. Jika ingin menggunakan 100% data untuk prediksi final (tanpa validation split), modifikasi `data_loader.py` dengan memanggil `get_full_train_data()` dan latih model dengannya.

### Q: Bagaimana cara menambah model baru?
**A:**
1. Buat file baru di `src/models/nama_model.py` yang mewarisi `BasePatternModel`
2. Implementasikan method `_build_model()`
3. Daftarkan ke `MODEL_REGISTRY` di `src/models/__init__.py`
4. Model akan otomatis tersedia di `main.py` dan `run_single.py`
