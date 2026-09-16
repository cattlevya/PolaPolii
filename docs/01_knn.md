# K-Nearest Neighbors (KNN) — Penjelasan Kode Per Baris

**File:** `src/models/knn_model.py`  
**Akurasi:** 95.75% | **F1-Score:** 95.72% | **Training Time:** <0.01 detik

---

## Kode Lengkap dengan Penjelasan

```python
"""
Implementasi Model K-Nearest Neighbors (KNN) untuk Pengenalan Pola
"""
```
> **Baris 1–3:** Docstring modul — keterangan singkat isi file ini.

---

```python
from sklearn.neighbors import KNeighborsClassifier
```
> **Baris 4:** Import kelas `KNeighborsClassifier` dari scikit-learn.  
> Ini adalah implementasi algoritma KNN yang akan digunakan untuk klasifikasi.

```python
from sklearn.preprocessing import StandardScaler
```
> **Baris 5:** Import `StandardScaler` — digunakan jika fitur ingin dinormalisasi.  
> StandardScaler mengubah tiap fitur sehingga memiliki **mean = 0** dan **std = 1**.  
> Formula: `z = (x - mean) / std`

```python
from sklearn.pipeline import Pipeline
```
> **Baris 6:** Import `Pipeline` — menggabungkan beberapa langkah preprocessing + model menjadi satu objek yang dapat dipanggil `.fit()` dan `.predict()` secara otomatis berurutan.

```python
from src.models.base_model import BasePatternModel
```
> **Baris 7:** Import abstract base class `BasePatternModel`.  
> Semua model di proyek ini mewarisi kelas ini sehingga memiliki antarmuka seragam (`fit`, `predict`, `evaluate`, `save_test_predictions`).

---

```python
class KNNPatternModel(BasePatternModel):
```
> **Baris 9:** Definisi kelas `KNNPatternModel` yang **mewarisi** `BasePatternModel`.  
> Dengan inheritance, kelas ini otomatis mendapat method `fit()`, `evaluate()`, `predict_test()`, dll. dari kelas induk — tanpa perlu menulis ulang.

---

### Blok `__init__` — Konstruktor

```python
def __init__(self, n_neighbors: int = 9, weights: str = "distance",
             metric: str = "euclidean", scale_features: bool = False, **kwargs):
```
> **Baris 20:** Konstruktor kelas. Mendefinisikan hyperparameter dengan nilai default:
> - `n_neighbors=9` → Jumlah tetangga yang dipertimbangkan saat klasifikasi. Nilai ganjil untuk menghindari seri voting.
> - `weights="distance"` → Tetangga yang lebih dekat mendapat bobot lebih besar. Alternatif: `"uniform"` (semua sama).
> - `metric="euclidean"` → Jarak dihitung dengan rumus Euclidean: `d = sqrt(sum((xi - xj)^2))`
> - `scale_features=False` → Tidak menggunakan StandardScaler secara default. Pada dataset ini, KNN tanpa scaling lebih akurat (95.75% vs 56%) karena fitur `ram` yang berskala besar justru informatif.
> - `**kwargs` → Menerima argumen tambahan opsional yang diteruskan ke kelas induk.

```python
    self.n_neighbors = n_neighbors
    self.weights = weights
    self.metric = metric
    self.scale_features = scale_features
```
> **Baris 21–24:** Menyimpan nilai argumen sebagai **atribut instance**.  
> Pola ini memungkinkan `_build_model()` mengakses nilai hyperparameter lewat `self`.

```python
    super().__init__(
        name="K-Nearest Neighbors",
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        scale_features=scale_features,
        **kwargs
    )
```
> **Baris 25–32:** Memanggil konstruktor kelas induk `BasePatternModel`.  
> - `name="K-Nearest Neighbors"` → Nama yang ditampilkan di terminal dan grafik.
> - Hyperparameter diteruskan agar kelas induk dapat menyimpan metadata model.
> - Di dalam `super().__init__()`, kelas induk otomatis memanggil `self._build_model()` untuk membuat estimator scikit-learn.

---

### Method `_build_model` — Membangun Estimator

```python
def _build_model(self):
```
> **Baris 34:** Method abstrak dari `BasePatternModel` yang **wajib** diimplementasikan.  
> Tugasnya: mengembalikan objek estimator scikit-learn yang siap di-`fit()`.

```python
    knn = KNeighborsClassifier(
        n_neighbors=self.n_neighbors,
        weights=self.weights,
        metric=self.metric
    )
```
> **Baris 35–39:** Membuat objek `KNeighborsClassifier` dengan hyperparameter yang sudah disimpan.  
>
> **Cara kerja KNN saat `fit()`:**  
> KNN adalah *lazy learner* — tidak membangun model matematika. `fit()` hanya **menyimpan seluruh data latih** ke memori.
>
> **Cara kerja KNN saat `predict()`:**  
> Untuk setiap sampel uji:
> 1. Hitung jarak Euclidean ke semua 1600 sampel latih
> 2. Ambil 9 tetangga dengan jarak terkecil
> 3. Hitung bobot tiap tetangga (`weight = 1/d`)
> 4. Kelas dengan total bobot terbesar → prediksi label

```python
    if self.scale_features:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", knn)
        ])
    return knn
```
> **Baris 40–45:** Logika bersyarat berdasarkan `scale_features`:
>
> - Jika `scale_features=True`: mengembalikan **Pipeline** dua langkah:
>   1. `StandardScaler()` → normalisasi fitur (`fit_transform` saat training, `transform` saat testing)
>   2. `KNeighborsClassifier` → model KNN
>   Pipeline menjamin tidak ada *data leakage* (scaler hanya di-fit pada data latih).
>
> - Jika `scale_features=False` (default): mengembalikan objek KNN langsung tanpa preprocessing.

---

## Alur Eksekusi Saat `python run_single.py --model knn`

```
1. KNNPatternModel() dipanggil
       ↓
2. __init__() menyimpan hyperparameter
       ↓
3. super().__init__() memanggil _build_model()
       ↓
4. _build_model() membuat KNeighborsClassifier(n_neighbors=9, weights='distance', metric='euclidean')
       ↓
5. model.fit(X_train, y_train)   → menyimpan 1600 baris data latih
       ↓
6. model.predict(X_val)          → hitung jarak ke semua data, ambil 9 tetangga
       ↓
7. evaluate() → hitung Accuracy, Precision, Recall, F1-Score
       ↓
8. save_test_predictions()       → prediksi 1000 baris test.csv
       ↓
9. Simpan ke outputs/predictions/test_pred_knn.csv
```

---

## Hyperparameter & Pengaruhnya

| Hyperparameter | Default | Kecil | Besar |
|---|---|---|---|
| `n_neighbors` | 9 | Terlalu sensitif (overfitting) | Terlalu umum (underfitting) |
| `weights` | `distance` | — | — |
| `metric` | `euclidean` | — | — |

**Eksperimen empiris pada dataset ini:**

| Setting | Akurasi |
|---|---|
| k=9, tanpa scaling | **95.75%** ✅ |
| k=9, dengan StandardScaler | 56.50% |
| k=9, dengan MinMaxScaler | 45.50% |

> Scaling menurunkan performa drastis karena menghilangkan dominasi fitur `ram` (korelasi 0.917 dengan label) yang justru sangat informatif untuk perhitungan jarak.
