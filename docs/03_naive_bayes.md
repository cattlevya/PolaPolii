# Naive Bayes (Gaussian NB) — Penjelasan Kode Per Baris

**File:** `src/models/naive_bayes_model.py`  
**Akurasi:** 81.00% | **F1-Score:** 81.05% | **Training Time:** <0.01 detik

---

## Kode Lengkap dengan Penjelasan

```python
"""
Implementasi Model Naive Bayes (Gaussian NB) untuk Pengenalan Pola
"""
```
> **Baris 1–3:** Docstring modul.

---

```python
from sklearn.naive_bayes import GaussianNB
```
> **Baris 4:** Import `GaussianNB` dari scikit-learn.  
> Ini adalah implementasi Naive Bayes yang mengasumsikan bahwa nilai setiap fitur kontinu **berdistribusi Gaussian (normal)** pada masing-masing kelas.  
> Scikit-learn juga menyediakan varian lain: `MultinomialNB` (untuk data frekuensi/teks) dan `BernoulliNB` (untuk data biner).

```python
from src.models.base_model import BasePatternModel
```
> **Baris 5:** Import abstract base class untuk antarmuka seragam.

---

```python
class NaiveBayesPatternModel(BasePatternModel):
```
> **Baris 7:** Definisi kelas. Mewarisi `BasePatternModel`.

---

### Blok `__init__` — Konstruktor

```python
def __init__(self, var_smoothing: float = 1e-9, **kwargs):
```
> **Baris 19:** Konstruktor dengan satu hyperparameter:
>
> - `var_smoothing=1e-9` → Nilai kecil yang ditambahkan ke semua variansi fitur.
>
>   **Alasan:** Jika sebuah fitur memiliki variansi = 0 (semua nilai identik dalam satu kelas), pembagi dalam formula Gaussian menjadi 0, menyebabkan pembagian dengan nol (error). `var_smoothing` mencegah hal ini dengan menambahkan nilai sangat kecil (0.000000001) ke variansi.
>
>   Formula Gaussian yang digunakan:
>   ```
>   P(xi | C) = 1 / sqrt(2π(σ² + var_smoothing)) * exp(-(xi - μ)² / (2(σ² + var_smoothing)))
>   ```

```python
    self.var_smoothing = var_smoothing
```
> **Baris 20:** Menyimpan hyperparameter sebagai atribut instance.

```python
    super().__init__(
        name="Naive Bayes",
        var_smoothing=var_smoothing,
        **kwargs
    )
```
> **Baris 21–25:** Memanggil konstruktor kelas induk yang akan menjalankan `_build_model()` secara otomatis dan menyimpan hasilnya ke `self.model`.

---

### Method `_build_model` — Membangun Estimator

```python
def _build_model(self):
    return GaussianNB(var_smoothing=self.var_smoothing)
```
> **Baris 27–28:** Membuat dan mengembalikan objek `GaussianNB`.
>
> **Cara kerja saat `fit(X_train, y_train)`:**
>
> Untuk setiap kelas C (0, 1, 2, 3) dan setiap fitur xi, hitung dan simpan:
> - **Prior probabilitas kelas:** `P(C) = jumlah_sampel_kelas_C / total_sampel`
> - **Mean (μ):** rata-rata nilai xi untuk semua sampel berkelas C
> - **Variansi (σ²):** variansi nilai xi untuk semua sampel berkelas C
>
> Total parameter yang disimpan: `4 kelas × 20 fitur × 2 nilai (mean, variance)` = 160 parameter.
>
> **Cara kerja saat `predict(X_val)`:**
>
> Untuk setiap sampel baru x = [x1, x2, ..., x20]:
> 1. Hitung **log-posterior** untuk setiap kelas C:
>    ```
>    log P(C|x) = log P(C) + Σ log P(xi|C)  untuk i=1..20
>    ```
>    (Logaritma digunakan untuk menghindari underflow perkalian banyak probabilitas kecil)
>
> 2. Pilih kelas dengan **log-posterior tertinggi** sebagai prediksi:
>    ```
>    y_pred = argmax_C [ log P(C) + Σ log P(xi|C) ]
>    ```
>
> **Mengapa disebut "Naive"?**  
> Karena mengasumsikan semua fitur **saling independen bersyarat** terhadap kelas:
> ```
> P(x|C) = P(x1|C) × P(x2|C) × ... × P(x20|C)
> ```
> Asumsi ini "naif" karena dalam kenyataan fitur sering berkorelasi (contoh: `px_height` dan `px_width`).

---

## Alur Eksekusi Saat `python run_single.py --model naive_bayes`

```
1. NaiveBayesPatternModel() dipanggil
       ↓
2. __init__() → simpan var_smoothing=1e-9
       ↓
3. super().__init__() → panggil _build_model()
       ↓
4. _build_model() → buat GaussianNB(var_smoothing=1e-9)
       ↓
5. model.fit(X_train, y_train)
       ↓ Hitung P(C) untuk 4 kelas
       ↓ Hitung mean & variance tiap fitur per kelas (4×20=80 nilai)
       → Selesai dalam <0.01 detik!
       ↓
6. model.predict(X_val)
       ↓ Hitung log P(C|x) untuk 400 sampel × 4 kelas
       ↓ Pilih argmax
       ↓
7. evaluate() → Accuracy=81.00%, F1=81.05%
       ↓
8. save_test_predictions() → prediksi 1000 baris test.csv
       ↓
9. Simpan ke outputs/predictions/test_pred_naive_bayes.csv
```

---

## Kelebihan dan Keterbatasan Naive Bayes

| Aspek | Nilai |
|---|---|
| Kecepatan training | ⚡ Sangat cepat (<0.01 detik) |
| Kebutuhan data latih | Sedikit sudah cukup |
| Perlu scaling | Tidak |
| Toleransi data tidak relevan | Sedang |
| Akurasi pada dataset ini | 81% — terendah di antara semua metode |

**Mengapa akurasi relatif rendah pada dataset ini?**
1. **Pelanggaran asumsi independensi:** Fitur `px_height` dan `px_width` berkorelasi tinggi satu sama lain.
2. **Asumsi Gaussian mungkin tidak berlaku:** Fitur biner seperti `blue`, `dual_sim`, `four_g` tidak berdistribusi normal.
3. **Fitur tidak informatif mengganggu:** Dengan asumsi independensi, fitur noise seperti `m_dep` (korelasi 0.001) ikut berkontribusi, padahal seharusnya diabaikan.
