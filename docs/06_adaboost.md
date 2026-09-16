# AdaBoost (Adaptive Boosting) — Penjelasan Kode Per Baris

**File:** `src/models/adaboost_model.py`  
**Akurasi:** 93.25% | **F1-Score:** 93.28% | **Training Time:** ~1.8 detik

---

## Kode Lengkap dengan Penjelasan

```python
"""
Implementasi Model AdaBoost (Adaptive Boosting) untuk Pengenalan Pola
"""
```
> **Baris 1–3:** Docstring modul.

---

```python
from sklearn.ensemble import AdaBoostClassifier
```
> **Baris 4:** Import `AdaBoostClassifier` dari scikit-learn.  
> Mengimplementasikan algoritma AdaBoost yang pertama kali diperkenalkan oleh Freund & Schapire (1997). scikit-learn mendukung dua varian:
> - `SAMME` (Stagewise Additive Modeling using Multi-class Exponential loss) → untuk multi-kelas
> - `SAMME.R` (versi Real/probabilistic) → deprecated di scikit-learn >= 1.4, akan dihapus di versi 1.6

```python
from sklearn.tree import DecisionTreeClassifier
```
> **Baris 5:** Import `DecisionTreeClassifier` — digunakan sebagai **base estimator** (weak learner) AdaBoost.  
> Meski AdaBoost bisa menggunakan sembarang classifier sebagai base learner, Decision Tree sangat lazim digunakan karena mudah dikontrol kompleksitasnya lewat `max_depth`.

```python
from src.models.base_model import BasePatternModel
```
> **Baris 6:** Import abstract base class untuk antarmuka seragam.

```python
from src.config import RANDOM_STATE
```
> **Baris 7:** Import seed `RANDOM_STATE = 42` untuk reproduktifitas.

---

```python
class AdaBoostPatternModel(BasePatternModel):
```
> **Baris 9:** Definisi kelas yang mewarisi `BasePatternModel`.

---

### Blok `__init__` — Konstruktor

```python
def __init__(self, n_estimators: int = 100, learning_rate: float = 0.5,
             max_depth: int = 3, random_state: int = RANDOM_STATE, **kwargs):
```
> **Baris 20:** Konstruktor dengan hyperparameter:
>
> - `n_estimators=100` → **Jumlah weak learner** (pohon) yang dilatih secara berurutan.  
>   Setiap iterasi menambah satu pohon baru yang berfokus pada sampel yang sebelumnya salah diklasifikasi.
>
> - `learning_rate=0.5` → **Mengontrol kontribusi tiap weak learner** terhadap prediksi akhir.  
>   Lebih tepatnya: bobot tiap estimator `alpha_t` dikalikan dengan `learning_rate`:
>   ```
>   H(x) = sign( Σ learning_rate × alpha_t × h_t(x) )
>   ```
>   - learning_rate kecil (0.01–0.1) → perlu n_estimators lebih banyak
>   - learning_rate besar (0.5–1.0) → belajar lebih cepat tapi bisa tidak stabil  
>   - Default 0.5 dipilih berdasarkan grid search: depth=3 + lr=0.5 → 93.25%
>
> - `max_depth=3` → **Kedalaman pohon base estimator**.  
>   Ini perbedaan utama dengan Gradient Boosting dalam implementasinya:
>   - AdaBoost klasik: `max_depth=1` (decision stump — hanya 1 split)
>   - Pada dataset ini: `max_depth=3` jauh lebih baik (93.25% vs 75% untuk depth=1)  
>   - Pohon depth=3 mampu menangkap interaksi antar fitur
>
> - `random_state=42` → Seed untuk reproduktifitas pembuatan pohon.

```python
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.random_state = random_state
```
> **Baris 21–24:** Menyimpan hyperparameter sebagai atribut instance.

```python
    super().__init__(
        name="AdaBoost",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
```
> **Baris 25–32:** Memanggil konstruktor kelas induk yang akan menjalankan `_build_model()` dan menyimpan hasilnya ke `self.model`.

---

### Method `_build_model` — Membangun Estimator

```python
def _build_model(self):
    base_estimator = DecisionTreeClassifier(
        max_depth=self.max_depth,
        random_state=self.random_state
    )
```
> **Baris 34–38:** Membuat **base estimator** — pohon keputusan yang sengaja dibuat dangkal (depth=3).  
> Ini adalah "weak learner" yang akan dilatih berulang kali dengan bobot sampel yang berubah tiap iterasi.  
> Disebut "weak" karena sengaja tidak sempurna — akurasi sedikit di atas random guessing sudah cukup.

```python
    try:
        # scikit-learn >= 1.2 menggunakan parameter 'estimator'
        return AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm="SAMME",
            random_state=self.random_state
        )
```
> **Baris 39–47:** Membuat `AdaBoostClassifier` dengan parameter `estimator` (nama baru di scikit-learn >= 1.2).
>
> **Penjelasan parameter `algorithm="SAMME"`:**
> - `SAMME` (Stagewise Additive Modeling using Multi-class Exponential loss) → dirancang untuk multi-kelas, tidak butuh output probabilitas dari base learner
> - `SAMME.R` → menggunakan probabilitas kelas, lebih akurat tapi deprecated di sklearn >= 1.4
> - Dipilih `SAMME` untuk kompatibilitas ke depan (forward compatibility)
>
> **Cara kerja AdaBoost dengan SAMME saat `fit(X_train, y_train)`:**
>
> ```
> Inisialisasi: bobot sampel wi = 1/N  untuk semua i (N=1600)
>
> Untuk t = 1 ... 100 (n_estimators):
>
>   1. Latih pohon h_t pada data dengan bobot {wi}
>      (sampel dengan bobot besar → pohon lebih fokus pada sampel ini)
>
>   2. Hitung weighted error:
>      ε_t = Σ(wi × 1[h_t(xi) ≠ yi]) / Σ(wi)
>
>   3. Hitung bobot estimator (semakin kecil error → bobot makin besar):
>      α_t = learning_rate × log((1-ε_t)/ε_t) + log(K-1)
>      (K = jumlah kelas = 4)
>
>   4. Update bobot sampel:
>      wi ← wi × exp(α_t × 1[h_t(xi) ≠ yi])
>      → Sampel yang SALAH diklasifikasi mendapat bobot LEBIH BESAR
>
>   5. Normalisasi bobot: wi ← wi / Σ(wi)
>
> Prediksi akhir:
>   H(x) = argmax_k [ Σ(α_t × 1[h_t(x) = k]) ]  untuk k = 0,1,2,3
> ```

```python
    except TypeError:
        # Kompatibilitas mundur jika versi scikit-learn lama
        return AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm="SAMME",
            random_state=self.random_state
        )
```
> **Baris 48–56:** Blok `except TypeError` untuk **kompatibilitas mundur** (backward compatibility).  
> scikit-learn < 1.2 menggunakan parameter `base_estimator` (bukan `estimator`).  
> Jika `AdaBoostClassifier(estimator=...)` melempar `TypeError` (parameter tidak dikenal), kode otomatis mencoba lagi dengan `base_estimator=...`.  
> Ini membuat kode berjalan di **berbagai versi scikit-learn** tanpa modifikasi.

---

## Perbedaan AdaBoost vs Gradient Boosting

| Aspek | AdaBoost | Gradient Boosting |
|---|---|---|
| Cara "fokus" pada error | Menambah **bobot sampel** yang salah | Melatih pohon pada **residual/gradien** |
| Base learner default | Decision stump (depth=1) | Pohon kecil (depth=3) |
| Fungsi Loss | Exponential loss | Log-loss / deviance |
| Sensitif terhadap noise | Ya (bobot outlier membesar) | Lebih tahan |
| Kecepatan | Lebih cepat (~1.8 detik) | Lebih lambat (~3.5 detik) |
| Akurasi (dataset ini) | 93.25% | 91.25% |

---

## Alur Eksekusi Saat `python run_single.py --model adaboost`

```
1. AdaBoostPatternModel() dipanggil
       ↓
2. __init__() → simpan n_estimators=100, learning_rate=0.5, max_depth=3
       ↓
3. super().__init__() → panggil _build_model()
       ↓
4. _build_model():
       a. Buat base_estimator = DecisionTreeClassifier(max_depth=3)
       b. Coba AdaBoostClassifier(estimator=base_estimator, algorithm='SAMME', ...)
       c. Jika TypeError → gunakan base_estimator=... (scikit-learn lama)
       ↓
5. model.fit(X_train, y_train)
       ↓ Inisialisasi bobot seragam 1/1600 untuk semua sampel
       ↓ Iterasi 1..100:
           - Latih pohon depth=3 dengan bobot saat ini
           - Hitung error, hitung alpha_t
           - Update bobot (perbesar untuk yang salah)
           - Normalisasi bobot
       → ~1.8 detik
       ↓
6. model.predict(X_val)
       ↓ Voting berbobot dari 100 pohon untuk 400 sampel
       ↓
7. evaluate() → Accuracy=93.25%, F1=93.28%
       ↓
8. save_test_predictions() → prediksi 1000 baris test.csv
       ↓
9. Simpan ke outputs/predictions/test_pred_adaboost.csv
```

---

## Hasil Grid Search Hyperparameter (Empiris)

| max_depth | learning_rate | Akurasi Validasi |
|---|---|---|
| 1 | 0.1 | 74.50% |
| 1 | 0.5 | 70.75% |
| 2 | 0.5 | 88.50% |
| 2 | 1.0 | 90.25% |
| **3** | **0.5** | **93.25%** ✅ (default) |
| 3 | 1.0 | 93.25% |
| 4 | 0.5 | 95.25% |
| 4 | 1.0 | 95.25% |

> `max_depth=3, learning_rate=0.5` dipilih karena akurasi kompetitif dengan waktu training yang efisien.  
> `max_depth=4` memberikan akurasi lebih tinggi (95.25%) tetapi training lebih lambat.
