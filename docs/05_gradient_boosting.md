# Gradient Boosting — Penjelasan Kode Per Baris

**File:** `src/models/gradient_boosting_model.py`  
**Akurasi:** 91.25% | **F1-Score:** 91.23% | **Training Time:** ~3–4 detik

---

## Kode Lengkap dengan Penjelasan

```python
"""
Implementasi Model Gradient Boosting untuk Pengenalan Pola
"""
```
> **Baris 1–3:** Docstring modul.

---

```python
from sklearn.ensemble import GradientBoostingClassifier
```
> **Baris 4:** Import `GradientBoostingClassifier` dari scikit-learn.  
> Ini adalah implementasi **GBDT (Gradient Boosted Decision Trees)** — algoritma ensemble sequential yang sangat populer. Versi yang lebih cepat adalah `HistGradientBoostingClassifier` (histogram-based) atau library eksternal seperti XGBoost dan LightGBM.

```python
from src.models.base_model import BasePatternModel
```
> **Baris 5:** Import abstract base class.

```python
from src.config import RANDOM_STATE
```
> **Baris 6:** Import seed untuk reproduktifitas hasil.

---

```python
class GradientBoostingPatternModel(BasePatternModel):
```
> **Baris 8:** Definisi kelas yang mewarisi `BasePatternModel`.

---

### Blok `__init__` — Konstruktor

```python
def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
             max_depth: int = 3, random_state: int = RANDOM_STATE, **kwargs):
```
> **Baris 19:** Konstruktor dengan hyperparameter utama:
>
> - `n_estimators=100` → **Jumlah pohon keputusan** yang dibangun secara berurutan (sequential).  
>   Setiap pohon baru mencoba memperbaiki kesalahan pohon-pohon sebelumnya.  
>   - Terlalu sedikit → underfitting (model terlalu simpel)
>   - Terlalu banyak → overfitting + komputasi makin lama
>
> - `learning_rate=0.1` → **Kontribusi setiap pohon baru** terhadap prediksi akhir (disebut juga "shrinkage").
>   ```
>   F_m(x) = F_{m-1}(x) + learning_rate × h_m(x)
>   ```
>   - Nilai kecil (0.01–0.1) → model lebih konservatif, butuh lebih banyak estimator
>   - Nilai besar (0.5–1.0) → bisa overfitting, konvergensi lebih cepat
>   - `learning_rate × n_estimators` perlu diseimbangkan: jika learning_rate turun, tambah n_estimators.
>
> - `max_depth=3` → **Kedalaman setiap pohon individu** dalam ensemble.  
>   Pohon dangkal (depth 2–5) disebut **weak learner** — sengaja dibuat tidak sempurna.  
>   Kumpulan banyak weak learner → **strong learner**.
>   - depth=1 → decision stump (hanya 1 split)
>   - depth=3 → dapat menangkap interaksi hingga 3 fitur
>
> - `random_state=42` → Seed untuk subsampling data dan fitur.

```python
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.random_state = random_state
```
> **Baris 20–23:** Menyimpan hyperparameter sebagai atribut instance.

```python
    super().__init__(
        name="Gradient Boosting",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
```
> **Baris 24–31:** Memanggil konstruktor kelas induk yang menjalankan `_build_model()`.

---

### Method `_build_model` — Membangun Estimator

```python
def _build_model(self):
    return GradientBoostingClassifier(
        n_estimators=self.n_estimators,
        learning_rate=self.learning_rate,
        max_depth=self.max_depth,
        random_state=self.random_state
    )
```
> **Baris 33–39:** Membuat dan mengembalikan objek `GradientBoostingClassifier`.
>
> **Cara kerja saat `fit(X_train, y_train)`:**
>
> Untuk multi-kelas (4 kelas), Gradient Boosting menggunakan strategi **One-vs-Rest** — melatih **4 × 100 = 400 pohon** total (100 pohon untuk tiap kelas).
>
> **Iterasi Boosting (per kelas):**
>
> ```
> Iterasi 0:
>   F_0(x) = nilai awal (rata-rata target / log-odds)
>
> Iterasi 1:
>   r_i = -∂L(yi, F_0(xi))/∂F_0(xi)   ← negatif gradien loss (residual semu)
>   h_1 = pohon yang difit pada {xi, r_i}
>   F_1(x) = F_0(x) + 0.1 × h_1(x)
>
> Iterasi 2:
>   r_i = -∂L(yi, F_1(xi))/∂F_1(xi)
>   h_2 = pohon yang difit pada {xi, r_i}
>   F_2(x) = F_1(x) + 0.1 × h_2(x)
>
> ... (100 iterasi)
>
> Prediksi akhir: argmax_C [ F_100^C(x) ]
> ```
>
> **Fungsi Loss** yang digunakan oleh default: `deviance` (log-loss untuk klasifikasi multikelas).
>
> **Mengapa 3–4 detik?**  
> Harus membangun 400 pohon secara **sequential** (tidak bisa diparalelkan seperti Random Forest).

---

## Visualisasi Proses Boosting

```
Iterasi 1: Pohon simple → prediksi kasar, error besar
Iterasi 2: Pohon fokus pada error iterasi 1 → error berkurang
Iterasi 3: Pohon fokus pada sisa error → error makin berkurang
...
Iterasi 100: Kombinasi semua → prediksi akurat

Ensemble: F(x) = F_0 + 0.1·h_1 + 0.1·h_2 + ... + 0.1·h_100
```

---

## Alur Eksekusi Saat `python run_single.py --model gradient_boosting`

```
1. GradientBoostingPatternModel() dipanggil
       ↓
2. __init__() → simpan n_estimators=100, learning_rate=0.1, max_depth=3
       ↓
3. super().__init__() → panggil _build_model()
       ↓
4. _build_model() → buat GradientBoostingClassifier(...)
       ↓
5. model.fit(X_train, y_train)
       ↓ Bangun 4 × 100 = 400 pohon secara sequential
       ↓ Tiap pohon difit pada negatif gradien loss
       → ~3–4 detik
       ↓
6. model.predict(X_val)
       ↓ Jumlahkan 100 prediksi per kelas, ambil argmax
       ↓
7. evaluate() → Accuracy=91.25%, F1=91.23%
       ↓
8. save_test_predictions() → prediksi 1000 baris test.csv
       ↓
9. Simpan ke outputs/predictions/test_pred_gradient_boosting.csv
```

---

## Hyperparameter & Pengaruhnya

| Hyperparameter | Default | Kecil | Besar |
|---|---|---|---|
| `n_estimators` | 100 | Underfitting | Overfitting + lambat |
| `learning_rate` | 0.1 | Butuh n_estimators lebih banyak | Overfitting cepat |
| `max_depth` | 3 | Weak learner terlalu sederhana | Model terlalu kompleks |

> **Tips:** `learning_rate` dan `n_estimators` harus diseimbangkan. Menurunkan learning_rate biasanya membutuhkan penambahan n_estimators untuk performa sama.

**Catatan:** Gradient Boosting tidak memerlukan scaling fitur karena menggunakan decision tree sebagai base learner (pemisahan bersifat ordinal).
