# Decision Tree (Pohon Keputusan) — Penjelasan Kode Per Baris

**File:** `src/models/decision_tree_model.py`  
**Akurasi:** 84.50% | **F1-Score:** 84.54% | **Training Time:** ~0.02 detik

---

## Kode Lengkap dengan Penjelasan

```python
"""
Implementasi Model Decision Tree (Pohon Keputusan) untuk Pengenalan Pola
"""
```
> **Baris 1–3:** Docstring modul.

---

```python
from sklearn.tree import DecisionTreeClassifier
```
> **Baris 4:** Import `DecisionTreeClassifier` dari scikit-learn.  
> Kelas ini mengimplementasikan algoritma CART (Classification and Regression Trees) yang membangun pohon keputusan biner secara rekursif.

```python
from src.models.base_model import BasePatternModel
```
> **Baris 5:** Import abstract base class untuk antarmuka seragam antar semua model.

```python
from src.config import RANDOM_STATE
```
> **Baris 6:** Import konstanta `RANDOM_STATE = 42`.  
> Decision Tree menggunakan random state untuk memastikan hasil reproduktif — saat ada beberapa fitur yang menghasilkan split sama baiknya, urutan pemilihannya ditentukan secara acak namun tetap (deterministic).

---

```python
class DecisionTreePatternModel(BasePatternModel):
```
> **Baris 8:** Definisi kelas. Mewarisi `BasePatternModel` untuk mendapatkan method `fit()`, `evaluate()`, `predict_test()`, dll.

---

### Blok `__init__` — Konstruktor

```python
def __init__(self, criterion: str = "gini", max_depth: int = 10,
             min_samples_split: int = 5, random_state: int = RANDOM_STATE, **kwargs):
```
> **Baris 19:** Konstruktor dengan hyperparameter:
>
> - `criterion="gini"` → Kriteria pengukuran impuritas untuk memilih split terbaik.
>   - **Gini Impurity:** `Gini(S) = 1 - Σ(p_k)²` — mengukur probabilitas salah klasifikasi acak.
>   - Alternatif: `"entropy"` menggunakan Information Gain: `IG = H(parent) - Σ(w_i * H(child_i))`
>
> - `max_depth=10` → Kedalaman maksimal pohon. Membatasi kompleksitas untuk mencegah overfitting.
>   - Tanpa batas: pohon bisa sangat dalam dan hafal data latih (overfitting).
>   - Terlalu dangkal: underfitting.
>
> - `min_samples_split=5` → Jumlah minimum sampel yang harus ada di sebuah node agar node tersebut **boleh dipecah (split)**.
>   - Nilai 5 mencegah pohon membuat percabangan dari hanya 1–2 sampel yang tidak representatif.
>
> - `random_state=42` → Seed untuk reproduktifitas hasil.

```python
    self.criterion = criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.random_state = random_state
```
> **Baris 20–23:** Menyimpan hyperparameter sebagai atribut instance.

```python
    super().__init__(
        name="Decision Tree",
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        **kwargs
    )
```
> **Baris 24–31:** Memanggil konstruktor kelas induk.  
> Di dalam `super().__init__()`, kelas induk memanggil `self._build_model()` yang mengembalikan estimator scikit-learn, lalu menyimpannya sebagai `self.model`.

---

### Method `_build_model` — Membangun Estimator

```python
def _build_model(self):
    return DecisionTreeClassifier(
        criterion=self.criterion,
        max_depth=self.max_depth,
        min_samples_split=self.min_samples_split,
        random_state=self.random_state
    )
```
> **Baris 33–39:** Membuat dan mengembalikan objek `DecisionTreeClassifier`.
>
> **Cara kerja saat `fit(X_train, y_train)`:**
> 1. Mulai dari root node dengan semua 1600 sampel
> 2. Untuk setiap fitur dan setiap nilai threshold yang mungkin, hitung **Gini gain** dari split tersebut
> 3. Pilih kombinasi fitur + threshold dengan **Gini gain tertinggi** (impuritas turun paling banyak)
> 4. Pecah data menjadi dua subset (kiri: ≤ threshold, kanan: > threshold)
> 5. Ulangi rekursif pada tiap subset sampai kondisi berhenti:
>    - Kedalaman mencapai `max_depth=10`, atau
>    - Jumlah sampel < `min_samples_split=5`, atau
>    - Node sudah "murni" (hanya satu kelas)
>
> **Cara kerja saat `predict(X_val)`:**
> 1. Untuk setiap sampel, ikuti aturan di pohon dari root ke leaf
> 2. Kelas mayoritas di leaf node → prediksi label

---

## Visualisasi Proses Split

```
Root Node: 1600 sampel
    ├── ram <= 1500?
    │     ├── YES: 800 sampel
    │     │     ├── ram <= 800?
    │     │     │     ├── YES → Kelas 0 (Low)
    │     │     │     └── NO  → Kelas 1 (Medium)
    │     │     └── ...
    │     └── NO: 800 sampel
    │           ├── ram <= 2500?
    │           │     ├── YES → Kelas 2 (High)
    │           │     └── NO  → Kelas 3 (Very High)
    │           └── ...
    └── ...
```
> Fitur `ram` dengan korelasi 0.917 cenderung selalu menjadi split pertama (root).

---

## Alur Eksekusi Saat `python run_single.py --model decision_tree`

```
1. DecisionTreePatternModel() dipanggil
       ↓
2. __init__() → simpan hyperparameter
       ↓
3. super().__init__() → panggil _build_model()
       ↓
4. _build_model() → buat DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=5)
       ↓
5. model.fit(X_train, y_train) → bangun pohon secara rekursif dari 1600 data
       ↓
6. model.predict(X_val) → telusuri pohon untuk 400 data validasi
       ↓
7. evaluate() → Accuracy=84.50%, F1=84.54%
       ↓
8. save_test_predictions() → prediksi 1000 baris test.csv
       ↓
9. Simpan ke outputs/predictions/test_pred_decision_tree.csv
```

---

## Hyperparameter & Pengaruhnya

| Hyperparameter | Default | Efek jika Terlalu Kecil | Efek jika Terlalu Besar |
|---|---|---|---|
| `max_depth` | 10 | Underfitting (terlalu simpel) | Overfitting (hafal data latih) |
| `min_samples_split` | 5 | Overfitting (split terlalu kecil) | Underfitting (split tidak cukup) |
| `criterion` | `gini` | — | — |

**Catatan:** Decision Tree tidak memerlukan normalisasi/scaling fitur karena pemisahan bersifat **ordinal** (hanya membandingkan nilai relatif: "apakah ram ≤ 1500?"), bukan berbasis jarak.
