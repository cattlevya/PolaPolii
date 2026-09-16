# Support Vector Machine (SVM) — Penjelasan Kode Per Baris

**File:** `src/models/svm_model.py`  
**Akurasi:** 98.25% | **F1-Score:** 98.25% | **Training Time:** ~9 detik  
**Peringkat: #1 — Model Terbaik**

---

## Kode Lengkap dengan Penjelasan

```python
"""
Implementasi Model Support Vector Machine (SVM) untuk Pengenalan Pola
"""
```
> **Baris 1–3:** Docstring modul.

---

```python
from sklearn.svm import SVC
```
> **Baris 4:** Import `SVC` (Support Vector Classifier) dari scikit-learn.  
> `SVC` adalah implementasi SVM untuk klasifikasi. Untuk regresi tersedia `SVR`.  
> scikit-learn menggunakan library **libsvm** di bawahnya untuk komputasi yang efisien.

```python
from sklearn.preprocessing import StandardScaler
```
> **Baris 5:** Import StandardScaler — digunakan secara opsional lewat flag `scale_features`.

```python
from sklearn.pipeline import Pipeline
```
> **Baris 6:** Import Pipeline untuk menggabungkan scaler + SVM menjadi satu objek.

```python
from src.models.base_model import BasePatternModel
```
> **Baris 7:** Import abstract base class.

```python
from src.config import RANDOM_STATE
```
> **Baris 8:** Import konstanta `RANDOM_STATE = 42` untuk reproduktifitas.

---

```python
class SVMPatternModel(BasePatternModel):
```
> **Baris 10:** Definisi kelas yang mewarisi `BasePatternModel`.

---

### Blok `__init__` — Konstruktor

```python
def __init__(self, C: float = 1.0, kernel: str = "linear",
             gamma: str = "scale", scale_features: bool = False,
             probability: bool = False, random_state: int = RANDOM_STATE, **kwargs):
```
> **Baris 23:** Konstruktor dengan hyperparameter:
>
> - `C=1.0` → **Parameter regularisasi** (trade-off antara margin lebar vs kesalahan klasifikasi).
>   - **C besar** → margin lebih ketat, toleransi error kecil → bisa overfitting
>   - **C kecil** → margin lebih lebar, lebih toleran error → bisa underfitting
>   - Dari eksperimen: C=1.0 optimal untuk kernel linear pada dataset ini (akurasi 98.25%)
>
> - `kernel="linear"` → Jenis fungsi kernel untuk transformasi ruang fitur:
>   - `"linear"`: `K(xi, xj) = xi^T · xj` → Cocok jika data hampir terpisah linear
>   - `"rbf"` (Radial Basis Function): `K(xi, xj) = exp(-γ||xi-xj||²)` → Untuk batas keputusan non-linear
>   - `"poly"`: `K(xi, xj) = (γ·xi^T·xj + r)^d` → Polynomial kernel
>   - `"sigmoid"`: mirip fungsi aktivasi neural network
>   > Kernel linear terbaik pada dataset ini (98.25%) karena kelas-kelas hampir terpisah linear, terutama berkat fitur `ram`.
>
> - `gamma="scale"` → Parameter untuk kernel RBF/poly/sigmoid.
>   - `"scale"` = `1 / (n_features * X.var())` → Nilai otomatis yang adaptif terhadap data
>   - Tidak berpengaruh untuk kernel linear
>
> - `scale_features=False` → Tidak pakai StandardScaler. Sama seperti KNN, SVM tanpa scaling lebih baik pada dataset ini.
>
> - `probability=False` → Jika True, SVM menghitung probabilitas tiap kelas menggunakan Platt Scaling (cross-validation internal), yang membuat training **jauh lebih lambat**. Default False karena prediksi label sudah cukup (tidak butuh probabilitas).
>
> - `random_state=42` → Seed untuk shuffle data di internal solver.

```python
    self.C = C
    self.kernel = kernel
    self.gamma = gamma
    self.scale_features = scale_features
    self.probability = probability
    self.random_state = random_state
```
> **Baris 24–29:** Menyimpan semua hyperparameter sebagai atribut instance.

```python
    super().__init__(
        name="Support Vector Machine",
        C=C, kernel=kernel, gamma=gamma,
        scale_features=scale_features,
        probability=probability,
        random_state=random_state,
        **kwargs
    )
```
> **Baris 30–39:** Memanggil konstruktor kelas induk yang akan menjalankan `_build_model()`.

---

### Method `_build_model` — Membangun Estimator

```python
def _build_model(self):
    svc = SVC(
        C=self.C,
        kernel=self.kernel,
        gamma=self.gamma,
        probability=self.probability,
        random_state=self.random_state
    )
```
> **Baris 41–48:** Membuat objek `SVC` (Support Vector Classifier).
>
> **Cara kerja SVM saat `fit(X_train, y_train)`:**
>
> Untuk masalah multi-kelas (4 kelas), scikit-learn menggunakan strategi **One-vs-One (OVO)**:
> - Dibuat `C(4,2) = 6` classifier biner: kelas 0 vs 1, kelas 0 vs 2, ..., kelas 2 vs 3
> - Setiap classifier menemukan hyperplane pemisah optimal:
>   ```
>   Minimize:  (1/2)||w||² + C·Σ(ξᵢ)
>   Subject to: yᵢ(w^T·xᵢ + b) ≥ 1 - ξᵢ,   ξᵢ ≥ 0
>   ```
>   (ξᵢ adalah slack variable untuk soft margin)
> - Data yang berada di batas margin → **Support Vectors** → hanya ini yang disimpan
>
> **Cara kerja saat `predict(X_val)`:**
> - Jalankan 6 classifier biner pada setiap sampel
> - Kelas yang paling sering "menang" dari 6 pertandingan → prediksi label

```python
    if self.scale_features:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", svc)
        ])
    return svc
```
> **Baris 49–54:** Logika bersyarat — dengan atau tanpa StandardScaler.
>
> **Hasil eksperimen empiris:**
>
> | Konfigurasi | Akurasi |
> |---|---|
> | Linear, tanpa scaling | **98.25%** ✅ |
> | Linear, dengan scaling | 96.25% |
> | RBF + scaling (C=10) | 89.00% |
>
> Kernel linear tanpa scaling terbaik karena batas keputusan antar kelas bersifat hampir linear di ruang fitur asli.

---

## Konsep Support Vector — Visualisasi

```
Kelas A (●)        Kelas B (■)
    ●   ●            ■   ■
      ●     ← margin →  ■
      ●  ← hyperplane → ■
      ●      ←  margin → ■
    ●   ●       ← support vector (●)
                ← support vector (■)
```
> SVM hanya menyimpan **support vectors** (data di batas margin), bukan semua data latih.  
> Ini membuat prediksi efisien meski training lambat untuk dataset besar.

---

## Alur Eksekusi Saat `python run_single.py --model svm`

```
1. SVMPatternModel() dipanggil
       ↓
2. __init__() → simpan C=1.0, kernel='linear', probability=False
       ↓
3. super().__init__() → panggil _build_model()
       ↓
4. _build_model() → buat SVC(C=1.0, kernel='linear', probability=False)
       ↓
5. model.fit(X_train, y_train)
       ↓ Buat 6 classifier biner (One-vs-One untuk 4 kelas)
       ↓ Selesaikan Quadratic Programming untuk tiap classifier
       ↓ Temukan support vectors dan hyperplane optimal
       → ~9 detik
       ↓
6. model.predict(X_val)
       ↓ Voting dari 6 binary classifier untuk 400 sampel
       ↓
7. evaluate() → Accuracy=98.25%, F1=98.25%
       ↓
8. save_test_predictions() → prediksi 1000 baris test.csv
       ↓
9. Simpan ke outputs/predictions/test_pred_svm.csv
```

---

## Hyperparameter & Pengaruhnya

| Hyperparameter | Default | Pengaruh |
|---|---|---|
| `C` | 1.0 | Trade-off regularisasi vs training error |
| `kernel` | `linear` | Jenis transformasi ruang fitur |
| `gamma` | `scale` | Radius pengaruh kernel RBF (tidak aktif untuk linear) |
| `probability` | False | True = lebih lambat ~3-5x, tapi bisa output probabilitas |
