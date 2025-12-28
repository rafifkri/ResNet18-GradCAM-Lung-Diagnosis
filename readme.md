
# Klasifikasi Penyakit Paru dengan Grad-CAM

Proyek ini mengimplementasikan **sistem klasifikasi CNN untuk penyakit paru** dari gambar X-ray dengan  **interpretasi Grad-CAM** . Sistem ini dapat memprediksi penyakit seperti  **NORMAL, PNEUMONIA, TUBERCULOSIS, dan UNKNOWN** , serta menampilkan area pada X-ray yang berkontribusi pada prediksi.

---

## Fitur

* Klasifikasi gambar X-ray paru menjadi 4 kelas.
* Membuat **Grad-CAM heatmap** dan overlay pada gambar asli.
* Aplikasi **Streamlit** interaktif untuk mengunggah gambar dan melihat prediksi.
* Tampilkan **bar probabilitas horizontal** untuk setiap kelas.
* Visualisasi klinis dengan kontrol transparansi overlay Grad-CAM.

---

## Struktur Proyek

```
lung-disease-gradcam/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   └── resnet18_best.pth
├── results/               # Hasil output Grad-CAM overlay
├── src/
│   ├── dataset.py
│   ├── gradcam.py
│   └── inference.py
├── app.py                 # Aplikasi Streamlit
├── requirements.txt
└── README.md
```

---

## Instalasi

1. Clone repository:

```bash
git clone <url-repo-anda>
cd lung-disease-gradcam
```

2. Buat virtual environment (opsional tapi direkomendasikan):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install semua dependency:

```bash
pip install -r requirements.txt
```

---

## Cara Penggunaan

### 1. Menjalankan Inference pada Satu Gambar

```bash
python -m src.inference --image data/test/<kelas>/<nama_gambar>.jpg
```

* Hasil overlay dan heatmap akan disimpan di folder `results/`.

### 2. Menjalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

* Unggah gambar X-ray.
* Lihat kelas prediksi dan tingkat kepercayaan (confidence).
* Lihat Grad-CAM overlay dan bar probabilitas horizontal untuk semua kelas.

---

## Catatan

* Transparansi Grad-CAM dapat diatur langsung di aplikasi Streamlit.
* Model yang digunakan: `ResNet-18` yang dilatih pada dataset Anda (weights di `models/resnet18_best.pth`).
* Pastikan struktur folder dataset benar:

```
data/
├── train/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│   ├── TUBERCULOSIS/
│   └── UNKNOWN/
├── val/
└── test/
```

---

## Kontribusi

Silakan buka issue atau pull request untuk:

* Menambahkan metrik interpretabilitas tambahan.
* Meningkatkan UI/UX Streamlit.
* Menambahkan dukungan untuk lebih banyak kelas penyakit paru.

---

## Lisensi

MIT License
