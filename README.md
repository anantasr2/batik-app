# 🪕 Real-Time Batik Motif Classification Desktop Application using MobileNetV2

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)](https://tensorflow.org)
[![GUI](https://img.shields.io/badge/GUI-Tkinter-green)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-CC--BY--SA-lightgrey)](https://creativecommons.org/licenses/by-sa/4.0/)

Aplikasi desktop berbasis *Deep Learning* yang dirancang untuk mengklasifikasikan motif batik Indonesia secara **real-time** lewat kamera perangkat (*webcam*) maupun melalui **unggah gambar manual**. Project ini menggunakan metode *Transfer Learning* dengan arsitektur **MobileNetV2** untuk menghasilkan performa inferensi yang cepat dan efisien pada perangkat desktop dengan sumber daya komputasi terbatas.

Project ini dikembangkan sebagai bagian dari penelitian ilmiah di **Universitas Dian Nuswantoro** (Fakultas Ilmu Komputer, Program Studi Teknik Informatika) untuk mendukung digitalisasi, identifikasi otomatis, serta pelestarian warisan budaya Indonesia.

---

## 🚀 Fitur Utama
- **Real-Time Classification via Webcam:** Identifikasi motif batik secara langsung melalui kamera secara kontinu.
- **Manual Image Upload:** Unggah file gambar batik (`.png`, `.jpg`, `.jpeg`) melalui tombol penjelajah (*browse*) untuk langsung diprediksi oleh model.
- **Interactive GUI (Tkinter):** Antarmuka desktop yang ringan, responsif, dan mudah digunakan oleh pengguna awam.
- **Model Evaluation Report:** Menyajikan visualisasi laporan performa model (*Confusion Matrix* dan *Classification Report*) langsung di dalam sistem aplikasi.

---

## 📊 Dataset & Cakupan Kelas
Dataset yang digunakan dalam penelitian ini mencakup **4 kategori utama** motif batik Indonesia yang bersumber dari platform Kaggle dengan distribusi data yang seimbang (total 72 citra latih/uji awal):
1. **Pintu Aceh** (Karakteristik geometris simetris khas Aceh)
2. **Madura Mataketeran** (Karakteristik elemen repetitif bunga/lengkung warna cerah)
3. **Maluku Pala** (Karakteristik elemen rempah-rempah Maluku)
4. **Solo Parang** (Karakteristik pola diagonal khas Jawa Tengah)

---

## 🧠 Metodologi & Arsitektur Model

### 1. Pre-processing & Augmentasi Gambar
- **Resizing:** Dimensi gambar diseragamkan menjadi ukuran `224×224×3` piksel untuk memenuhi standar input arsitektur MobileNetV2.
- **Data Augmentation:** Untuk mengatasi keterbatasan jumlah dataset dan menghindari *overfitting*, diterapkan teknik augmentasi berupa *Horizontal Flip, Rotation (30°), Zoom Range (0.3), Width/Height Shift (0.3),* dan *Shear Range (0.3)* dengan pengisian area kosong menggunakan metode *Nearest*.

### 2. Arsitektur Deep Learning
Model memanfaatkan keunggulan *Transfer Learning* dari **MobileNetV2** yang berbasis *depthwise separable convolution* dan *inverted residual block* untuk meminimalkan jumlah parameter tanpa mengorbankan akurasi. Model dikompilasi menggunakan optimizer **Adam** dan fungsi loss **Categorical Crossentropy**.

---

## 📈 Hasil Evaluasi & Performa Penjajakan

Model mencapai titik optimal pada **Epoch ke-7** dengan metrik performa akhir proses *training*:
- **Akurasi Validasi Model:** 94%
- **Loss Validasi:** 0.4032

### Kinerja Pengujian pada Aplikasi Desktop:
Pengujian dilakukan melalui dua skenario (Statis via unggah gambar dan Real-time via Webcam) untuk melihat stabilitas akurasi sistem di bawah kondisi dunia nyata:

| Kategori Motif Batik | Akurasi (Statis / Upload Gambar) | Akurasi (Real-Time Webcam) |
| :--- | :---: | :---: |
| **Solo Parang** | 96% - 98% | 92% - 95% |
| **Pintu Aceh** | 95% - 97% | 90% |
| **Maluku Pala** | 93% - 94% | 90% - 92% |
| **Madura Mataketeran** | 90% - 92% | 89% - 91% |

*Kesimpulan Pengujian:* Metode pengujian statis menghasilkan akurasi yang lebih tinggi karena kualitas citra masukan terkontrol secara digital. Kendati demikian, performa real-time menggunakan webcam tetap tangguh (konsisten di atas 89%) meskipun dihadapkan pada variasi pencahayaan (*noise* lingkungan).

---

## 📁 Struktur Folder Project
```text
📦 batik-classification-desktop
├── models/
│   └── model_batik.h5        # File Trained Model berformat H5 (Keras/TensorFlow)
├── app/
│   ├── app_gui.py            # File utama untuk menjalankan aplikasi antarmuka Tkinter
│   └── utils.py              # File penunjang untuk preprocessing citra input
├── docs/
│   └── Jurnal_Batik.docx     # Dokumen paper penelitian terkait
├── .gitignore                # Mengabaikan file cache/temporary (*__pycache__*)
├── requirements.txt          # Daftar library dependency Python
└── README.md                 # Dokumentasi utama project (file ini)
--

## 🛠️ Cara Menjalankan Project di Lokal

### 1. Clone Repository
Buka Terminal atau Git Bash, lalu jalankan perintah berikut untuk mengunduh kode ke komputermu:
``bash
git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
cd nama-repo-kamu

### 2. Install Dependencies
Pastikan kamu menggunakan Python 3.8 ke atas. Install seluruh library yang dibutuhkan dengan menjalankan perintah:
``bash
pip install -r requirements.txt

### 3. Jalankan Aplikasi Desktop
Untuk membuka GUI Tkinter dan mulai melakukan deteksi, jalankan perintah berikut:
``bash
python app/app_gui.py