# 🚗 Drowsiness Detection (Deteksi Mengantuk) using YOLOv8n & Streamlit

[![Python Version](https://img.shields.io/badge/python-3.9.13-blue.svg)](https://www.python.org/downloads/release/python-3913/)
[![Framework](https://img.shields.io/badge/Framework-YOLOv8%20%7C%20Ultralytics-red)](https://github.com/ultralytics/ultralytics)
[![GUI](https://img.shields.io/badge/GUI-Streamlit-orange)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Aplikasi berbasis komputer visi (*Computer Vision*) dan *Deep Learning* yang dirancang untuk mendeteksi tingkat kantuk pada pengemudi atau pengguna secara *real-time* melalui kamera perangkat (webcam). 

Project ini menggunakan arsitektur **YOLOv8n (YOLOv8 Nano)** yang sangat ringan dan efisien untuk pemrosesan berkecepatan tinggi, serta diintegrasikan dengan **Streamlit** sebagai antarmuka (GUI) berbasis web yang interaktif dan mudah dioperasikan.

---

## 🚀 Fitur Utama
- **Real-Time Webcam Detection:** Mendeteksi tanda-tanda mengantuk (seperti mata terpejam atau menguap) secara langsung melalui kamera secara kontinu.
- **High-Speed Inference:** Menggunakan varian model YOLOv8n (Nano) sehingga proses deteksi berjalan sangat lancar dengan *latency* yang rendah.
- **Web-Based Interface:** Antarmuka modern dan responsif yang berjalan langsung di browser berkat framework Streamlit.
- **Visual Alert Indicator:** Memberikan penanda visual pada layar ketika sistem mendeteksi pengguna sedang dalam kondisi mengantuk.

---

## 🛠️ Cara Menjalankan Project di Lokal

Ikuti langkah-langkah di bawah ini untuk memasang dan menjalankan project ini di komputer atau laptop kamu:

### 1. Prasyarat (Prerequisites)
Project ini dikembangkan dan berjalan optimal menggunakan **Python 3.9.13**. Jika belum terinstal, silakan unduh versi resminya di sini:
- [Download Python 3.9.13](https://www.python.org/downloads/release/python-3913/)

### 2. Clone Repository
Buka Terminal, Git Bash, atau Command Prompt, lalu klon repository ini ke komputer lokalmu:
```bash
git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
cd nama-repo-kamu

3. Setup Environment & Install Dependencies
Sangat disarankan untuk menggunakan virtual environment (venv) agar library tidak bentrok dengan project Python lainnya. Jalankan perintah berikut untuk menginstal semua package pendukung yang dibutuhkan:
Bash
pip install -r requirements.txt
(Pastikan package utama seperti ultralytics, streamlit, dan opencv-python sudah terdaftar di dalam file requirements.txt kamu).

4. Jalankan Aplikasi Streamlit
Untuk membuka dan menjalankan antarmuka aplikasi di browser web, ketik perintah berikut di terminal:
Bash
streamlit run Welcome.py

📁 Struktur Folder Project
📦 nama-repo-kamu
├── .gitignore               # Mengabaikan file cache/temporary saat push ke GitHub
├── README.md                # Dokumentasi utama project (file ini)
├── requirements.txt         # Daftar library Python yang dibutuhkan
├── Welcome.py               # File utama/entry point untuk aplikasi Streamlit
└── models/
    └── best.pt              # File weights hasil training model YOLOv8