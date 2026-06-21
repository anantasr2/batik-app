import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # PENTING
import os

class BatikVintageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Klasifikasi Batik - Vintage Edition")
        self.root.state('zoomed') 
        
        # --- Konfigurasi Warna ---
        self.colors = {
            "bg_main": "#1E1310", "bg_panel": "#3E2723",
            "gold": "#FFD700", "antique_gold": "#C5A059",
            "text": "#EFEBE9", "btn_bg": "#5D4037", "btn_fg": "#FFD700"
        }
        self.root.configure(bg=self.colors["bg_main"])

        # --- Variabel ---
        self.model = None
        self.cap = None
        self.is_running = False
        
        # --- NAMA KELAS (Sesuaikan dengan Notebook) ---
        self.class_names = [
            'Aceh_Pintu_Aceh', 
            'Madura_Mataketeran', 
            'Maluku_Pala', 
            'Solo_Parang'
        ]

        # --- SETTING UKURAN GAMBAR ---
        self.img_height = 224 
        self.img_width = 224

        self.setup_ui()
        
        # Load model otomatis setelah UI muncul
        self.root.after(100, self.load_model_standard)

    def load_model_standard(self):
        # Pastikan path ini benar
        model_path = "best_model.h5"
        
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"File '{model_path}' tidak ditemukan!\nSimpan file model di folder yang sama dengan kode ini.")
            self.status_label.config(text="File Model Hilang!", fg="red")
            return

        try:
            self.status_label.config(text="Memuat model...", fg="yellow")
            self.root.update()

            # Load Model
            self.model = tf.keras.models.load_model(model_path)
            
            # Pemanasan (Warm-up)
            dummy = np.zeros((1, self.img_height, self.img_width, 3))
            dummy = preprocess_input(dummy)
            self.model.predict(dummy, verbose=0)
            
            self.status_label.config(text="Model Siap Digunakan", fg="#00FF00")
            print("Model berhasil dimuat.")
            
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error Load", f"Gagal memuat model.\n\nDetail: {e}")
            self.status_label.config(text="Error Memuat Model", fg="red")

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg=self.colors["bg_panel"], pady=15)
        header.pack(fill="x")
        
        tk.Label(header, text="⚜ KLASIFIKASI BATIK NUSANTARA ⚜", 
                 font=("Garamond", 28, "bold"), bg=self.colors["bg_panel"], fg=self.colors["gold"]).pack()

        self.status_label = tk.Label(header, text="Menunggu...", 
                 font=("Helvetica", 12, "italic"), bg=self.colors["bg_panel"], fg=self.colors["text"])
        self.status_label.pack()

        # Canvas
        self.canvas_frame = tk.Frame(self.root, bg=self.colors["bg_main"])
        self.canvas_frame.pack(expand=True, fill="both", padx=20, pady=10)
        
        self.display_lbl = tk.Label(self.canvas_frame, bg="black", bd=8, relief="ridge", highlightbackground=self.colors["antique_gold"])
        self.display_lbl.pack(expand=True)

        # Kontrol
        control_panel = tk.Frame(self.root, bg=self.colors["bg_panel"], pady=20)
        control_panel.pack(fill="x", side="bottom")

        btn_style = {"font": ("Times New Roman", 14, "bold"), "bg": self.colors["btn_bg"], "fg": self.colors["btn_fg"], "width": 18}
        
        tk.Button(control_panel, text="📂 Upload Foto", command=self.upload_image, **btn_style).pack(side="left", padx=20, expand=True)
        tk.Button(control_panel, text="📷 Live Webcam", command=self.start_webcam, **btn_style).pack(side="left", padx=20, expand=True)
        tk.Button(control_panel, text="⏹ STOP", command=self.stop_stream, bg="#8B0000", fg="white", font=("Times New Roman", 14, "bold"), width=10).pack(side="left", padx=20, expand=True)

    def preprocess_image(self, img_array):
        # Input img_array dari OpenCV adalah format BGR
        
        # 1. KONVERSI BGR KE RGB (Sangat Penting!)
        # Notebook Anda pakai load_img (RGB), sedangkan OpenCV pakai BGR.
        # Kalau tidak diubah, warna biru jadi merah, prediksi pasti salah.
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # 2. Resize sesuai input model
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # 3. Konversi ke float array
        img = img.astype("float32")
        
        # 4. PREPROCESSING MOBILENETV2 (Wajib sama dengan Notebook)
        # Jangan pakai manual / 255.0, pakai fungsi bawaan agar konsisten -1 s/d 1
        img = preprocess_input(img)
        
        # 5. Tambah dimensi batch (1, 224, 224, 3)
        return np.expand_dims(img, axis=0)

    def predict_and_annotate(self, frame):
        if self.model is None: return frame

        # PERBAIKAN: HAPUS LOGIKA ROI (CROP TENGAH)
        # Di notebook, Anda menggunakan seluruh gambar (load_img tanpa crop).
        # Jadi di sini kita juga harus menggunakan seluruh frame untuk diproses.
        
        try:
            # Gunakan SELURUH frame untuk prediksi
            # preprocess_image akan meresize seluruh frame ke 224x224 (squashing)
            input_data = self.preprocess_image(frame)
            
            # Prediksi
            preds = self.model.predict(input_data, verbose=0)
            
            # Handling output shape jika model outputnya list
            if isinstance(preds, list): preds = preds[0]
            
            # Ambil kelas tertinggi
            idx = np.argmax(preds)
            conf = np.max(preds) * 100
            
            if idx < len(self.class_names):
                label = self.class_names[idx]
            else:
                label = f"Class {idx}"
            
            # Visualisasi Teks di Pojok Kiri Atas (Overlay)
            # Agar tidak mengganggu gambar, kita taruh teks di atas
            label_text = label.replace("_", " ").replace("- CLEAN", "")
            
            # Background semi-transparan untuk teks
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 80), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # Garis tepi
            cv2.rectangle(frame, (0, 0), (400, 80), (0, 215, 255), 2)
            
            # Tulisan
            cv2.putText(frame, f"Prediksi: {label_text}", (10, 35), 
                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 215, 255), 1)
            
        except Exception as e:
            print(f"Prediction Error: {e}")
                
        return frame

    def start_webcam(self):
        self.stop_stream()
        if self.model:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error Webcam", "Tidak dapat mengakses webcam.")
                return
            self.is_running = True
            self.update_webcam()
        else:
            messagebox.showwarning("Info", "Tunggu sebentar, model sedang dimuat...")

    def update_webcam(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Webcam biasanya mirror, kita flip
                frame = cv2.flip(frame, 1) 
                
                # Prediksi & Gambar Kotak
                frame = self.predict_and_annotate(frame)
                
                # Tampilkan ke layar GUI
                self.show_frame(frame)
                self.root.after(30, self.update_webcam)

    def upload_image(self):
        self.stop_stream()
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path and self.model:
            # imread membaca sebagai BGR
            frame = cv2.imread(path)
            if frame is not None:
                frame = self.predict_and_annotate(frame)
                self.show_frame(frame)
            else:
                messagebox.showerror("Error", "Gagal membaca gambar.")

    def show_frame(self, frame):
        # Konversi BGR (OpenCV) ke RGB (Tkinter/PIL) untuk ditampilkan
        # Perhatikan: Ini HANYA untuk display mata manusia. 
        # Prediksi AI sudah dilakukan sebelumnya di predict_and_annotate pakai BGR->RGB sendiri.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize agar pas di layar GUI
        h, w, _ = frame.shape
        display_h = 600
        scale = display_h / h
        display_w = int(w * scale)
        
        frame = cv2.resize(frame, (display_w, display_h))
        
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.display_lbl.imgtk = img
        self.display_lbl.configure(image=img)

    def stop_stream(self):
        self.is_running = False
        if self.cap: 
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    root = tk.Tk()
    app = BatikVintageApp(root)
    root.mainloop()