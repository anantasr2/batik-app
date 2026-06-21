import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

class BatikSemanticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi Semantic Batik - Distance Based")
        self.root.state('zoomed')
        
        # --- KONFIGURASI WARNA ---
        self.colors = {
            "bg_main": "#212121", "bg_panel": "#424242",
            "accent": "#00E676", "text": "#FFFFFF", 
            "btn_bg": "#616161", "btn_fg": "#00E676"
        }
        self.root.configure(bg=self.colors["bg_main"])

        # --- KONFIGURASI MODEL & DATA ---
        self.model_path = "best_model_semantic.h5"  # Ganti dengan nama model semantic Anda
        
        # Ganti path ini ke folder dataset 'train' Anda di Laptop/PC
        # Struktur folder harus: database/NamaKelas/foto.jpg
        self.database_path = "dataset/train" 
        
        self.img_size = (224, 224)
        self.prototypes = {} # Menyimpan ingatan vektor rata-rata per kelas
        self.model = None
        self.cap = None
        self.is_running = False

        # UI Setup
        self.setup_ui()
        
        # Load Model & Database Otomatis
        self.root.after(100, self.init_system)

    def init_system(self):
        """Memuat model dan membangun database referensi"""
        # 1. Load Model
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model '{self.model_path}' tidak ditemukan!")
            self.status_lbl.config(text="Model Hilang!", fg="red")
            return

        try:
            self.status_lbl.config(text="Memuat Model Embedding...", fg="yellow")
            self.root.update()
            
            # PERBAIKAN DI SINI: Tambahkan safe_mode=False
            # Ini diperlukan karena kita pakai Lambda layer
            self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)
            print("✅ Model loaded.")
            
            # 2. Bangun Database Referensi
            if not os.path.exists(self.database_path):
                messagebox.showwarning("Warning", f"Folder Database '{self.database_path}' tidak ditemukan.\nSilakan set path di kodingan.")
                self.status_lbl.config(text="Database Tidak Ditemukan", fg="red")
                return

            self.status_lbl.config(text="Membangun Ingatan (Prototypes)...", fg="yellow")
            self.root.update()
            self.build_reference_prototypes()
            
            self.status_lbl.config(text=f"Siap! ({len(self.prototypes)} Kelas Terdaftar)", fg=self.colors["accent"])

        except Exception as e:
            print(f"Error Init: {e}")
            messagebox.showerror("Critical Error", str(e))

    def build_reference_prototypes(self):
        """Membaca folder gambar dan mengubahnya menjadi vektor referensi"""
        self.prototypes = {}
        classes = [d for d in os.listdir(self.database_path) if os.path.isdir(os.path.join(self.database_path, d))]
        
        total_imgs = 0
        
        for cls in classes:
            cls_path = os.path.join(self.database_path, cls)
            vectors = []
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            # Ambil maksimal 5 gambar per kelas untuk mempercepat loading (bisa diubah)
            for fname in files[:5]: 
                img_path = os.path.join(cls_path, fname)
                try:
                    # Preprocess sama persis seperti training
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    img = img.astype("float32")
                    img = preprocess_input(img)
                    img = np.expand_dims(img, axis=0)
                    
                    # Prediksi Vektor
                    vec = self.model.predict(img, verbose=0)
                    vectors.append(vec)
                    total_imgs += 1
                except: continue
            
            if vectors:
                # Hitung rata-rata vektor (Prototype) untuk kelas ini
                self.prototypes[cls] = np.mean(vectors, axis=0)
                print(f"   -> Kelas '{cls}' didaftarkan ({len(vectors)} sampel).")
        
        print(f"✅ Selesai. Total {total_imgs} gambar diproses.")

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg=self.colors["bg_panel"], pady=10)
        header.pack(fill="x")
        tk.Label(header, text="🔍 SEMANTIC BATIK RECOGNIZER", font=("Roboto", 24, "bold"), 
                 bg=self.colors["bg_panel"], fg=self.colors["accent"]).pack()
        self.status_lbl = tk.Label(header, text="Menunggu inisialisasi...", font=("Consolas", 11), 
                                   bg=self.colors["bg_panel"], fg="white")
        self.status_lbl.pack()

        # Main Canvas
        self.canvas_frame = tk.Frame(self.root, bg=self.colors["bg_main"])
        self.canvas_frame.pack(expand=True, fill="both", padx=20, pady=10)
        self.display_lbl = tk.Label(self.canvas_frame, bg="black", bd=2, relief="sunken")
        self.display_lbl.pack(expand=True)

        # Footer Controls
        footer = tk.Frame(self.root, bg=self.colors["bg_panel"], pady=15)
        footer.pack(fill="x", side="bottom")
        
        btn_cfg = {"font": ("Arial", 12, "bold"), "bg": self.colors["btn_bg"], "fg": self.colors["btn_fg"], "width": 15}
        tk.Button(footer, text="📁 Upload File", command=self.upload_file, **btn_cfg).pack(side="left", padx=20, expand=True)
        tk.Button(footer, text="🎥 Webcam Live", command=self.start_webcam, **btn_cfg).pack(side="left", padx=20, expand=True)
        tk.Button(footer, text="❌ Stop", command=self.stop_stream, bg="#D32F2F", fg="white", font=("Arial", 12, "bold"), width=10).pack(side="left", padx=20, expand=True)

    def get_embedding(self, frame_roi):
        """Mengubah ROI gambar menjadi vektor"""
        try:
            img = cv2.resize(frame_roi, self.img_size)
            img = img.astype("float32")
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)
            return self.model.predict(img, verbose=0)
        except:
            return None

    def predict_semantic(self, frame):
        if not self.prototypes: return frame # Belum ada database

        h, w, _ = frame.shape
        # Kotak ROI di tengah
        box_len = int(min(h, w) * 0.6)
        x1, y1 = (w - box_len)//2, (h - box_len)//2
        x2, y2 = x1 + box_len, y1 + box_len
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size > 0:
            query_vec = self.get_embedding(roi)
            
            if query_vec is not None:
                # --- LOGIKA PENCARIAN JARAK TERDEKAT ---
                min_dist = float('inf')
                best_match = "Unknown"
                
                for cls_name, proto_vec in self.prototypes.items():
                    # Euclidean Distance
                    dist = np.linalg.norm(query_vec - proto_vec)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_match = cls_name
                
                # Visualisasi
                # Threshold Jarak: < 0.6 Mirip, > 0.6 Meragukan
                color = (0, 255, 0) if min_dist < 0.6 else (0, 165, 255) # Hijau vs Oranye
                
                # Gambar Kotak
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Teks Label & Jarak
                label_txt = best_match.replace("_", " ").replace("- CLEAN", "")
                
                # Background Teks
                cv2.rectangle(frame, (x1, y1-60), (x2, y1), color, -1)
                cv2.putText(frame, label_txt, (x1+10, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                cv2.putText(frame, f"Dist: {min_dist:.4f}", (x1+10, y1-10), cv2.FONT_HERSHEY_MONO, 0.6, (0,0,0), 1)

        return frame

    # --- FUNGSI KAMERA & DISPLAY (Sama seperti sebelumnya) ---
    def start_webcam(self):
        self.stop_stream()
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.update_frame()

    def upload_file(self):
        self.stop_stream()
        path = filedialog.askopenfilename()
        if path:
            frame = cv2.imread(path)
            if frame is not None:
                frame = self.predict_semantic(frame)
                self.show_image(frame)

    def update_frame(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame = self.predict_semantic(frame)
                self.show_image(frame)
                self.root.after(30, self.update_frame)

    def show_image(self, frame):
        # Resize agar fit di window
        h, w = frame.shape[:2]
        disp_h = 600
        disp_w = int(w * (disp_h/h))
        frame = cv2.resize(frame, (disp_w, disp_h))
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img))
        self.display_lbl.configure(image=img_tk)
        self.display_lbl.image = img_tk

    def stop_stream(self):
        self.is_running = False
        if self.cap: self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = BatikSemanticApp(root)
    root.mainloop()