# 🪕 Real-Time Batik Motif Classification Desktop Application using MobileNetV2

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)](https://tensorflow.org)
[![GUI](https://img.shields.io/badge/GUI-Tkinter-green)](https://docs.python.org/3/library/tkinter.html)
[![License](https://img.shields.io/badge/License-CC--BY--SA-lightgrey)](https://creativecommons.org/licenses/by-sa/4.0/)

A desktop application based on Deep Learning for real-time Indonesian batik motif classification. The system supports both webcam-based recognition and manual image upload, utilizing Transfer Learning with the MobileNetV2 architecture to provide fast and efficient inference on devices with limited computational resources.

This project was developed as part of a research study at **Dian Nuswantoro University**, Faculty of Computer Science, Informatics Engineering Program, with the aim of supporting the digitalization, automatic identification, and preservation of Indonesia's cultural heritage.

---

## 🚀 Features

* **Real-Time Classification via Webcam**
  Detect and classify batik motifs continuously using a webcam.

* **Manual Image Upload**
  Upload `.png`, `.jpg`, or `.jpeg` images for instant prediction.

* **Interactive Desktop GUI**
  Lightweight and user-friendly interface developed with Tkinter.

* **Performance Visualization**
  Display model evaluation metrics including Confusion Matrix and Classification Report.

---

## 📊 Dataset

The dataset consists of four Indonesian batik motif categories collected from Kaggle.

| Class                  | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| **Pintu Aceh**         | Symmetrical geometric patterns characteristic of Aceh     |
| **Madura Mataketeran** | Repetitive floral and curved patterns with vibrant colors |
| **Maluku Pala**        | Motifs inspired by Maluku's spice heritage                |
| **Solo Parang**        | Diagonal patterns characteristic of Central Java          |

The initial dataset contains **72 images**, which are distributed into training and testing sets with balanced class proportions.

---

## 🧠 Methodology

### Image Preprocessing and Data Augmentation

* Image size normalization to **224 × 224 × 3** pixels.
* Data augmentation techniques:

  * Horizontal Flip
  * Rotation (30°)
  * Zoom Range (0.3)
  * Width Shift (0.3)
  * Height Shift (0.3)
  * Shear Range (0.3)

These techniques help reduce overfitting and improve model generalization.

### Model Architecture

This project employs **Transfer Learning** using **MobileNetV2**, which utilizes:

* Depthwise Separable Convolution
* Inverted Residual Blocks

The model is compiled using:

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy

---

## 📈 Model Performance

The best model performance was achieved at **Epoch 7**.

| Metric              |                    Value |
| ------------------- | -----------------------: |
| Validation Accuracy |                      94% |
| Validation Loss     |                   0.4032 |
| Input Size          |            224 × 224 × 3 |
| Architecture        |              MobileNetV2 |
| Optimizer           |                     Adam |
| Loss Function       | Categorical Crossentropy |

### Application Testing Results

| Batik Motif        | Static Image Accuracy | Real-Time Webcam Accuracy |
| ------------------ | :-------------------: | :-----------------------: |
| Solo Parang        |       96% – 98%       |         92% – 95%         |
| Pintu Aceh         |       95% – 97%       |            90%            |
| Maluku Pala        |       93% – 94%       |         90% – 92%         |
| Madura Mataketeran |       90% – 92%       |         89% – 91%         |

**Conclusion:**
Static image testing generally produces higher accuracy because the image quality is digitally controlled. Nevertheless, real-time webcam classification remains robust, maintaining accuracy above 89% despite variations in lighting conditions and environmental noise.

---

## 📁 Project Structure

```text
📦 batik-classification-desktop
├── models/
│   └── model_batik.h5
├── app/
│   ├── app_gui.py
│   └── utils.py
├── docs/
│   └── Jurnal_Batik.docx
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Install Dependencies

Make sure Python 3.8 or higher is installed.

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the desktop application:

```bash
python app/app_gui.py
```

---

## 📦 Requirements

* Python 3.8+
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pillow
* Tkinter

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🎓 Research Context

This work was conducted at:

**Faculty of Computer Science**
**Informatics Engineering Program**
**Dian Nuswantoro University**

The project aims to contribute to the preservation and digitalization of Indonesian cultural heritage through computer vision and deep learning technologies.

---

## 📜 License

This project is distributed under the **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** License.
