# 👕 YOLOv8 Custom Fashion & Ethnic Wear Detector

![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![GPU](https://img.shields.io/badge/GPU-L4%2024GB-green)

An end-to-end **YOLOv8 fine-tuning pipeline** for fashion garment detection, extending the original Fashionpedia dataset with custom Indian ethnic wear categories.

This repository trains a custom `YOLOv8n` model capable of detecting **55 fashion classes (0–54)**, including:

* Sarees
* Kurtas
* Co-ord Sets
* Tracksuits
* Suits
* Heels
* Flats
* Hoodies
* Swimwear

The model is specifically designed for:

* 👗 Virtual Try-On Engines
* 🧥 Fashion Recommendation Systems
* 🛍️ E-commerce AI
* 📦 Garment Extraction Pipelines
* 🎯 Fashion Detection APIs

---

# ✨ Key Features

* 🔥 Fine-tuned YOLOv8n on an extended Fashionpedia dataset
* 👘 Added Indian ethnic wear categories
* ⚡ Optimized for L4 GPUs (24GB VRAM)
* 📦 Transfer learning from pre-trained fashion weights
* 🎯 High-speed real-time fashion detection
* 🌐 URL-based inference support
* 📊 Automatic visualization generation
* 🧠 Mixed precision (`AMP`) training enabled

---

# 📊 Dataset Overview

The original Fashionpedia dataset was expanded with additional custom classes for Indian and modern fashion garments.

## 🏷️ Custom Added Classes

| Class ID | Label      |
| -------- | ---------- |
| 46       | saree      |
| 47       | co-ord_set |
| 48       | kurta      |
| 49       | hoodie     |
| 50       | swimsuit   |
| 51       | heels      |
| 52       | flats      |
| 53       | tracksuit  |
| 54       | suit       |

Total Classes: **55**

---

# 🚀 Demo (Google Colab)

Try the trained model instantly in Google Colab without local setup.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

The demo notebook includes:

* ✅ Model loading
* ✅ URL image inference
* ✅ Bounding box visualization
* ✅ Confidence score extraction
* ✅ Fashion garment detection
* ✅ Automatic annotated image generation

---

# 🧠 Model Architecture

```text
YOLOv8n Backbone
       ↓
Transfer Learning
       ↓
Fashionpedia Base Model
       ↓
Custom Fine-Tuning
       ↓
55-Class Fashion Detector
```

---

# 🚀 Training Configuration

The model was fine-tuned using the Ultralytics YOLOv8 framework on an NVIDIA L4 GPU.

## ⚙️ Hyperparameters

| Parameter       | Value                |
| --------------- | -------------------- |
| Model           | YOLOv8n              |
| Epochs          | 10                   |
| Image Size      | 640                  |
| Batch Size      | 64                   |
| Optimizer       | AdamW                |
| Initial LR      | 0.01                 |
| Scheduler       | Cosine Annealing     |
| Mixed Precision | Enabled (`amp=True`) |
| GPU             | NVIDIA L4 24GB       |

---

# 📁 Project Structure

```text
YOLOv8-Custom-Fashion-Detector/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   │
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   │
│   └── data.yaml
│
├── train.py
├── test_model.py
├── bbox.pt
├── README.md
```

---

# 🚀 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/YOLOv8-Custom-Fashion-Detector.git

cd YOLOv8-Custom-Fashion-Detector
```

---

## 2️⃣ Install Dependencies

```bash
pip install ultralytics pyyaml requests opencv-python
```

---

# 🏋️ Training

Run the training script:

```bash
python train.py
```

# 📈 Training Results

The model successfully learned both the original Fashionpedia categories and the newly added ethnic wear classes.

## 🖼️ Detection Results

<table>
<tr>
<td align="center"><b>Original Image</b></td>
<td align="center"><b>YOLOv8 Detection Result</b></td>
</tr>

<tr>
<td>
<img src="https://github.com/user-attachments/assets/344b6bb6-d712-442d-b487-c6a217e185f0" width="350"/>
</td>

<td>
<img src="https://github.com/user-attachments/assets/f66dcca1-e82e-4605-af79-01b9ba3a4ffa" width="350"/>
</td>
</tr>
</table>

---

# 📄 Example Detection Output

```text
👘 Label: SAREE
📊 Confidence: 87.4%

📍 Bounding Box:
Top-Left     : (210, 150)
Bottom-Right : (680, 920)
```

# 💡 Use Cases

* 👗 Virtual Try-On Systems
* 🛒 Fashion E-commerce
* 🧥 Outfit Detection Engines
* 📸 Fashion Image Tagging
* 🧠 AI Stylist Systems
* 📦 Garment Cropping Pipelines

---

# 🔮 Future Improvements

* Segmentation Support
* Fashion Attribute Classification
* DensePose Integration
* TensorRT Optimization
* ONNX Export
* Streamlit Demo UI
* Real-Time Webcam Detection


# ⭐ Acknowledgements

* Ultralytics YOLOv8
* Fashionpedia Dataset
* PyTorch
* OpenCV
