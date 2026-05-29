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

# 🚀 Live Demo (Google Colab)

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
| Epochs          | 20                   |
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

---

# 🧠 Training Script

```python
import os
import sys

# --- Step 1: Setup & Install ---
try:
    from ultralytics import YOLO
    import yaml

except ImportError:
    print("Installing required libraries...")

    os.system(
        f'{sys.executable} -m pip install ultralytics pyyaml'
    )

    from ultralytics import YOLO
    import yaml


# --- Step 2: Bulletproof Path Fix ---
dataset_dir = os.path.abspath("dataset")

print(f"Targeting dataset at: {dataset_dir}")

yaml_path = os.path.join(dataset_dir, "data.yaml")

if not os.path.exists(yaml_path):
    raise FileNotFoundError(
        f"Could not find data.yaml in {dataset_dir}"
    )

print("\nConfiguring data.yaml paths...")

with open(yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)

# Remove relative path variable
if 'path' in yaml_data:
    del yaml_data['path']

# Absolute training path
yaml_data['train'] = os.path.join(
    dataset_dir,
    "images",
    "train"
)

# Validation folder handling
val_path = os.path.join(
    dataset_dir,
    "images",
    "val"
)

valid_path = os.path.join(
    dataset_dir,
    "images",
    "valid"
)

if os.path.exists(val_path):
    yaml_data['val'] = val_path

elif os.path.exists(valid_path):
    yaml_data['val'] = valid_path

else:
    print("Validation folder not found.")
    yaml_data['val'] = yaml_data['train']

# Remove test path
if 'test' in yaml_data:
    del yaml_data['test']

# Save corrected YAML
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"Train Path: {yaml_data['train']}")
print(f"Validation Path: {yaml_data['val']}")

# --- Step 3: Training ---
print("\nStarting YOLOv8 Fine-Tuning...")

model = YOLO("bbox.pt")

results = model.train(
    data=yaml_path,
    epochs=20,
    imgsz=640,
    batch=64,
    workers=8,
    device=0,
    patience=10,
    optimizer="AdamW",
    lr0=0.01,
    cos_lr=True,
    amp=True,
    project="finetune",
    name="yolov8n_20_epochs"
)

print("\n✅ Training completed successfully!")

print(
    "Best weights saved at:\n"
    "finetune/yolov8n_20_epochs/weights/best.pt"
)
```

---

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
<img src="https://github.com/user-attachments/assets/ORIGINAL_IMAGE_ID" width="350"/>
</td>

<td>
<img src="https://github.com/user-attachments/assets/ANNOTATED_IMAGE_ID" width="350"/>
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

---

# 🔍 Inference

Run inference using your trained weights:

```bash
python test_model.py
```

---

# 🧪 Example Inference Script

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True
)

print("Inference completed.")
```

---

# ⚡ Performance Highlights

* Real-time inference capable
* High accuracy on layered garments
* Strong ethnic wear detection
* Optimized for production deployment
* Supports fashion API pipelines

---

# 📦 Requirements

```txt
ultralytics
pyyaml
opencv-python
requests
torch
torchvision
```

---

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

---

# 📜 License

This project is released under the MIT License.

---

# ⭐ Acknowledgements

* Ultralytics YOLOv8
* Fashionpedia Dataset
* PyTorch
* OpenCV
