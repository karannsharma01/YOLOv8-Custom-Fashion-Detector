# YOLOv8-Custom-Fashion-Detector
An end-to-end YOLOv8 fine-tuning pipeline expanding the Fashionpedia dataset to 54 classes, including custom Indian ethnic wear (Sarees, Kurtas etc.) for virtual try-on engines.
# 👕 YOLOv8 Custom Fashion & Ethnic Wear Detector

![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)

This repository contains an end-to-end fine-tuning pipeline for **YOLOv8**, designed to detect and extract clothing items for virtual try-on fashion engines. 

It utilizes transfer learning, starting with a base model trained on the 46-class Fashionpedia dataset, and fine-tunes it to recognize 8 additional custom categories (including Indian ethnic wear like Sarees, Kurtas, and Co-ord sets), bringing the total to **54 distinct fashion classes**.

## ✨ Key Features
* **Transfer Learning:** Extends an existing 46-class fashion model to 54 classes without catastrophic forgetting.
* **Custom Categories:** Specifically trained to accurately draw bounding boxes around complex garments like `saree`, `kurta`, and `co-ord_set`.
* **URL Inference:** Includes a production-ready test script that securely downloads images from URLs, runs inference, and extracts exact bounding box coordinates.
* **Hardware Optimized:** Training pipeline configured for high VRAM environments (like L4 GPUs) using the `AdamW` optimizer and mixed precision (`amp=True`).

---

## 📊 The Dataset (54 Classes)
The model detects standard western wear (shirts, pants, jackets, dresses) alongside newly integrated custom categories:
* `46: saree`
* `47: co-ord_set`
* `48: kurta`
* `49: hoodie`
* `50: swimsuit`
* `51: heels`
* `52: flats`
* `53: tracksuit`

---

## 🚀 Training Setup & Hyperparameters

The model was trained using the `ultralytics` framework on an L4 GPU. To prioritize learning the new categories quickly while maintaining base knowledge, a **20-epoch** fine-tuning approach was used.

**Hyperparameters:**
* **Epochs:** 20
* **Image Size:** 640x640
* **Batch Size:** 64 (Optimized for 24GB VRAM)
* **Optimizer:** AdamW (`lr0=0.01`, Cosine Annealing enabled)
* **Device:** `cuda:0`

### Run Training
Ensure your dataset is perfectly formatted in the `dataset/` folder, then execute:
```bash
python train.py
📈 Training Results
After 20 epochs, the model successfully stabilized its validation metrics across both the base Fashionpedia classes and the newly introduced custom apparel.

Training Metrics (mAP & Loss)
Below is the validation curve showing the model's accuracy improvement over the 20 epochs.

(Note: You can find this graph in your runs/detect/fashion_finetune/yolov8n_20_epochs/results.png file!)

🔍 Inference & Testing
To test the model on a new image from the internet, update the MODEL_PATH in the test script to point to your best.pt weights and run:

Bash
python test_model.py
Example Output
The script outputs precise bounding box pixel coordinates and generates a visualized .jpg copy.

Plaintext
👕 Label: SAREE
📊 Confidence: 87.4%
📍 Bounding Box (Pixels): Top-Left(210, 150) to Bottom-Right(680, 920)
Visual Verification:

📦 Requirements
To run the scripts in this repository, install the dependencies:

Bash
pip install ultralytics pyyaml requests
