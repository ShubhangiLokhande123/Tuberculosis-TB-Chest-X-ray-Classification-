---
title: TB Chest X-Ray Classification
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "6.12.0"
app_file: app.py
pinned: false
license: mit
---

# 🫁 Tuberculosis (TB) Chest X-Ray Classification

An AI-powered web application for classifying chest X-rays as **Normal** or **Tuberculosis**, built with **EfficientNet-B0**, **Grad-CAM explainability**, and a **Gradio** interactive interface.

## 🔍 Features

- **EfficientNet-B0** fine-tuned on the TB Chest Radiography Dataset
- **Grad-CAM heatmap** — visualises which regions of the X-ray influenced the prediction
- **DOCX diagnostic report** — downloadable report with images, confidence scores, and clinical recommendation
- **Gradio web UI** — runs locally and on Hugging Face Spaces

## 🧠 Model Architecture

| Component | Detail |
|---|---|
| Base model | EfficientNet-B0 (ImageNet pre-trained) |
| Custom head | Dropout(0.5) → Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→2) |
| Input size | 224 × 224 |
| Classes | Normal · Tuberculosis |
| Optimiser | Adam (lr=0.001, weight decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Epochs | 7 |

## 📂 Repository Structure

```
TB_Chest_Xray_Classification_V5.ipynb   # Full training notebook
app.py                                   # Standalone Gradio web app
requirements.txt                         # Python dependencies
models/
    efficientnet_b0_tb.pth              # Trained model weights
```

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/ShubhangiLokhande123/Tuberculosis-TB-Chest-X-ray-Classification-.git
cd Tuberculosis-TB-Chest-X-ray-Classification-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the web app
python app.py
# Open http://127.0.0.1:7860 in your browser
```

## 📊 Dataset

[TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) — 700 TB + 3500 Normal chest X-ray images.
Dataset is **not** included in this repository. Download from Kaggle and place it at `TB_Chest_Radiography_Database/`.

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.  
It does **not** constitute a medical diagnosis and must not replace professional clinical judgment.  
Always consult a qualified healthcare professional.

## 📄 License

MIT License — see [LICENSE](LICENSE) file.
