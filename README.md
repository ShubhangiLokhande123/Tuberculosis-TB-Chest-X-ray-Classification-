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

> **Live Demo:** [https://huggingface.co/spaces/shubhlokhugging/TB-Chest-Xray-Classification](https://huggingface.co/spaces/shubhlokhugging/TB-Chest-Xray-Classification)

An AI-powered end-to-end pipeline for classifying chest X-rays as **Normal** or **Tuberculosis**, combining deep learning with explainable AI and an interactive web deployment. Built with **EfficientNet-B0** transfer learning, **Grad-CAM** visual explainability, and a **Gradio** web interface deployed on Hugging Face Spaces.

---

## 📋 Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Features](#features)
3. [Model Architecture & Comparison](#model-architecture--comparison)
4. [Dataset](#dataset)
5. [Training Pipeline](#training-pipeline)
6. [Explainability — Grad-CAM](#explainability--grad-cam)
7. [Diagnostic Report Generation](#diagnostic-report-generation)
8. [Web Interface & Deployment](#web-interface--deployment)
9. [Repository Structure](#repository-structure)
10. [Quick Start (Local)](#quick-start-local)
11. [Disclaimer](#disclaimer)

---

## 🔄 Pipeline Overview

```
Raw Chest X-Ray Image
        │
        ▼
┌─────────────────────────────┐
│  1. DATA PREPARATION        │
│  • Kaggle TB Radiography DB │
│  • 4,200 images (2 classes) │
│  • Stratified 70/15/15 split│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. PREPROCESSING &         │
│     AUGMENTATION            │
│  • Resize → 224×224         │
│  • Random Flip / Rotation   │
│  • Colour Jitter            │
│  • Affine Transform         │
│  • ImageNet Normalisation   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. MODEL TRAINING          │
│  • Custom CNN (baseline)    │
│  • ResNet50 (transfer)      │
│  • DenseNet121 (transfer)   │
│  • EfficientNet-B0 ✅ best  │
│  • ViT-B/16 (transformer)   │
│  • Differential LR + AdamW  │
│  • CosineAnnealingLR        │
│  • Early stopping (p=5)     │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. EVALUATION              │
│  • Accuracy / Loss curves   │
│  • Confusion Matrix         │
│  • ROC-AUC Curve            │
│  • Classification Report    │
│  • Per-class F1 / Precision │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  5. EXPLAINABILITY          │
│  • Grad-CAM (manual hooks)  │
│  • Heatmap on features[-1]  │
│  • Overlay on original X-ray│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. REPORT GENERATION       │
│  • DOCX diagnostic report   │
│  • Confidence table         │
│  • Side-by-side images      │
│  • Clinical recommendation  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  7. DEPLOYMENT              │
│  • Gradio Blocks web UI     │
│  • Hugging Face Spaces      │
│  • Interactive API endpoint │
└─────────────────────────────┘
```

---

## ✨ Features

| Feature | Detail |
|---|---|
| Multi-model comparison | Custom CNN · ResNet50 · DenseNet121 · EfficientNet-B0 · ViT-B/16 |
| Transfer learning | ImageNet pre-trained weights with differential learning rates |
| Data augmentation | Flip, rotation, colour jitter, affine transforms |
| Explainability | Grad-CAM heatmap via manual PyTorch forward/backward hooks |
| Report generation | Downloadable DOCX with images, confidence scores, clinical advice |
| Web interface | Gradio Blocks UI — upload, analyse, download in one click |
| Live deployment | Hosted on Hugging Face Spaces (no installation required) |
| Reproducibility | Fixed random seeds (42) across NumPy, PyTorch, CUDA |

---

## 🧠 Model Architecture & Comparison

### Selected Model: EfficientNet-B0

EfficientNet-B0 was chosen as the deployment model based on the best balance of accuracy, parameter efficiency, and inference speed.

```
Input Image (3 × 224 × 224)
        │
        ▼
EfficientNet-B0 Backbone
(MBConv blocks — ImageNet pre-trained)
        │
        ▼  features[-1]  ←── Grad-CAM target layer
        │
   AdaptiveAvgPool2d
        │
        ▼
   Dropout (p=0.5)
   Linear (1280 → 256)
   ReLU
   Dropout (p=0.3)
   Linear (256 → 2)
        │
        ▼
   Softmax → [P(Normal), P(TB)]
```

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 7 (early stopping patience = 5) |
| Optimiser | Adam |
| Learning rate (backbone) | 0.0001 |
| Learning rate (head) | 0.001 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |

### Benchmark Results (Internal Test Set — 15% hold-out)

| Model | Parameters | Test Accuracy | Test Loss | AUC-ROC |
|:---|:---:|:---:|:---:|:---:|
| Custom CNN (from scratch) | ~3.5M | 98.73% | 0.0315 | 0.9995 |
| ResNet50 | ~25M | 99.37% | 0.0163 | 1.0000 |
| DenseNet121 | ~8M | 99.37% | 0.0163 | 1.0000 |
| **EfficientNet-B0** ✅ | **~5M** | **99.52%** | **0.0125** | **1.0000** |
| Vision Transformer ViT-B/16 | ~86M | 99.68% | 0.0120 | 1.0000 |

> EfficientNet-B0 selected for deployment: near-top accuracy with the lowest parameter count among transfer learning models, making it the most practical for real-time inference.

---

## 📊 Dataset

**TB Chest Radiography Database** — [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

| Split | Images | Normal | Tuberculosis |
|---|---|---|---|
| Train (70%) | ~2,940 | ~2,450 | ~490 |
| Validation (15%) | ~630 | ~525 | ~105 |
| Test (15%) | ~630 | ~525 | ~105 |
| **Total** | **4,200** | **3,500** | **700** |

- Stratified splitting ensures equal class ratio across all splits
- No patient or image overlap between train / val / test sets
- Dataset is **not** included in this repository — download from Kaggle and place at `TB_Chest_Radiography_Database/`

---

## 🏋️ Training Pipeline

### Data Augmentation (Training only)

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])
```

### Training Features
- **Differential learning rates** — backbone LR × 0.1, head LR × 1.0 to preserve pre-trained features
- **Early stopping** — patience of 5 epochs to prevent overfitting
- **Best weights checkpoint** — restores the epoch with highest validation accuracy
- **CosineAnnealingLR** — smooth learning rate decay for stable convergence

---

## 🔥 Explainability — Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions of the chest X-ray most responsible for the model's prediction. Implemented using **manual PyTorch forward and backward hooks** (no external library — compatible with Python 3.13+).

```
Forward pass  →  capture activations at features[-1]
Backward pass →  capture gradients at features[-1]
                         │
                         ▼
         weights  =  global_avg_pool(gradients)
         CAM      =  ReLU( Σ weights × activations )
                         │
                         ▼
         Resize CAM → 224×224
         Apply COLORMAP_JET
         Blend 50% original + 50% heatmap
                         │
                         ▼
         PIL Image overlay (returned to UI)
```

---

## 📄 Diagnostic Report Generation

Each analysis produces a downloadable **Microsoft Word (.docx)** report containing:

1. **Title block** — "TB Chest X-Ray Analysis Report" with timestamp
2. **Classification result** — predicted class with colour-coded confidence table
3. **X-Ray images** — original chest X-ray and Grad-CAM overlay side-by-side
4. **Clinical recommendation** — conditional advisory text based on the prediction
5. **Disclaimer** — AI tool for research/educational use only

Reports are generated using `python-docx` with styled headings, tables, and formatted text.

---

## 🌐 Web Interface & Deployment

### Live Application

🚀 **[https://huggingface.co/spaces/shubhlokhugging/TB-Chest-Xray-Classification](https://huggingface.co/spaces/shubhlokhugging/TB-Chest-Xray-Classification)**

No installation required — open the link, upload a chest X-ray, and receive results instantly.

### Gradio Interface Layout

```
┌──────────────────────────────────────────────────────┐
│         🫁 TB Chest X-Ray Analysis System            │
│    EfficientNet-B0 · Grad-CAM · DOCX Report          │
├───────────────────┬──────────────────────────────────┤
│  Upload X-Ray     │  Original X-Ray │ Grad-CAM        │
│  [Image Input]    │  [Output]       │ Heatmap [Output]│
│                   │                 │                  │
│  [🔍 Analyze]     │                 │                  │
├───────────────────┴──────────────────────────────────┤
│  Diagnostic Report (Markdown)                        │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 🔴/🟢 Diagnosis: Normal / Tuberculosis          │ │
│  │ Confidence Table | Timestamp | Recommendation   │ │
│  └─────────────────────────────────────────────────┘ │
│  [📥 Download DOCX Report]                           │
└──────────────────────────────────────────────────────┘
```

### Gradio API Usage

The deployed Space exposes a REST API automatically via Gradio. You can call it programmatically:

```python
from gradio_client import Client

client = Client("shubhlokhugging/TB-Chest-Xray-Classification")
result = client.predict(
    image="path/to/chest_xray.png",
    api_name="/predict"
)
# Returns: (original_image, heatmap_image, report_markdown, docx_file_path)
print(result)
```

Install the client:
```bash
pip install gradio-client
```

### API via HTTP (cURL)

```bash
curl -X POST \
  https://shubhlokhugging-tb-chest-xray-classification.hf.space/run/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["data:image/png;base64,<BASE64_IMAGE>"]}'
```

---

## 📂 Repository Structure

```
TB_Chest_Xray_Classification_V5.ipynb   # Full training & evaluation notebook
app.py                                   # Standalone Gradio web application
requirements.txt                         # Python dependencies
README.md                                # Project documentation
.gitignore                               # Git ignore rules
models/
    efficientnet_b0_tb.pth              # Trained EfficientNet-B0 weights (16.8 MB)
TB_Chest_Radiography_Database/          # Dataset (NOT included — download from Kaggle)
    Normal/
    Tuberculosis/
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/ShubhangiLokhande123/Tuberculosis-TB-Chest-X-ray-Classification-.git
cd Tuberculosis-TB-Chest-X-ray-Classification-

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the web app
python app.py
# Open http://127.0.0.1:7860 in your browser
```

**Requirements:** Python 3.10+, PyTorch 2.0+

---

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**.  
It does **not** constitute a medical diagnosis and must **not** replace professional clinical judgment.  
Always consult a qualified healthcare professional for diagnosis and treatment decisions.

The model was trained and evaluated on the [TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) and is optimised for that specific image distribution. Performance on clinical or other external data sources may vary.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

*Built with PyTorch · Gradio · Hugging Face Spaces*
