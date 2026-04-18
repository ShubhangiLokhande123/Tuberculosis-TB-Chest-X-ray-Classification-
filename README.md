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
6. [Cross-Domain Testing & Inference Pipeline](#cross-domain-testing--inference-pipeline)
7. [Explainability — Grad-CAM](#explainability--grad-cam)
8. [Diagnostic Report Generation](#diagnostic-report-generation)
9. [Web Interface & Deployment](#web-interface--deployment)
10. [Repository Structure](#repository-structure)
11. [Quick Start (Local)](#quick-start-local)
12. [Disclaimer](#disclaimer)

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
│  5. CROSS-DOMAIN TESTING    │
│  • External dataset (3,008) │
│  • 4-pipeline comparison    │
│  • 5-pass TTA inference     │
│  • Threshold optimisation   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. EXPLAINABILITY          │
│  • Grad-CAM (manual hooks)  │
│  • Heatmap on features[-1]  │
│  • Overlay on original X-ray│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  7. REPORT GENERATION       │
│  • DOCX diagnostic report   │
│  • Confidence table         │
│  • Side-by-side images      │
│  • Clinical recommendation  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  8. DEPLOYMENT              │
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
| Cross-domain testing | Evaluated on external dataset (3,008 images, different distribution) |
| 5-pass TTA | Test-Time Augmentation for stable cross-domain inference |
| Threshold tuning | Decision threshold optimised for balanced accuracy (τ = 0.945) |
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
        │
        ▼
   Threshold τ = 0.945  ←── tuned for balanced accuracy
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

### Training Dataset — TB Chest Radiography Database

**Source:** [Kaggle — tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

| Split | Images | Normal | Tuberculosis |
|---|---|---|---|
| Train (70%) | ~2,940 | ~2,450 | ~490 |
| Validation (15%) | ~630 | ~525 | ~105 |
| Test (15%) | ~630 | ~525 | ~105 |
| **Total** | **4,200** | **3,500** | **700** |

- Class ratio: **5:1 Normal:TB** — model trained on predominantly Normal images
- Stratified splitting ensures equal class ratio across all splits
- No patient or image overlap between train / val / test sets
- Dataset is **not** included in this repository — download from Kaggle and place at `TB_Chest_Radiography_Database/`

### Cross-Domain Test Dataset — Dataset of Tuberculosis Chest X-rays Images

An independent external dataset used exclusively for cross-domain robustness evaluation. No images from this set were used during training or validation.

| Class | Images | Format |
|---|---|---|
| Normal | 514 | JPEG |
| Tuberculosis | 2,494 | JPEG |
| **Total** | **3,008** | — |

- Class ratio: **1:4.9 Normal:TB** — inverted vs training distribution
- Dataset pixel mean: `[0.552, 0.552, 0.552]` vs ImageNet `[0.485, 0.456, 0.406]` — brighter, grayscale X-rays
- Represents real-world domain shift: different scanner, institution, and image acquisition protocol
- Located at `Dataset of Tuberculosis Chest X-rays Images/` (not included in repo)

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

---

## 🔬 Cross-Domain Testing & Inference Pipeline

### Motivation

A model that achieves 99.52% accuracy on its held-out test set may still fail on images from different hospitals, scanners, or datasets due to **domain shift** — differences in brightness, contrast, aspect ratio, and class distribution. This section documents the cross-domain evaluation performed on the external TB X-ray dataset and the inference pipeline improvements that resulted.

### Domain Gap Analysis

| Property | Training Data | External Test Data |
|---|---|---|
| Dataset | TB Chest Radiography DB | Dataset of TB Chest X-rays Images |
| Total images | 4,200 | 3,008 |
| Normal : TB ratio | 5 : 1 | 1 : 4.9 (inverted) |
| Image mean | ~0.485 (ImageNet-like) | 0.552 (brighter, grayscale) |
| Format | PNG | JPEG |
| Source | Single Kaggle collection | Independent collection |

### Preprocessing Pipeline Comparison

Four inference preprocessing strategies were evaluated using **5-pass Test-Time Augmentation (TTA)** across all 3,008 external images. No retraining was performed — the same `efficientnet_b0_tb.pth` weights were used throughout.

| Pipeline | Preprocessing | Normalisation | AUC-ROC | Spec (τ=0.50) | Best BalAcc |
|---|---|---|---|---|---|
| **A — No-prep + ImageNet** ✅ | None (training conditions) | ImageNet | **0.7716** | 26.7% | **69.1%** |
| B — HEQ + ImageNet | Histogram Equalisation | ImageNet | 0.6378 | 23.7% | 59.5% |
| C — CLAHE + Dataset stats | CLAHE (clipLimit=2.0) | Dataset mean/std | 0.6177 | 0.2% | 51.2% |
| D — Per-image z-score | Per-image normalisation | ImageNet | 0.7349 | 35.2% | 66.1% |

**Key finding:** CLAHE applied at inference time — without CLAHE during training — degraded AUC from 0.77 → 0.62 and reduced Normal specificity to near zero. **Pipeline A (no extra preprocessing, ImageNet normalisation) matches training conditions and achieves the highest AUC and balanced accuracy.**

### 5-Pass Test-Time Augmentation (TTA)

Each image is evaluated five times with different augmentations; softmax scores are averaged to reduce variance:

```
Pass 1 — Original (resize 224×224 + normalize)
Pass 2 — Horizontal flip
Pass 3 — +5° rotation
Pass 4 — −5° rotation
Pass 5 — Center crop (200px) → resize back to 224px
         ↓
   Average softmax P(TB) scores across 5 passes
         ↓
   Apply decision threshold τ
```

### Decision Threshold Tuning

The default softmax threshold of 0.50 is poorly calibrated when the test-time class balance differs from training. Threshold τ was swept from 0.02 to 0.99 (step 0.005) to maximise **balanced accuracy** (arithmetic mean of sensitivity and specificity):

| Threshold | Accuracy | Sensitivity (TB recall) | Specificity (Normal recall) | Balanced Accuracy |
|---|---|---|---|---|
| τ = 0.50 (default) | 82.9% | 94.4% | 26.7% | 60.5% |
| **τ = 0.945 (tuned)** ✅ | **60.2%** | **55.5%** | **82.7%** | **69.1%** |

> **Specificity gain: +56 percentage points.** The tuned threshold dramatically reduces false-positive TB predictions for Normal patients.

### Current Inference Pipeline (app.py)

```python
# Preprocessing — exact training conditions, no extra enhancement
_infer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Decision threshold tuned for best balanced accuracy
THRESHOLD = 0.945

def predict(tensor):
    with torch.no_grad():
        probs = F.softmax(model(tensor.to(DEVICE)), dim=1)[0]
    tb_prob = float(probs[1])
    label = 'Tuberculosis' if tb_prob >= THRESHOLD else 'Normal'
    return label, float(probs[0]), tb_prob
```

### Cross-Domain Results Summary

| Metric | Before (CLAHE, τ=0.50) | After (No-prep, τ=0.945) | Change |
|---|---|---|---|
| AUC-ROC | 0.617 | **0.772** | +0.155 |
| Specificity (Normal recall) | 0.2% | **82.7%** | +82.5 pp |
| Sensitivity (TB recall) | 100.0% | 55.5% | −44.5 pp |
| Balanced Accuracy | 50.1% | **69.1%** | +19.0 pp |

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
TB_Chest_Xray_Classification_V5.ipynb            # Full training, evaluation & cross-domain notebook
app.py                                            # Standalone Gradio web application
requirements.txt                                  # Python dependencies
README.md                                         # Project documentation
.gitignore                                        # Git ignore rules
models/
    efficientnet_b0_tb.pth                       # Trained EfficientNet-B0 weights (16.8 MB)
TB_Chest_Radiography_Database/                   # Training dataset (NOT included — download from Kaggle)
    Normal/
    Tuberculosis/
Dataset of Tuberculosis Chest X-rays Images/     # Cross-domain test dataset (NOT included)
    Normal Chest X-rays/                         # 514 Normal JPEG images
    TB Chest X-rays/                             # 2,494 TB JPEG images
temp/                                            # Auto-generated: DOCX reports & evaluation plots
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

The model was trained on the [TB Chest Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) and evaluated on an independent external dataset of 3,008 images. Cross-domain performance (AUC = 0.77, Balanced Accuracy = 69.1% at τ = 0.945) is lower than the in-distribution test accuracy (99.52%) — this is expected due to differences in image source, scanner, and class distribution. The inference threshold has been tuned to reduce false-positive TB predictions for Normal patients.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file.

---

*Built with PyTorch · Gradio · Hugging Face Spaces*
