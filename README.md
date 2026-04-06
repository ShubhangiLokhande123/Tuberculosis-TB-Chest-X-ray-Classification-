# 🫁 Tuberculosis (TB) Chest X-ray Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)
![Models](https://img.shields.io/badge/Models-5-orange)
![Best AUC](https://img.shields.io/badge/Best%20AUC--ROC-0.999-brightgreen)

**Comparing Training from Scratch vs Transfer Learning for automated TB detection in chest radiographs**

*Custom CNN · ResNet50 · DenseNet121 · EfficientNet-B0 · Vision Transformer (ViT-B/16)*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Clinical Motivation](#-clinical-motivation)
- [Dataset](#-dataset)
- [Models](#-models)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Conclusions & Recommendations](#-conclusions--recommendations)
- [References](#-references)

---

## 🔍 Overview

This project tackles **binary classification of chest X-rays** — distinguishing *Normal* lungs from those showing signs of *Tuberculosis (TB)*. Five deep learning models are trained and rigorously compared:

| Approach | Model | Description |
|---|---|---|
| From Scratch | **Custom CNN** | Baseline 4-block ConvNet with no pre-trained weights |
| Transfer Learning | **ResNet50** | 50-layer residual network pre-trained on ImageNet |
| Transfer Learning | **DenseNet121** | Dense-connection network (basis of CheXNet) |
| Transfer Learning | **EfficientNet-B0** | Compound-scaled efficient CNN |
| Transfer Learning | **ViT-B/16** ⭐ | Vision Transformer with patch-based self-attention |

The study highlights the significant performance gap between training from scratch and leveraging ImageNet pre-training, and demonstrates that **Vision Transformers outperform all CNN-based architectures** on this medical imaging task.

---

## 🏥 Clinical Motivation

Tuberculosis remains one of the **top 10 causes of death worldwide** (WHO). Early, accurate detection via chest X-ray is critical, especially in resource-limited settings where radiologist access is scarce.

> ⚠️ **Key clinical metric: TB Recall (Sensitivity)**  
> A *missed TB case* (false negative) poses far greater public health risk than a false alarm. All models are therefore evaluated with particular attention to **sensitivity for the TB class**.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle – Tuberculosis (TB) Chest X-ray Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) |
| **Classes** | Normal · Tuberculosis |
| **Normal images** | ~3,500 |
| **TB images** | ~700 |
| **Total images** | ~4,200 |
| **Image format** | PNG / JPG |
| **Input size** | 224 × 224 px |

> ⚠️ **Class Imbalance**: The dataset has a ~5:1 Normal-to-TB ratio. This is addressed through weighted loss functions and careful attention to per-class recall.

### Data Splits

| Split | Ratio | Purpose |
|---|---|---|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning & early stopping |
| Test | 15% | Final unbiased evaluation |

### Data Augmentation (Training Only)

```
RandomHorizontalFlip(p=0.5)
RandomRotation(±10°)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
RandomAffine(translate=5%)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## 🧠 Models

### 1. Custom CNN (Baseline — From Scratch)

A custom 4-block ConvNet designed and trained **without any pre-trained weights**:

```
Input (3 × 224 × 224)
    ↓
ConvBlock(3→32)    → 112 × 112
ConvBlock(32→64)   →  56 × 56
ConvBlock(64→128)  →  28 × 28
ConvBlock(128→256) →  14 × 14
    ↓
GlobalAvgPool → 256 × 1 × 1
    ↓
FC(256→512) → BN → ReLU → Dropout(0.5)
FC(512→256) → BN → ReLU → Dropout(0.25)
FC(256→2)   → Output
```

Each ConvBlock: `Conv2D → BatchNorm → ReLU → MaxPool`  
Weight initialisation: Kaiming (He) uniform

---

### 2. ResNet50 (Transfer Learning)

- **~25M parameters** | 50-layer residual network
- Pre-trained on ImageNet; final FC layer replaced for 2-class output
- **Skip connections** prevent vanishing gradients
- **Differential learning rates**: backbone LR × 0.1, head LR × 1.0

---

### 3. DenseNet121 (Transfer Learning)

- **~8M parameters** | 121-layer dense network
- Direct inspiration for CheXNet (chest pathology detection)
- Dense feature reuse makes it particularly well-suited to medical imaging
- Most parameter-efficient of the CNN-based models

---

### 4. EfficientNet-B0 (Transfer Learning)

- **~5M parameters** | Compound-scaled architecture
- Scales depth, width, and resolution simultaneously for optimal efficiency
- Best accuracy-per-parameter ratio among CNN-based architectures
- Fastest inference time of the transfer learning models

---

### 5. Vision Transformer ViT-B/16 ⭐ (Transfer Learning)

Based on *"An Image is Worth 16×16 Words"* (Dosovitskiy et al., 2020):

1. **Patch Embedding** — 224×224 image split into 196 patches of 16×16
2. **Positional Encoding** — learnable positional embeddings per patch
3. **CLS Token** — prepended learnable token used for classification
4. **Transformer Encoder** — 12 layers × 12 attention heads
5. **Classification Head** — linear layer on `[CLS]` token output

**~86M parameters** | `torchvision.models.vit_b_16`

> **Why ViT for chest X-rays?** Unlike CNNs that build global context hierarchically layer-by-layer, ViT captures **global long-range dependencies from the very first layer** — essential for interpreting whole-lung structural patterns in radiographs.

---

## 📈 Results

### Model Performance Summary

| Model | Test Accuracy | AUC-ROC | Macro F1 | TB Recall |
|---|---|---|---|---|
| Custom CNN (Scratch) | ~88.0% | 0.928 | — | — |
| ResNet50 | ~97.0% | 0.993 | — | — |
| DenseNet121 | ~98.0% | 0.996 | — | — |
| EfficientNet-B0 | ~98.0% | 0.997 | — | — |
| **ViT-B/16** ⭐ | **~98.6%** | **0.999** | **—** | **~96%** |

### Training from Scratch vs Transfer Learning

| Aspect | From Scratch | Transfer Learning |
|---|---|---|
| Accuracy | ~88% | ~97–99% |
| Convergence speed | Slow (20 epochs) | Fast (10–15 epochs) |
| Data requirement | High | Low-to-moderate |
| Feature quality | Task-specific only | General + fine-tuned |
| Best suited for | Large custom datasets | Small/medium datasets |

### CNN vs Vision Transformer

| Aspect | CNN (ResNet/DenseNet/EfficientNet) | ViT-B/16 |
|---|---|---|
| Receptive field | Local → global (hierarchical) | Global from layer 1 |
| Inductive bias | Strong (translation equivariance) | Minimal |
| Training data need | Lower | Higher (covered by ImageNet pre-training) |
| Interpretability | Grad-CAM | Attention Rollout |
| Speed | Faster inference | Slower, higher memory |

---

## 📁 Project Structure

```
Tuberculosis-TB-Chest-X-ray-Classification-/
│
├── TB_Chest_Xray_Classification.ipynb   # Main notebook (all sections)
├── README.md
└── LICENSE
```

The notebook is organised into four sections:

| Section | Content |
|---|---|
| Section 1 | Custom CNN — architecture, training, evaluation |
| Section 2 | Transfer Learning — ResNet50, DenseNet121, EfficientNet-B0 |
| Section 3 | Vision Transformer ViT-B/16 |
| Section 4 | Cross-model comparison, ROC curves, confusion matrices |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; ViT-B/16 requires significant GPU memory)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install matplotlib seaborn scikit-learn pillow pandas numpy tqdm
pip install kaggle
```

### Download the Dataset

**Option A – Kaggle API**

```bash
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
unzip tuberculosis-tb-chest-xray-dataset.zip
```

**Option B – Manual Download**  
Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) and download `TB_Chest_Radiography_Database.zip`.

### Configure Dataset Path

Update `DATA_DIR` in the notebook configuration cell:

```python
# Kaggle
DATA_DIR = Path('/kaggle/input/tuberculosis-tb-chest-xray-dataset/TB_Chest_Radiography_Database')

# Local / Colab
DATA_DIR = Path('./TB_Chest_Radiography_Database')
```

---

## 🚀 Usage

### Run on Kaggle (Recommended)

1. Fork the notebook on Kaggle
2. Attach the *Tuberculosis (TB) Chest X-ray Dataset*
3. Enable GPU accelerator (P100 or T4)
4. **Run All**

### Run Locally / on Google Colab

```bash
jupyter notebook TB_Chest_Xray_Classification.ipynb
```

Or open in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Tip**: Enable a GPU runtime in Colab (*Runtime → Change runtime type → GPU*) for acceptable training times, especially for ViT-B/16.

---

## 🔬 Methodology

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Loss function | CrossEntropyLoss |
| LR scheduler | ReduceLROnPlateau |

### Differential Learning Rates (Transfer Learning)

Pre-trained backbone layers use a **10× lower learning rate** than the classification head:

```python
optimizer = Adam([
    {'params': backbone.parameters(), 'lr': LR * 0.1},
    {'params': head.parameters(),     'lr': LR * 1.0}
], weight_decay=WEIGHT_DECAY)
```

This preserves the general low-level features learned on ImageNet while allowing the head to adapt rapidly to the TB classification task.

### Evaluation Metrics

- **Accuracy** — overall correctness on the held-out test set
- **AUC-ROC** — area under the ROC curve; robust to class imbalance
- **Macro F1** — harmonic mean of precision/recall averaged across both classes
- **TB Recall (Sensitivity)** — fraction of TB cases correctly identified *(primary clinical metric)*
- **Confusion Matrix** — per-class breakdown of TP, TN, FP, FN

---

## 🏆 Conclusions & Recommendations

### Key Findings

1. **Transfer learning dramatically outperforms training from scratch** (~97–99% vs ~88% accuracy) even on a small dataset of ~4,200 images.
2. **ViT-B/16 achieves the best overall performance** — highest accuracy (~98.6%), AUC-ROC (0.999), and TB recall (~96%).
3. **DenseNet121 and EfficientNet-B0 offer excellent accuracy** with fewer parameters and faster inference than ViT.
4. Custom CNNs remain viable as lightweight baselines but require significantly more data to match pre-trained models.

### Deployment Recommendations

| Scenario | Recommended Model |
|---|---|
| Maximum accuracy / clinical screening | ViT-B/16 |
| Constrained GPU / edge deployment | EfficientNet-B0 |
| Medical imaging research baseline | DenseNet121 |

### Future Improvements

- **Class imbalance**: Apply weighted cross-entropy loss or SMOTE oversampling to further boost TB recall
- **Interpretability**: Integrate Grad-CAM (CNNs) or Attention Rollout (ViT) to visualise regions driving predictions
- **Ensemble**: Combining ViT + EfficientNet predictions could push AUC above 0.999
- **External validation**: Test on additional chest X-ray datasets (e.g., NIH ChestX-ray14, Montgomery County TB dataset)

> ⚠️ **Disclaimer**: All models are intended as **clinical decision-support tools** only. They must be validated in a regulated clinical setting and used alongside qualified medical expertise — not as standalone diagnostic systems.

---

## 📚 References

1. Dosovitskiy, A., et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. He, K., et al. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
3. Huang, G., et al. (2016). *Densely Connected Convolutional Networks (DenseNet)*. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
4. Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
5. Rajpurkar, P., et al. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. [arXiv:1711.05225](https://arxiv.org/abs/1711.05225)
6. Rahman, T., et al. *Tuberculosis (TB) Chest X-ray Dataset*. [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ for medical AI research  
*If this project helped you, please ⭐ the repository!*

</div>