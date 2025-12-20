# ResNet-ViT Hybrid Architecture for Philippine Currency Classification

**Computer Science VG Finals Project**

**Group Members:**
- Febron Jr. B. Sedoriosa
- Imroz Mae Khan

---

## Abstract

This case study explores the efficacy of "Architecture Fusion" in computer vision by combining a Convolutional Neural Network (CNN) with a Vision Transformer (ViT) for the fine-grained classification of Philippine currency. Leveraging a dataset of 16 distinct classes, including both coins and banknotes, we implemented a hybrid ResNetViTHybrid model. The architecture fuses the local feature extraction capabilities of a pretrained ResNet18 with the global context modeling of a Transformer Encoder. To address class imbalance, we employed Stratified Random Sampling. The model demonstrated rapid convergence, achieving >98% validation accuracy within the first epoch, suggesting that hybrid architectures effectively capture the subtle visual distinctions in currency denominations.

**Keywords:** Architecture Fusion · ResNet · Vision Transformer · Image Classification · Philippine Currency

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

The automation of currency recognition is a critical task for financial systems, automated vending machines, and assistive technologies designed for the visually impaired. However, distinguishing between currency denominations presents significant challenges due to:

- **Subtle inter-class similarities** (e.g., visual resemblance between old and new series of 10 Peso coins)
- **Intra-class variations** caused by wear and tear

### The Challenge

- **Standard CNNs** excel at detecting local patterns like edges and textures but struggle to model long-range spatial dependencies
- **Vision Transformers (ViTs)** are adept at modeling global relationships through self-attention mechanisms but typically require massive datasets to learn low-level features effectively

### Our Solution

This study proposes a **hybrid architecture** that fuses a CNN backbone with a Transformer Encoder. The primary objective is to combine the inductive bias of CNNs with the attention mechanisms of Transformers to achieve robust, fine-grained classification of Philippine currency.

---

## Dataset Description

The study utilized a custom dataset of **Philippine Currency** obtained from [Roboflow Universe](https://universe.roboflow.com/).

### Classes (16 Total)

**Banknotes:**
- 20 Pesos
- 50 Pesos
- 100 Pesos
- 200 Pesos
- 500 Pesos
- 1000 Pesos (standard)
- 1000 Pesos (Polymer variant)

**Coins:**
- 25 Centavos (Old)
- 25 Centavos (New)
- 1 Peso Coin (Old)
- 1 Peso Coin (New)
- 5 Peso Coin (Old)
- 5 Peso Coin (New)
- 10 Peso Coin (Old)
- 10 Peso Coin (New)
- 20 Peso Coin

### Addressing Class Imbalance

To address significant class imbalance in the raw data, we implemented **Stratified Random Sampling**:
- **Training set:** 100 images per class (~1,600 total)
- **Validation set:** 20 images per class (~320 total)
- **Test set:** Full dataset retained

### Preprocessing Pipeline

1. **Bounding Box Cropping:** Images cropped using provided COCO annotations to remove background noise
2. **Resizing:** All images resized to 224×224 pixels to match backbone architecture requirements
3. **Normalization:** Standard ImageNet normalization applied

---

## Methodology

### Architecture Fusion Strategy

The core contribution is the implementation of the **ResNetViTHybrid** class, a dual-stage pipeline:

#### Stage 1: Local Feature Extraction (CNN)
- **Backbone:** Pretrained ResNet18 (final FC layer removed)
- **Input:** 224×224×3 RGB image
- **Output:** Dense feature map of shape (512×7×7)
- **Purpose:** Captures high-level local features (textures, ridges, patterns)

#### Stage 2: Global Context Modeling (Transformer)
- **Token Generation:** Spatial features flattened into sequence of 49 tokens
- **Transformer Encoder:** Multi-Head Self-Attention mechanism
- **Purpose:** Models global structural relationships across all image patches
- **Final Layer:** Linear classifier mapping to 16 class logits

### Training Configuration

- **Environment:** Google Colab with T4 GPU
- **Epochs:** 5
- **Batch Size:** 16
- **Optimizer:** Adam (learning rate: 1×10⁻⁴)
- **Loss Function:** CrossEntropyLoss
- **Regularization:** Transfer learning with pretrained ResNet18 weights

---

## Architecture

```
Input Image (224×224×3)
        ↓
┌───────────────────┐
│  ResNet18 Backbone│
│  (Pretrained)     │
│  Local Features   │
└────────┬──────────┘
         ↓
   Feature Map
   (512×7×7)
         ↓
┌───────────────────┐
│  Flatten to       │
│  49 Tokens        │
│  (49×512)         │
└────────┬──────────┘
         ↓
┌───────────────────┐
│  Positional       │
│  Encoding         │
└────────┬──────────┘
         ↓
┌───────────────────┐
│  Transformer      │
│  Encoder Layer    │
│  Self-Attention   │
└────────┬──────────┘
         ↓
┌───────────────────┐
│  Mean Pooling     │
│  (1×512)          │
└────────┬──────────┘
         ↓
┌───────────────────┐
│  Linear Classifier│
│  (16 classes)     │
└───────────────────┘
```

---

## Results

### Quantitative Performance

| Metric | Epoch 1 | Epoch 5 |
|--------|---------|---------|
| **Training Loss** | 0.23 | 0.04 |
| **Validation Accuracy** | ~99% | ~99% |

### Key Findings

✅ **Rapid Convergence:** Model achieved >98% validation accuracy within the first epoch

✅ **High Precision:** Normalized confusion matrix shows strong diagonal dominance

✅ **Fine-Grained Distinction:** Successfully differentiated between visually similar classes:
- 1000 Pesos (Polymer) vs. 1000 Pesos (Standard)
- 10 Pesos (Old) vs. 10 Pesos (New)

### Qualitative Analysis

The Transformer's self-attention mechanism proved crucial for:
- Detecting texture patterns (e.g., shiny surface of polymer notes)
- Understanding global structural relationships across the image
- Distinguishing subtle visual differences between coin series

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Colab (optional)

### Setup

```bash
# Clone the repository
git clone https://github.com/beachater/resnet50-vit.git
cd resnet50-vit

# Install required packages
pip install torch torchvision
pip install pycocotools
pip install pillow matplotlib pandas numpy scikit-learn

# For Google Colab
# The notebook automatically mounts Google Drive
```

---

## Usage

### Running the Notebook

1. Open `PIT.ipynb` in Google Colab or Jupyter Notebook
2. Mount your Google Drive (if using Colab)
3. Update dataset paths in the notebook:
   ```python
   DATASET_ROOT = "/content/drive/MyDrive/philippine_currency_dataset"
   ```
4. Run all cells sequentially

### Training the Model

```python
# Toggle sampling mode
SAMPLE_MODE = False  # True for balanced subset, False for full dataset

# Configure sampling parameters (if SAMPLE_MODE=True)
TRAIN_SAMPLES_PER_CLASS = 100
VAL_SAMPLES_PER_CLASS = 20

# Train the model
# (See notebook for complete training loop)
```

### Making Predictions

```python
# Load trained model
model = ResNetViTHybrid(num_classes=16)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Predict on new image
# (See notebook for inference pipeline)
```

---

## Project Structure

```
resnet50-vit/
│
├── PIT.ipynb                           # Main Jupyter notebook
├── Computer_Science_VG_Finals.pdf      # Detailed project report
├── README.md                           # This file
│
└── (Google Drive structure)
    └── philippine_currency_dataset/
        ├── annotations_train.json      # COCO format annotations
        ├── annotations_val.json
        ├── annotations_test.json
        ├── train/                      # Training images
        ├── valid/                      # Validation images
        └── test/                       # Test images
```

---

## Discussion

### Why Architecture Fusion Works

1. **Complementary Strengths:**
   - ResNet provides robust low-level feature extraction
   - Transformer captures global contextual relationships

2. **Transfer Learning Advantage:**
   - Pretrained ResNet18 (ImageNet) provides strong visual foundation
   - Reduces training time and data requirements

3. **Stratified Sampling Impact:**
   - Balanced class distribution prevents majority class bias
   - Forces model to learn distinctive features for all denominations

### Limitations

⚠️ **Dependency on Pre-Cropped Images:** Real-world deployment requires integration with object detection pipeline

⚠️ **Dataset Size:** While sufficient for this study, larger datasets might improve generalization

⚠️ **Computational Cost:** Transformer layers add overhead compared to pure CNN approaches

---

## Conclusion

This case study successfully demonstrated the power of **Architecture Fusion** by combining CNNs and Vision Transformers. The ResNetViTHybrid model achieved:

- **Near-perfect accuracy** (~99%) on Philippine Currency classification
- **Minimal training time** (convergence within 1 epoch)
- **Robust fine-grained distinction** between visually similar classes

The fusion strategy effectively leveraged the efficiency of CNNs with the contextual power of Transformers, providing a scalable solution for fine-grained image classification tasks.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.

2. Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.

3. Philippine Currency Dataset. Roboflow Universe. Retrieved from https://universe.roboflow.com/

---

## License

This project is for educational purposes as part of Computer Science VG Finals.

---

## Acknowledgments

- Roboflow Universe for providing the Philippine Currency dataset
- Google Colab for providing free GPU resources
- PyTorch and torchvision teams for excellent deep learning frameworks

---

**For questions or collaboration, please contact the authors.**