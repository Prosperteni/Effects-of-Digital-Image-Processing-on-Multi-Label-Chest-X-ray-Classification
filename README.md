# Model-Dependent and Class-Specific Effects of Digital Image Processing on Multi-Label Chest X-ray Classification

> **Does image preprocessing actually help? It depends on your model.**
> A controlled large-scale study of Digital Image Processing (DIP) techniques across CNN, Transformer, and Hybrid deep learning architectures on the NIH ChestX-ray14 dataset.

---

## Overview

Standard preprocessing is often treated as a neutral, fixed step in medical imaging pipelines. This work challenges that assumption.

We present a **systematic, controlled evaluation** of 15 preprocessing configurations across 5 deep learning architectures, revealing that preprocessing effectiveness is **strongly architecture-dependent** and **pathology-dependent** — not universally beneficial.

---

## Experimental Pipeline
<p align="center">
  <img src="Pipeline.png" alt="Pipeline" width="800"/>
  <br>
  <em>Fig 1. Overview of the experimental framework used in this study. The NIH ChestXray14 dataset was first partitioned using patient-wise splitting and standardized preprocessing. Baseline DL models were trained without DIP techniques, followed by controlled DIP-based experiments across multiple filter configurations. Final evaluation included per-class analysis, architecture comparison, and preprocessing sensitivity assessment.</em>
</p>


---

## Findings

- **CNN-based models** (ResNet-50, DenseNet-121) frequently achieved optimal performance under **baseline conditions** — excessive preprocessing degraded performance by disrupting local texture features.
- **Transformer and hybrid models** (ViT-B16, Swin-Tiny, ConvFormer) **benefited from contrast enhancement** (CLAHE, Histogram Equalization), likely due to their reliance on global contextual relationships.
- **No single preprocessing configuration** consistently improves performance across all 14 thoracic pathologies.
- Aggressive smoothing (large Gaussian kernels, high bilateral filter parameters) consistently **degraded performance** across most architectures.

---

## Repository Structure

```
.
├── dataset/
│   ├── Data_Entry_2017.csv          # NIH ChestXray14 labels
│   ├── train_val_list_NIH.txt       # Patient-wise train/val split
│   └── images-224/                  # Pre-resized CXR images (224×224)
│
├── models/
│   └── best_*.pth                   # Saved model checkpoints per config
│
├── results/
│   ├── result_*.json                # Per-experiment AUC results
│   └── *_ablation_summary.csv       # Summary tables per architecture
│
├── densenet_ablation.py             # DenseNet-121 ablation experiments
├── resnet_ablation.py               # ResNet-50 ablation experiments
├── vit_ablation.py                  # ViT-B16 ablation experiments
├── swin_ablation.py                 # Swin-Tiny ablation experiments
├── convformer_ablation.py           # ConvFormer ablation experiments
├── filters.py                       # DIP filter implementations
└── README.md
```

---

## ⚙️ Preprocessing Configurations

All 15 configurations were evaluated under **identical training conditions** (same optimizer, learning rate, data split, and augmentation) for fair comparison.

| # | Filter | Parameters |
|---|--------|------------|
| 1 | Baseline | No DIP applied |
| 2–6 | CLAHE | Clip limits: 1.0, 2.0, 3.0, 4.0, 5.0 |
| 7–10 | Gaussian Blur | Kernel sizes: 3, 5, 7, 9 |
| 11–14 | Bilateral Filter | Diameters: 5, 7, 9, 11 |
| 15 | Histogram Equalization | Global |

---

## Architectures

| Model | Type | Backbone |
|-------|------|----------|
| ResNet-50 | CNN | Residual connections |
| DenseNet-121 | CNN | Dense connections |
| ViT-B16 | Transformer | Patch-based self-attention |
| Swin-Tiny | Transformer | Shifted window attention |
| ConvFormer | Hybrid | Conv + attention |

All models use **ImageNet pre-trained weights** with a multi-label sigmoid output head (14 classes).

---

## Dataset

**NIH ChestX-ray14** — Wang et al., 2017

| Property | Value |
|----------|-------|
| Images | 112,120 frontal-view CXRs |
| Patients | 30,805 |
| Labels | 14 thoracic pathologies (multi-label) |
| Split | Patient-wise (no patient leakage) |

**Pathologies:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

> Dataset available at: [Kaggle](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)

Expected structure:

data/
├── Data_Entry_2017.csv
└── images/
    ├── 00000001_000.png
    ├── 00000002_000.png
    └── ...

---

## Getting Started

### Prerequisites

```bash
pip install torch torchvision opencv-python scikit-learn pandas numpy pillow tqdm
```

### Run an ablation study

```bash
# DenseNet-121 across all 15 configs
python densenet_ablation.py

# ResNet-50
python resnet_ablation.py
```

Results are saved to `results/` as `.json` per experiment and a summary `.csv`.

### Filter implementations

All DIP filters are implemented in `filters.py` and applied **before** the torchvision transform pipeline on PIL images:

```python
from filters import apply_filter

# Example: CLAHE with clip limit 2.0
image = apply_filter(image, filter_type='clahe', params={'clip_limit': 2.0, 'tile_grid': 8})
```

---

## Training Details

| Setting | Value |
|---------|-------|
| Image size | 224 × 224 |
| Batch size | 128 |
| Optimizer | Adam (lr = 1e-4) |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Loss function | BCEWithLogitsLoss |
| Early stopping | Patience = 10 (val loss) |
| Max epochs | 100 |
| Mixed precision | AMP (float16) |
| Metric | AUC-ROC (per-class + mean) |

**Augmentation (train only):** Random horizontal flip, ±10° rotation, brightness/contrast jitter (0.1).

---

## Evaluation

Performance is measured using **AUC-ROC** per pathology class and mean AUC across all 14 classes. Results are visualized as:

- Per-architecture mean AUC tables across all 15 configs
- Per-disease × per-config heatmaps showing preprocessing sensitivity

---
## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

- NIH Clinical Center for the ChestX-ray14 dataset
- PyTorch and torchvision teams
- OpenCV for DIP filter implementations
