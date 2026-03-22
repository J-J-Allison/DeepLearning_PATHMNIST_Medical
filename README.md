# PathMNIST — Colorectal Cancer Histology Classification

Deep learning project comparing four architectures on the [PathMNIST](https://medmnist.com/) dataset (89K training images, 9 tissue types, 28×28 RGB). Built for the **Deep Learning 2025–2026** final project at Université Paris 1 Panthéon-Sorbonne.

## Results

| Model | Parameters | Test Accuracy | Training Time |
|-------|-----------|---------------|---------------|
| MLP (3 hidden layers) | 1,370,121 | 64.62% | ~15 min |
| CNN (3 blocks + augmentation) | 289,065 | 89.62% | ~25 min |
| ResNet-18 (full fine-tuning) | 11,181,129 | **94.79%** | 21.6 min |
| ViT from scratch (patch=7) | 1,212,297 | 79.29% | 5.1 min |

## Project Structure

```
├── PathMNIST_Final_Project.ipynb   # Full notebook (Colab-ready)
└── README.md
```

## Architecture Overview

### Part 1 — Data Exploration
Visualisation of class distributions, sample images per tissue type, and per-channel pixel statistics compared against ImageNet.

### Part 2 — MLP Baseline
Three hidden layers (512 → 256 → 128) with dropout. Establishes a baseline at ~65% accuracy and reveals overfitting after epoch 10.

### Part 3 — CNN from Scratch
Three convolutional blocks with batch normalisation, dropout, and `AdaptiveAvgPool2d`. Trained with and without data augmentation (random flips, rotations, colour jitter) to measure the effect — augmentation improved test accuracy from 85.5% to 89.6%.

### Part 4 — Transfer Learning (ResNet-18)
ImageNet-pretrained ResNet-18 with two experiments:
- **Frozen backbone** (head only): 92.62%
- **Full fine-tuning**: 94.79% (+1.96pp)

Images resized from 28×28 to 224×224 to match the pretrained input resolution.

### Part 5 — Vision Transformer (from scratch)
Custom ViT with patch embedding, learnable CLS token, and positional embeddings. Three experiments:
- **Patch size 7** (16 patches): 79.29%
- **Patch size 14** (4 patches): 69.65% — confirms that too few tokens cripple self-attention
- **No positional embeddings**: 79.21% — virtually no drop, likely because a 4×4 grid has too little spatial complexity for position to matter

### Part 6 — Grad-CAM Interpretability
Custom Grad-CAM implementation using PyTorch hooks (no external library). Heatmap overlays for correctly and incorrectly classified images, with dual-heatmap comparison (predicted vs true class) on misclassified samples.

### Part 7 — Final Comparison
Summary table, per-class F1 analysis, top confused class pairs, and written analysis covering architecture trade-offs, data efficiency, clinical deployment considerations, and ethics.

## Key Findings

- **Transfer learning dominates** on small medical datasets — pretrained features compensate for limited data.
- **CNNs outperform ViTs** at this scale due to inductive biases (locality, translation equivariance) that ViTs must learn from scratch.
- **Smooth muscle** is the most confused class — its pink elongated fiber texture overlaps with stroma, mucus, and adipose tissue.
- **Positional embeddings** are redundant on a coarse 4×4 patch grid, contrary to standard ViT assumptions.

## Setup

```bash
pip install torch torchvision medmnist scikit-learn seaborn tqdm matplotlib
```

Run in [Google Colab](https://colab.research.google.com/) with GPU for best performance. Checkpoints are saved to Google Drive to allow resuming across sessions.

## Dataset

PathMNIST from [MedMNIST v2](https://medmnist.com/) — 9-class colorectal cancer histology:

| Class | Tissue Type |
|-------|------------|
| 0 | Adipose |
| 1 | Background |
| 2 | Debris |
| 3 | Lymphocytes |
| 4 | Mucus |
| 5 | Smooth muscle |
| 6 | Normal colon mucosa |
| 7 | Cancer-associated stroma |
| 8 | Colorectal adenocarcinoma epithelium |

**Splits:** 89,996 train / 10,004 val / 7,180 test — 28×28 RGB images.

## Tech Stack

- PyTorch
- torchvision (ResNet-18 pretrained weights)
- MedMNIST
- scikit-learn (metrics, confusion matrices)
- matplotlib / seaborn (visualisation)

## Authors

- Sufyan Nadat
- Jacques Allison
- Alexandre Megard

## License

Academic project — dataset subject to [MedMNIST license](https://medmnist.com/).
