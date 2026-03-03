# Feature Extraction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

> Feature extraction pipeline for protein structure prediction using deep learning. Leverages pre-trained Keras CNN models to extract features from image datasets, with pre-computed precision matrix features for test proteins.

---

## Overview

This repository implements a **transfer learning feature extraction pipeline** combining two approaches:

1. **Keras-based image feature extraction** — A 3-step pipeline that builds a dataset, extracts CNN features using a pre-trained model (e.g., VGG16, ResNet), and trains a lightweight classifier on the extracted features.
2. **Pre-computed protein structure features** — NumPy precision matrix files (.npy) representing structural features of protein chains, used as test inputs for downstream prediction tasks.

---

## Repository Structure

```
Feature-Extraction/
├── keras-feature-extraction/       # Transfer learning pipeline
│   ├── build_dataset.py            # Build and organize image dataset
│   ├── extract_features.py         # Extract CNN features using pre-trained Keras model
│   ├── train.py                    # Train classifier on extracted features
│   └── pyimagesearch/              # Helper utilities (imutils-based)
│
└── test-features/                  # Pre-computed protein structure features
    ├── 1am9A.pre441.npy            # Precision matrix for protein 1AM9, Chain A
    ├── 1b33N.pre441.npy            # Precision matrix for protein 1B33, Chain N
    └── 1b9wA.pre441.npy            # Precision matrix for protein 1B9W, Chain A
```

---

## Modules

### keras-feature-extraction

A full transfer learning pipeline for image classification:

| Step | Script | Description |
|------|--------|--------------|
| 1 | build_dataset.py | Organizes raw images into class-labeled folder structure |
| 2 | extract_features.py | Passes images through a pre-trained CNN; saves extracted features to disk |
| 3 | train.py | Trains a lightweight classifier (e.g., Logistic Regression) on the saved features |

**Why this approach?** By extracting features once and saving them, the full network only runs once — dramatically reducing training time compared to end-to-end fine-tuning.

---

### test-features

Pre-computed **precision matrix** (inverse covariance matrix) features derived from protein 3D structure coordinates:

| File | Protein | Chain | Shape |
|------|---------|-------|-------|
| 1am9A.pre441.npy | 1AM9 | A | 441 x 441 |
| 1b33N.pre441.npy | 1B33 | N | 441 x 441 |
| 1b9wA.pre441.npy | 1B9W | A | 441 x 441 |

Each file stores a precision matrix capturing inter-residue correlation patterns from protein structural data.

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/nachammai779/Feature-Extraction.git
cd Feature-Extraction

# Install dependencies
pip install tensorflow keras numpy imutils scikit-learn

# Step 1: Build dataset
python keras-feature-extraction/build_dataset.py

# Step 2: Extract features
python keras-feature-extraction/extract_features.py

# Step 3: Train classifier
python keras-feature-extraction/train.py
```

---

## Requirements

- Python 3.6+
- TensorFlow / Keras
- NumPy
- scikit-learn
- imutils
