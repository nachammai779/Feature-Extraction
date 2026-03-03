# Test Features

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

> Pre-computed protein structure feature files in NumPy format used for testing the feature extraction pipeline.

---

## 📋 Overview

This folder contains pre-computed feature matrices extracted from protein structure data. Each `.npy` file stores a precision matrix (inverse covariance matrix) of size 441×441, derived from protein chain coordinates. These files serve as test inputs for the feature extraction and classification pipeline.

---

## 📁 Files

| File | Protein Chain | Description |
|------|--------------|-------------|
| `1am9A.pre441.npy` | 1AM9, Chain A | Precision matrix features for protein 1AM9 |
| `1b33N.pre441.npy` | 1B33, Chain N | Precision matrix features for protein 1B33 |
| `1b9wA.pre441.npy` | 1B9W, Chain A | Precision matrix features for protein 1B9W |

---

## 🔬 File Format

- **Format:** NumPy binary array (`.npy`)
- **Shape:** 441 × 441 precision matrix
- **Naming Convention:** `<PDBID><Chain>.pre441.npy`
  - PDB ID: 4-character protein databank identifier
  - Chain: Single-letter chain identifier
  - `pre441`: Indicates precision matrix of dimension 441

---

## 🚀 Usage

```python
import numpy as np

# Load a test feature file
features = np.load('1am9A.pre441.npy')
print(features.shape)  # (441, 441)

# Flatten for use as input to a classifier
flat_features = features.flatten()
```

---

## 📚 Context

These test features are intended to be used with the `keras-feature-extraction` pipeline. The precision matrix representation captures the correlation structure of protein residue positions, enabling downstream machine learning models to learn structural patterns from protein data.
