# Keras Feature Extraction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)

> Transfer learning feature extraction pipeline using Keras pre-trained models. Extracts deep features from image datasets and trains a lightweight classifier on top.

---

## 📋 Overview

This project implements a **transfer learning** approach where a pre-trained CNN (e.g., VGG16, ResNet) is used as a fixed feature extractor. The extracted features are saved to disk and then used to train a  simple classifier, avoiding the need to run the full network for every training epoch.

---

## 📁 File Structure

```
keras-feature-extraction/
├── build_dataset.py        # Build and organize image dataset
├── extract_features.py     # Extract CNN features using pre-trained Keras model
├── train.py                # Train classifier on extracted features
└── pyimagesearch/          # Helper utilities (imutils-based)
```

---

## 🔄 Pipeline

1. **Build Dataset** — Organize images into class folders
   ```bash
   python build_dataset.py
   ```

2. **Extract Features** — Run images through pre-trained CNN and save features
   ```bash
   python extract_features.py
   ```

3. **Train Classifier** — Train a Logistic Regression or MLP on extracted features
   ```bash
   python train.py
   ```

---

## ⚙️ Requirements

```bash
pip install tensorflow keras numpy scikit-learn imutils h5py
```

---

## 💡 Key Concepts

- **Transfer learning:** Reusing pre-trained CNN weights (ImageNet) for new tasks
- **Bottleneck features:** Extracting the final feature vector before the classification head
- **HDF5 storage:** Efficiently storing large feature arrays to disk
- **Fast training:** Since features are pre-extracted, classifier training is very fast
