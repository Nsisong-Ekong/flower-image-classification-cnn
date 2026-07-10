# Image Recognition to Identify Species of Flowers

Coursework project for the **Understanding Artificial Intelligence** module.
Author: **Nsisong Patrick Ekong**

## Overview

A Convolutional Neural Network (CNN) that classifies flower photographs into one
of five species using the TensorFlow `tf_flowers` dataset (~3,670 images):
**daisy, dandelion, roses, sunflowers, and tulips**.

## Approach

Images are resized to 180x180 and rescaled to the [0, 1] range. The training set
is augmented on the fly (random flips and rotation), shuffled, batched, and
prefetched. The report trains and compares **four CNN variants**, evaluating each
with accuracy, precision, recall, F1-score, and confusion matrices.

| Model | Key idea | Accuracy |
|-------|----------|----------|
| 1 | Baseline: 3 conv blocks (32-64-128), BatchNorm, Dropout | 71% |
| 2 | Larger 5x5 kernels, 2x2 strides, dropout 0.4 | 69% |
| 3 | Data augmentation, LR scheduler, early stopping | 61% |
| 4 | Deeper net + SMOTE + class weights for imbalance | **80%** |

Model 4 gave the best, most balanced performance. Future work: transfer learning.

## Contents

| File | Description |
|------|-------------|
| `flower_cnn.ipynb` | CNN notebook (data loading, augmentation, model, training) |
| `report.pdf` | Written report of methods, model iterations, and results |

## Tech stack

Python, TensorFlow / Keras, TensorFlow Datasets, NumPy, Matplotlib (Jupyter Notebook).
