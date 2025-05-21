# MNIST Digit Classification with PyTorch

## 📌 Overview

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. It includes custom dataset handling for raw IDX files and a training-evaluation pipeline.

## 📂 Dataset

The MNIST dataset consists of:

- **60,000 training images** and **10,000 test images**
- Each image is a grayscale image of **28x28 pixels**
- Dataset used in this project is in IDX format:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`
  - `t10k-images.idx3-ubyte`
  - `t10k-labels.idx1-ubyte`

A custom `MNISTDataset` class is created to parse these files.

## 🧠 Model Architecture

The model is defined in the `MnistClassifier` class. It consists of:

- **Conv2d(1, 16)** ➝ ReLU ➝ MaxPool2d(2)
- **Conv2d(16, 32)** ➝ ReLU ➝ MaxPool2d(2)
- **Conv2d(32, 64)** ➝ ReLU ➝ MaxPool2d(2)
- **Flatten** ➝ **Linear(64 * 3 * 3 → 64)** ➝ ReLU
- **Linear(64 → 10)** for classification

```python
Total parameters: 60,618
Trainable parameters: 60,618
