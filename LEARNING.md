# Understanding LeNet: A Learning Guide

## What is LeNet?

LeNet is one of the earliest and most influential Convolutional Neural Networks (CNNs), developed by **Yann LeCun** and his colleagues in the late 1980s and 1990s. The most famous version, **LeNet-5**, was published in 1998 and was designed for handwritten digit recognition (MNIST dataset).

LeNet-5 is historically significant because it established many of the core architectural patterns still used in modern deep learning:
- Convolutional layers for feature extraction
- Subsampling/pooling layers for spatial dimension reduction
- Fully connected layers for classification

## The Original LeNet-5 Architecture

The original LeNet-5 architecture (designed for 32×32 grayscale images) consists of:

| Layer | Type | Output Shape | Parameters |
|-------|------|--------------|------------|
| Input | - | 32×32×1 | - |
| C1 | Convolution (6 filters, 5×5, valid) | 28×28×6 | 156 |
| S2 | Average Pooling (2×2, stride 2) | 14×14×6 | 12 |
| C3 | Convolution (16 filters, 5×5, valid) | 10×10×16 | 2,416 |
| S4 | Average Pooling (2×2, stride 2) | 5×5×16 | 32 |
| C5 | Convolution (120 filters, 5×5, valid) | 1×1×120 | 48,120 |
| F6 | Fully Connected | 84 | 10,164 |
| Output | Fully Connected | 10 | 850 |

**Total parameters: ~60,000**

### Key Characteristics of LeNet-5:

1. **5×5 Convolution Kernels**: All convolutional layers use 5×5 filter sizes
2. **Average Pooling (2×2)**: Subsampling layers use 2×2 average pooling with stride 2
3. **Progressive Feature Maps**: 6 → 16 → 120 feature maps
4. **Activation Functions**: Originally used tanh or sigmoid (modern implementations often use ReLU)
5. **Fully Connected Layers**: 120 → 84 → num_classes

## What This Repository Does

This repository implements a **modified LeNet-style** CNN for basketball action classification (distinguishing between dunks and layups, or free throws). The key adaptations are:

1. **Larger Input Images**: Uses 320×180×3 RGB images instead of 32×32×1 grayscale
2. **Different Task**: Binary classification of basketball actions instead of 10-class digit recognition
3. **Modern Practices**: Uses ReLU activation and dropout for regularization

## Code Files Overview

### `LeNet_clean.py`
The inference script that:
- Loads a pre-trained model from `model.json` and `model.h5`
- Reads test data from a CSV file
- Evaluates the model's accuracy, precision, and recall

### `Jupyter_train_model.ipynb`
The training notebook that:
- Defines the LeNet-inspired architecture
- Trains the model on basketball action images
- Saves the trained model weights

### `success.ipynb`
An alternative training notebook (with some architectural issues - see below).

## Differences from Official LeNet-5

### Filter Counts
- **Original LeNet-5**: 6 → 16 filters
- **Jupyter_train_model.ipynb**: 6 → 16 filters ✓ (Correct!)
- **success.ipynb**: 32 → 64 filters (Different - more like VGG style)

### Kernel Sizes
- **Original LeNet-5**: 5×5 kernels
- **This Implementation**: Should use 5×5 for authentic LeNet (3×3 is more like VGGNet style)

### Pooling
- **Original LeNet-5**: 2×2 average pooling
- **This Implementation**: Should use 2×2 pooling (max pooling is acceptable in modern implementations)

### Fully Connected Layers
- **Original LeNet-5**: 120 → 84 → num_classes
- **Jupyter_train_model.ipynb**: 120 → 84 → 2 ✓ (Correct!)

## Known Issues Fixed

### Issue 1: Wrong Output Layer in success.ipynb
**Problem**: `Dense(1, activation='softmax')` is incorrect for classification.
- Softmax with 1 output always produces 1.0 (since softmax normalizes outputs to sum to 1)
- For binary classification, you need either:
  - `Dense(2, activation='softmax')` with `sparse_categorical_crossentropy`
  - `Dense(1, activation='sigmoid')` with `binary_crossentropy`

**Fix**: Changed to `Dense(2, activation='softmax')` to match the rest of the codebase.

### Issue 2: Kernel Size Not Matching LeNet-5
**Problem**: Using 3×3 kernels throughout.
- While 3×3 kernels work well (VGGNet style), they don't match the original LeNet-5 specification.

**Note**: For the adapted version in `Jupyter_train_model.ipynb`, we've kept the architecture closer to the original LeNet-5 with 5×5 kernels.

## How to Use This Code

### Training a Model
1. Prepare your data in CSV format with image paths and labels
2. Place images in the `data/` directory
3. Run the training notebook (`Jupyter_train_model.ipynb`)
4. The model will be saved as `model.json` and `model.h5`

### Running Inference
```python
from LeNet_clean import inference

results = inference('data/your_test_data.csv')
print(f"Accuracy: {results['accuracy']}")
print(f"Precision: {results['precision']}")
print(f"Recall: {results['recall']}")
```

## Further Reading

- [Original LeNet-5 Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) - "Gradient-Based Learning Applied to Document Recognition" by LeCun et al., 1998
- [Keras Documentation](https://keras.io/) - For understanding the API used in this code
- [Stanford CS231n](http://cs231n.stanford.edu/) - Excellent course on CNNs and deep learning for computer vision

## Summary

This repository contains a practical implementation of a LeNet-inspired CNN for image classification. While it adapts the original architecture for different input sizes and tasks, it maintains the core principles that made LeNet-5 groundbreaking:
- Hierarchical feature learning through convolutions
- Spatial dimension reduction through pooling
- Final classification through fully connected layers

Understanding LeNet-5 is essential for anyone learning deep learning, as it forms the foundation for modern architectures like AlexNet, VGGNet, and ResNet.
