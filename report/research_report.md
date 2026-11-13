# Comparative Analysis of Machine Learning Architectures for Fashion-MNIST Classification: Accuracy, Computational Cost, and Robustness to Gaussian Noise

**Project type:** Experimental Machine Learning Study  
**Dataset:** Fashion-MNIST  
**Framework:** PyTorch  
**Execution environment:** CPU  
**Status:** Experimental project report — Version 1

## Abstract

This study investigates the performance of three machine learning architectures — Logistic Regression, Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN) — on the Fashion-MNIST image classification dataset.

The models are compared using classification accuracy, precision, recall, F1-score, training time, model parameter count, and robustness to Gaussian noise.

On the clean test set, CNN achieved the highest accuracy of 88.56%, followed by MLP at 87.87% and Logistic Regression at 83.55%. In repeated experiments using three random seeds, CNN achieved the highest mean validation accuracy of 89.00% (standard deviation 0.366 percentage points).

However, under severe Gaussian noise, Logistic Regression demonstrated greater robustness, achieving 46.15% accuracy at a noise standard deviation of 0.4, compared with 40.48% for MLP and 26.08% for CNN.

These findings illustrate a trade-off between clean-image classification performance, computational cost, and robustness to image corruption.

## 1. Introduction

Image classification is an important task in computer vision. Different machine learning architectures learn image representations in different ways, which can influence their classification performance and robustness.

This project compares three architectures:

1. Logistic Regression — a classical linear classification baseline.
2. Multi-Layer Perceptron — a fully connected neural network.
3. Convolutional Neural Network — a neural network designed to learn spatial image features.

The central research question is:

**How do different machine learning architectures compare in classification accuracy, computational cost, and robustness to noisy images?**

## 2. Dataset

Fashion-MNIST contains 70,000 grayscale clothing images belonging to ten categories.

Each image has a resolution of 28 × 28 pixels.

The dataset was divided into:

| Split | Images |
|---|---:|
| Training | 54,000 |
| Validation | 6,000 |
| Testing | 10,000 |
| Total | 70,000 |

The ten categories are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

## 3. Methodology

### 3.1 Model architectures

**Logistic Regression**

The image is flattened into a 784-dimensional feature vector and passed through a linear layer with ten output units.

Trainable parameters: 7,850.

**Multi-Layer Perceptron**

Architecture:

- Input: 784 features
- Hidden layer 1: 256 neurons
- ReLU activation
- Dropout: 0.2
- Hidden layer 2: 128 neurons
- ReLU activation
- Output: 10 classes

Trainable parameters: 235,146.

**Convolutional Neural Network**

Architecture:

- Convolution: 1 to 16 channels, 3 × 3 kernel
- ReLU and max pooling
- Convolution: 16 to 32 channels, 3 × 3 kernel
- ReLU and max pooling
- Fully connected layer: 1568 to 64
- ReLU and dropout: 0.25
- Output layer: 10 classes

Trainable parameters: 105,866.

### 3.2 Training configuration

All models were trained using:

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Cross-Entropy Loss
- Training duration: 5 epochs
- Execution device: CPU

### 3.3 Evaluation metrics

The models were evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Training time
- Number of trainable parameters

### 3.4 Robustness evaluation

Gaussian noise was added to the test images at five standard deviation levels:

0.0, 0.1, 0.2, 0.3, and 0.4.

The noisy images were clipped to the valid pixel range of [0, 1].

All three trained models were evaluated at each noise level without retraining.

## 4. Experimental Results

### 4.1 Clean test-set performance

| Metric | Logistic Regression | MLP | CNN |
|---|---:|---:|---:|
| Accuracy | 83.55% | 87.87% | 88.56% |
| Macro precision | 0.8340 | 0.8784 | 0.8845 |
| Macro recall | 0.8355 | 0.8787 | 0.8856 |
| Macro F1-score | 0.8314 | 0.8775 | 0.8843 |

CNN achieved the highest clean test accuracy.

### 4.2 Computational cost

| Model | Parameters | Mean training time across three seeds |
|---|---:|---:|
| Logistic Regression | 7,850 | 35.88 seconds |
| MLP | 235,146 | 47.68 seconds |
| CNN | 105,866 | 174.19 seconds |

CNN required more training time in the CPU environment, while Logistic Regression had the smallest parameter count and shortest training time.

### 4.3 Reproducibility analysis

Each architecture was trained independently using three random seeds: 42, 123, and 2026.

| Model | Mean validation accuracy | Standard deviation |
|---|---:|---:|
| Logistic Regression | 84.344% | 0.255 |
| MLP | 87.922% | 0.212 |
| CNN | 89.000% | 0.366 |

CNN achieved the highest mean validation accuracy. MLP exhibited the smallest variation across the three runs.

### 4.4 Gaussian noise robustness

| Noise standard deviation | Logistic Regression | MLP | CNN |
|---|---:|---:|---:|
| 0.0 | 83.55% | 87.87% | 88.56% |
| 0.1 | 82.56% | 85.40% | 85.38% |
| 0.2 | 75.16% | 74.73% | 66.55% |
| 0.3 | 60.52% | 55.71% | 42.18% |
| 0.4 | 46.15% | 40.48% | 26.08% |

Logistic Regression demonstrated the highest accuracy at the two strongest noise levels, 0.3 and 0.4.

CNN experienced a substantial decline in accuracy as noise increased.

### 4.5 Class-wise error analysis

The confusion matrices and class-wise recall measurements revealed that:

- CNN performed particularly well on trousers, sandals, coats, dresses, and bags.
- MLP achieved the highest recall for pullovers and sneakers.
- Shirt classification was difficult for all architectures.
- CNN achieved 62.0% shirt recall, compared with 60.6% for MLP and 44.8% for Logistic Regression.

The frequent confusion between visually similar clothing categories suggests that low-resolution grayscale images present a significant classification challenge.

## 5. Discussion

The experimental results reveal that model complexity and clean-image classification accuracy do not necessarily translate into better robustness.

CNN achieved the highest clean test accuracy and highest mean validation accuracy across three seeds. Its ability to learn spatial features likely contributed to its strong performance on clean images.

However, Logistic Regression performed better under severe Gaussian noise. At a noise standard deviation of 0.4, its accuracy was 46.15%, substantially higher than the CNN's 26.08%.

This demonstrates that a model with lower clean-image accuracy can still be more robust under a particular form of image corruption.

The results also suggest that computational efficiency should be considered alongside predictive performance, particularly when models are trained or deployed on limited hardware.

## 6. Limitations

This study has several limitations:

1. The experiments use only one dataset.
2. Training was performed on a CPU.
3. Models were trained for only five epochs.
4. The reproducibility analysis uses three random seeds.
5. The clean test-set comparison represents one trained model per architecture.
6. The Gaussian noise experiment uses one corruption realization per noise level rather than averaging across many independent noise realizations.
7. Only Gaussian pixel noise was evaluated; other corruptions may produce different rankings.
8. The experiments do not establish statistical significance between model performances.

Future work should include additional datasets, more random seeds, longer training schedules, additional noise types, and repeated corruption trials.

## 7. Conclusion

This study compared Logistic Regression, MLP, and CNN architectures on Fashion-MNIST using clean-image classification, computational cost, class-wise analysis, and Gaussian noise robustness.

CNN achieved the best clean test accuracy of 88.56% and the highest mean validation accuracy of 89.00% across three random seeds.

However, Logistic Regression demonstrated better robustness under severe Gaussian noise, achieving 46.15% accuracy at a noise standard deviation of 0.4.

The results highlight an important distinction between classification performance on clean data and robustness under corrupted inputs.

Overall, the study demonstrates that model selection should consider multiple factors, including accuracy, computational cost, and robustness, rather than relying on a single evaluation metric.

## 8. Reproducibility

The experiments were implemented in Python using PyTorch and Torchvision.

The project includes:

- Dataset preparation
- Three model implementations
- Training and validation procedures
- Test-set evaluation
- Gaussian noise robustness experiments
- Confusion matrices
- Class-wise recall analysis
- Multiple-seed experiments
- CSV result files and plots

The source code and experimental outputs should be published together to support reproducibility.

---

**Keywords:** Fashion-MNIST, Image Classification, Logistic Regression, MLP, CNN, Robustness, Gaussian Noise, Deep Learning, Computer Vision.
