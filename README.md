# MIA: Centralized vs Federated (Shadow-model-free, Black-box)

This repository contains the experimental code used in the study **“Can Federated Models Keep Secrets Better?”**. It provides pipelines for **centralized training/inference** and **federated learning (FL)** training, along with scripts used to run **inference-time dropout analysis** and comparisons against prior baselines.

## Repository layout

Top-level modules:

- **centralized/**
  - Trains models in a standard centralized setup
  - Performs inference-time dropout analysis on the trained target model

- **centralized_medical/**
  - Centralized pipelines adapted for medical datasets and tasks

- **fl/**
  - Federated learning training (e.g., FedAvg)
  - Produces a final aggregated global model
  - Applies inference-time dropout analysis on the aggregated model

- **prior/**
  - Baseline and prior membership inference implementations
  - Used for comparative evaluation

> If you are new to the repo, start with `centralized/` and `fl/`. The `prior/` folder is mainly for reproducing baseline comparisons.

---

# Available Datasets and Models

## Datasets

### Image Classification Datasets

The following datasets are supported in the centralized and federated pipelines:

- **CIFAR-10**  
- **CIFAR-100**  
- **SVHN**  

These datasets are typically loaded using torchvision utilities and are split consistently across training and test sets.  
In federated learning experiments, the training data is partitioned across clients, while the test set remains centralized at the server.

---
## Medical Datasets

Medical experiments in this repository are conducted using datasets from the **MedMNIST** collection, a curated benchmark suite designed for lightweight and standardized evaluation of medical image classification tasks.

The following MedMNIST datasets are currently supported and used in this repository:

- **PneumoniaMNIST**  
  A chest X-ray dataset for binary classification, focusing on pneumonia detection.

- **OCTMNIST**  
  An optical coherence tomography (OCT) dataset for multi-class retinal disease classification.

- **BreastMNIST**  
  A breast ultrasound dataset used for binary classification of benign vs malignant cases.

The datasets are obtained directly from the **MedMNIST** benchmark and follow its standardized preprocessing and label definitions.

---

## Segmentation Dataset

### ISIC 2016 Skin Lesion Dataset

Segmentation experiments in this repository are conducted using the **ISIC 2016 Challenge dataset**, released by the International Skin Imaging Collaboration (ISIC).

The dataset consists of dermoscopic images with corresponding pixel-wise annotations for skin lesion segmentation. It is widely used as a benchmark for evaluating medical image segmentation models under realistic clinical variability.


---

## Model Architectures

### Classification Models

The following architectures are currently supported:

- **ResNet-18**
- **ResNet-34**
- **MobileNetV3-Small**
- **MobileNetV3-Large**

These models are used for both centralized and federated experiments.  Dropout layers are explicitly enabled during inference for sensitivity analysis.

---

## Segmentation Models

The following segmentation architectures are currently supported and evaluated:

- **U-Net**  
- **U-Net++**  
- **DeepLabV3**  
- **DeepLabV3+**  

---







## Setup

### 1) Create an environment (recommended)

Using conda:

```bash
conda create -n mia python=3.10 -y
conda activate mia
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```
### 3) Centralized learning pipeline

## Step 1: Train a centralized target model

```bash
python centralized/train.py \
  --dataset cifar10 \
  --model resnet18 \
  --epochs 50 \
  --batch_size 64 \
  --device cuda:0
```
This produces a trained target model, typically saved under a path such as:
```bash
dropout_results/<dataset>/<model>.pth
```

## Step 2: Inference-time dropout analysis (centralized)
Inference-time dropout is activated with varying probabilities, and multiple stochastic forward passes are used to estimate output deviation.

```bash
python centralized/abilation.py \
  --dataset cifar10 \
  --model resnet18 \
  --checkpoint dropout_results/cifar10/resnet18.pth \
  --device cuda:0
```

Typical experimental configuration used in this repository:
- Dropout probabilities: 0.01 to 0.10
- Number of stochastic passes per sample: T = 5
- Metrics: accuracy fluctuation, standard deviation, averaged deviation across samples


### 4) Federated learning pipeline

## Step 1: Train a federated model
  Run federated training with multiple clients and communication rounds:

```bash
python fl/train_fl.py \
  --dataset cifar10 \
  --model resnet18 \
  --fed_algo fedavg \
  --clients 10 \
  --rounds 10 \
  --local_epochs 10 \
  --device cuda:0
```

This produces a final aggregated global model, typically saved as:

```bash
checkpoints/fl/<dataset>_<model>_<fed_algo>_global.pth
```

## Step 2: Inference-time dropout analysis on the FL global model
The final aggregated global model is treated as the victim model for inference-time dropout analysis.

```bash

python fl/dropout_inference.py \
  --dataset cifar10 \
  --model resnet18 \
  --checkpoint checkpoints/fl/cifar10_resnet18_fedavg_global.pth \
  --device cuda:0
```

This enables direct comparison between centralized and federated models under an identical inference protocol.

## Troubleshooting
- Checkpoint loading errors
  - Some checkpoints may store weights under state_dict
  - Use strict=False if required
- Dropout not activating
  - Ensure dropout layers are explicitly set to training mode during inference
- GPU selection
  - Use --device cuda:X where X is the GPU index



