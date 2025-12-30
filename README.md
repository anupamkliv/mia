# MIA: Centralized vs Federated (Shadow-model-free, Black-box)

This repository contains the experimental code used in the study **“Can Federated Models Keep Secrets Better?”**. It provides pipelines for **centralized training/inference** and **federated learning (FL)** training, along with scripts used to run **inference-time dropout analysis** and comparisons against prior baselines.

## Repository layout

Top-level modules:

- `centralized/` — centralized training + inference-time dropout analysis utilities
- `centralized_medical/` — centralized experiments for medical datasets/tasks
- `fl/` — federated learning training and evaluation (e.g., global aggregated model checkpoints)
- `prior/` — prior / baseline implementations used for comparison

> If you are new to the repo, start with `centralized/` and `fl/`. The `prior/` folder is mainly for reproducing baseline comparisons.

---

## Setup

### 1) Create an environment (recommended)

Using conda:

```bash
conda create -n mia python=3.10 -y
conda activate mia
