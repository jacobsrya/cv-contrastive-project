# Contrastive Learning on CIFAR-10

Course project for **EEL4930/5930 – Computer Vision**: self-supervised contrastive learning on CIFAR-10 with downstream classification.

This repository implements:

- A **contrastive pretraining** stage (self-supervised) using strong stochastic augmentations.
- A planned **downstream classifier** stage on frozen embeddings (supervised).

---

## 1. Project Structure

```text
cv-contrastive-project/
│
├── configs/                  # (optional) config files (hyperparameters, paths)
├── src/
│   ├── __init__.py           # makes `src` a Python package
│   ├── datasets.py           # CIFAR-10 dataloaders + augmentations
│   ├── models.py             # Encoder (ResNet-18) + ProjectionHead
│   ├── contrastive_train.py  # Self-supervised contrastive pretraining loop
│   ├── eval_classifier.py    # (planned) Downstream classifier on frozen embeddings
│   ├── metrics.py            # (planned) Accuracy, precision, recall, AUC helpers
│   └── utils.py              # (planned) Seeding, checkpointing, logging
├── scripts/                  # (optional) shell scripts for running experiments
├── report/                   # project report (paper) lives here
├── requirements.txt          # Python dependencies
└── README.md                 # this file
```
---

## Setup

```bash
git clone https://github.com/yourusername/cv-contrastive-project.git
cd cv-contrastive-project
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt