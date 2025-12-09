# -------[ Contrastive Learning on CIFAR-10 ]-------

Course project for **EEL4930/5930 - Computer Vision**: self-supervised contrastive learning on CIFAR-10 with downstream classification.

This repository implements:

- A **contrastive pretraining** stage (self-supervised) using strong stochastic augmentations.
- A **downstream classifier** stage on frozen embeddings (supervised).


# --------------[ Project Structure ]--------------

```text
cv-contrastive-project/
│
├── src/
│   ├── __init__.py           # makes `src` a Python package
│   ├── datasets.py           # CIFAR-10 dataloaders + augmentations
│   ├── models.py             # Encoder (ResNet-18) + ProjectionHead
│   ├── contrastive_train.py  # Self-supervised contrastive pretraining loop
│   ├── eval_classifier.py    # Downstream classifier on frozen embeddings
│   └── metrics.py            # Accuracy, precision, recall, AUC helpers
├── requirements.txt          # Python dependencies
└── README.md                 # this file
```


# --------------------[ Setup ]--------------------

```bash
git clone https://github.com/jacobsrya/cv-contrastive-project
cd cv-contrastive-project
python -m venv .venv
source .venv/bin/activate   # On Windows (PowerShell): .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


# -----------[ Reproducing Experiments ]-----------

All of these commands were run ON WINDOWS from the project root with the virtual environment activated (`(.venv)` showing in the prompt).

# Pretraining runs:

Run 1 - 50-epoch snapshot(s) (tau = 0.2) (used in classifiers 1, 2, and 4):
```bash
python -m src.contrastive_train --data_dir ./data --batch_size 128 --epochs 50 --temperature 0.2 --out_dir ./checkpoints
```

Run 2 - 20-epoch snapshot(s) (tau = 0.5) (separate output directory so it does not overwrite Run 1 checkpoints) (used in classifier 3):
```bash
python -m src.contrastive_train --data_dir ./data --batch_size 128 --epochs 20 --temperature 0.5 --out_dir ./checkpoints_tau05
```


# Downstream Classifiers:

Run 1 downstream classifier @ epoch 1 (Using the epoch 1 checkpoint from Run 1):
```bash
python -m src.eval_classifier --data_dir ./data --checkpoint ./checkpoints/encoder_epoch_1.pt --batch_size 256 --epochs 20
```

Run 1 downstream classifier @ epoch 20 (Using the epoch 20 checkpoint from Run 1):
```bash
python -m src.eval_classifier --data_dir ./data --checkpoint ./checkpoints/encoder_epoch_20.pt --batch_size 256 --epochs 20
```

Run 1 downstream classifier @ epoch 50 (Using the epoch 50 checkpoint from Run 1):
```bash
python -m src.eval_classifier --data_dir ./data --checkpoint ./checkpoints/encoder_epoch_50.pt --batch_size 256 --epochs 20
```

Run 2 downstream classifier @ epoch 20 (Using the epoch 20 checkpoint from Run 2):
```bash
python -m src.eval_classifier --data_dir ./data --checkpoint ./checkpoints_tau05/encoder_epoch_20.pt --batch_size 256 --epochs 20
```

# See results.txt for the numerical metrics for each run (accuracy, precision, recall, F1, AUC).
