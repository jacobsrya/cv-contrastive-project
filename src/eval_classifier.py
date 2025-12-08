# src/eval_classifier.py

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .datasets import get_cifar10_eval_dataloaders
from .models import Encoder
from .metrics import compute_classification_metrics


class LinearClassifier(nn.Module):
    """
    Simple linear classifier on top of frozen encoder embeddings.
    Assumes encoder outputs 512-dim vectors, CIFAR-10 has 10 classes.
    """

    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Downstream classifier on frozen encoder")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def extract_embeddings(encoder, loader, device):
    encoder.eval()
    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            feats = encoder(images)               # [B, 512]
            all_embeds.append(feats.cpu())
            all_labels.append(labels.cpu())

    embeds = torch.cat(all_embeds, dim=0)        # [N, 512]
    labels = torch.cat(all_labels, dim=0)        # [N]
    return embeds, labels


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Load data ----
    train_loader, test_loader = get_cifar10_eval_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # ---- Load encoder ----
    encoder = Encoder().to(device)

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()  # freeze encoder weights
    for p in encoder.parameters():
        p.requires_grad = False

    # ---- Extract embeddings ----
    print("Extracting train embeddings...")
    train_embeds, train_labels = extract_embeddings(encoder, train_loader, device)
    print("Extracting test embeddings...")
    test_embeds, test_labels = extract_embeddings(encoder, test_loader, device)

    # ---- Classifier on embeddings ----
    classifier = LinearClassifier(in_dim=encoder.out_dim, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Simple supervised training loop on embeddings
    for epoch in range(args.epochs):
        classifier.train()
        # mini-batch over embeddings to avoid big matrix ops
        perm = torch.randperm(train_embeds.size(0))
        batch_size = args.batch_size

        running_loss = 0.0
        for i in range(0, train_embeds.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = train_embeds[idx].to(device)
            yb = train_labels[idx].to(device)

            optimizer.zero_grad()
            logits = classifier(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / (train_embeds.size(0) / batch_size)
        print(f"Epoch [{epoch+1}/{args.epochs}] Classifier train loss: {avg_loss:.4f}")

    # ---- Evaluation on test embeddings ----
    classifier.eval()
    with torch.no_grad():
        logits = classifier(test_embeds.to(device))     # [N_test, num_classes]
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

    y_true = test_labels.numpy()
    y_pred = preds
    y_proba = probs

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    print("Downstream evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    

if __name__ == "__main__":
    main()
