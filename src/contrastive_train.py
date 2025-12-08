# src/contrastive_train.py

import argparse
import os

import torch
import torch.nn.functional as F

from .datasets import get_cifar10_contrastive_dataloader
from .models import Encoder, ProjectionHead


def parse_args():
    parser = argparse.ArgumentParser(description="Contrastive pretraining on CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE / NT-Xent with cosine similarity."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.matmul(z, z.T)      # cosine sim if z normalized

    sim = sim / temperature
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)

    indices = torch.arange(2 * batch_size, device=z.device)
    pos_indices = (indices + batch_size) % (2 * batch_size)
    log_prob = F.log_softmax(sim, dim=1)
    loss = -log_prob[indices, pos_indices]
    return loss.mean()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_loader = get_cifar10_contrastive_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    encoder = Encoder().to(device)
    proj_head = ProjectionHead(in_dim=512, hidden_dim=512, out_dim=128).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        encoder.train()
        proj_head.train()
        running_loss = 0.0

        for step, (views, _) in enumerate(train_loader):
            v1, v2 = views  # views is a tuple: (v1_batch, v2_batch)

            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)


            optimizer.zero_grad()
            h1 = encoder(v1)
            h2 = encoder(v2)

            z1 = F.normalize(proj_head(h1), dim=1)
            z2 = F.normalize(proj_head(h2), dim=1)

            loss = info_nce_loss(z1, z2, temperature=args.temperature)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 50 == 0:
                avg = running_loss / (step + 1)
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step}/{len(train_loader)}] Loss: {avg:.4f}")

        ckpt_path = os.path.join(args.out_dir, f"encoder_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "proj_head_state_dict": proj_head.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
