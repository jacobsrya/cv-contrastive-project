# src/models.py

import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """ResNet-18 backbone that outputs a 512-dim feature vector."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None)  # no pretraining
        # keep everything except final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)          # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)     # [B, 512]
        return x


class ProjectionHead(nn.Module):
    """Small 2-layer MLP: 512 -> 512 -> 128."""

    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
