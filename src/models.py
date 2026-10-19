"""Arquiteturas originais do estudo; entradas RGB em NCHW e intervalo [0, 1]."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    """Vetor RGB -> 128 -> 64 -> seis logits; sem softmax antes da perda."""

    def __init__(self, width: int = 128, height: int = 96, classes: int = 6,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * height * width, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class CNN(nn.Module):
    """Três blocos convolucionais e GAP: parâmetros independem da resolução."""

    def __init__(self, classes: int = 6) -> None:
        super().__init__()
        blocks = []
        in_channels = 3
        for index, out_channels in enumerate((16, 32, 64)):
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=2 if index == 0 else 1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images)).flatten(1)
        return self.classifier(features)


def build_model(config: dict[str, Any]) -> nn.Module:
    """Cria um modelo novo; o chamador define a semente antes desta chamada."""
    if config["architecture"] == "mlp":
        return MLP(width=int(config["width"]), height=int(config["height"]),
                   dropout=float(config.get("dropout", 0.2)))
    if config["architecture"] == "cnn":
        return CNN()
    raise ValueError(f"Arquitetura desconhecida: {config['architecture']!r}")
