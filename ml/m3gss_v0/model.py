"""Modele M3GSS_v0_32x8 — CNN residuel leger de reconstruction x2.

Architecture EXACTE (verrouillee, aucun BatchNorm / attention / PixelShuffle) :

    entree bicubique RGB [N,3,H,W]
      -> Conv2d(3,32,3,pad=1) + LeakyReLU(0.1)          # head
      -> 8 x bloc residuel : Conv3x3 -> LeakyReLU -> Conv3x3
                              sortie_bloc = entree_bloc + F(entree_bloc)   # skip local
      -> Conv2d(32,32,3,pad=1)                           # trunk
      -> + sortie du head                                # SKIP GLOBAL (explicite)
      -> Conv2d(32,3,3,pad=1)  initialise ZERO           # tail (residu)
      -> sortie = bicubique + residu

Le skip global relie la sortie du head a la sortie du trunk : chemin direct de
l'entree vers la sortie du reseau (stabilise l'entrainement et conserve la
pleine plage dynamique, sans avoir besoin de BatchNorm).

A l'initialisation, la derniere convolution etant nulle, le residu est nul :
la sortie est exactement egale a l'entree bicubique, bit a bit.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "M3GSS_v0_32x8",
    "ResidualBlock",
    "count_parameters",
    "expected_parameter_count",
    "build_model",
    "CHANNELS",
    "BLOCKS",
    "IN_CHANNELS",
]

CHANNELS = 32
BLOCKS = 8
IN_CHANNELS = 3
KERNEL = 3


def _conv_params(in_ch: int, out_ch: int) -> int:
    """Parametres d'une Conv2d (poids + biais) — CALCULES, jamais recopies."""
    return out_ch * in_ch * KERNEL * KERNEL + out_ch


def expected_parameter_count(
    channels: int = CHANNELS,
    blocks: int = BLOCKS,
    in_channels: int = IN_CHANNELS,
) -> int:
    """Nombre de parametres attendu, RECALCULE depuis l'architecture.

    head + 8 blocs (2 convs chacun) + trunk + tail.
    Le skip global est une simple addition : il n'ajoute AUCUN parametre.
    """
    total = _conv_params(in_channels, channels)              # head
    total += blocks * 2 * _conv_params(channels, channels)  # blocs residuels
    total += _conv_params(channels, channels)               # trunk
    total += _conv_params(channels, in_channels)           # tail
    return total


class ResidualBlock(nn.Module):
    """Bloc residuel sans BatchNorm : Conv -> LeakyReLU -> Conv, + skip local."""

    def __init__(self, channels: int = CHANNELS) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=KERNEL, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=KERNEL, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class M3GSS_v0_32x8(nn.Module):
    """CNN residuel leger : apprend la correction residuelle au bicubique.

    Entree : image bicubique RGB [N, 3, H, W]
    Sortie : bicubique + residu predit, meme forme [N, 3, H, W]
    """

    def __init__(self, channels: int = CHANNELS, blocks: int = BLOCKS,
                 in_channels: int = IN_CHANNELS) -> None:
        super().__init__()
        self.channels = channels
        self.blocks_count = blocks

        self.head_conv = nn.Conv2d(in_channels, channels, kernel_size=KERNEL, padding=1)
        self.head_act = nn.LeakyReLU(negative_slope=0.1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels=channels) for _ in range(blocks)]
        )
        self.trunk = nn.Conv2d(channels, channels, kernel_size=KERNEL, padding=1)
        self.tail = nn.Conv2d(channels, in_channels, kernel_size=KERNEL, padding=1)

        self._zero_init_tail()

    def _zero_init_tail(self) -> None:
        """Derniere convolution nulle (poids ET biais) -> residu nul a l'init."""
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, bicubic: torch.Tensor) -> torch.Tensor:
        if bicubic.dim() != 4 or bicubic.size(1) != IN_CHANNELS:
            raise ValueError(
                f"entree attendue [N,{IN_CHANNELS},H,W], recu {tuple(bicubic.shape)}"
            )

        head_feat = self.head_act(self.head_conv(bicubic))

        feats = head_feat
        for block in self.blocks:
            feats = block(feats)

        trunk_feat = self.trunk(feats)

        # SKIP GLOBAL : sortie du head -> sortie du trunk (addition explicite).
        merged = trunk_feat + head_feat

        residual = self.tail(merged)
        return bicubic + residual

    def predict_residual(self, bicubic: torch.Tensor) -> torch.Tensor:
        """Diagnostic : renvoie uniquement le residu predit."""
        return self.forward(bicubic) - bicubic


def count_parameters(model: nn.Module) -> int:
    """Nombre de parametres entrainables (tenseurs)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model() -> M3GSS_v0_32x8:
    """Fabrique la reference M3GSS_v0_32x8 (32 canaux, 8 blocs)."""
    return M3GSS_v0_32x8(channels=CHANNELS, blocks=BLOCKS)
