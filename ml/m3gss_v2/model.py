"""Modele experimental M3GSS V2 : reconstruction residuelle x2 depuis la LR.

Toutes les convolutions operent a la resolution d'entree. PixelShuffle
reorganise les 12 canaux de la tete en une correction RGB haute resolution,
ajoutee a une baseline Catmull-Rom fournie par l'appelant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["M3GSS_v2", "ResidualBlock", "count_parameters"]

IN_CHANNELS = 3
CHANNELS = 24
BLOCKS = 4
SCALE_FACTOR = 2
KERNEL_SIZE = 3


class ResidualBlock(nn.Module):
    """Bloc residuel 24 canaux : Conv -> LeakyReLU -> Conv, plus skip local."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, KERNEL_SIZE, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, KERNEL_SIZE, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class M3GSS_v2(nn.Module):
    """CNN LR residuel x2 avec baseline HR Catmull-Rom externe.

    Entrees :
        lr: [N, 3, H, W], image basse resolution RGB.
        baseline_hr: [N, 3, 2H, 2W], baseline Catmull-Rom correspondante.

    Sortie : baseline_hr + residu appris, de forme [N, 3, 2H, 2W].
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, CHANNELS, KERNEL_SIZE, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.blocks = nn.ModuleList(ResidualBlock() for _ in range(BLOCKS))
        self.trunk = nn.Conv2d(CHANNELS, CHANNELS, KERNEL_SIZE, padding=1)
        self.reconstruction = nn.Conv2d(
            CHANNELS, IN_CHANNELS * SCALE_FACTOR**2, KERNEL_SIZE, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(SCALE_FACTOR)

        nn.init.zeros_(self.reconstruction.weight)
        nn.init.zeros_(self.reconstruction.bias)

    def _validate_lr(self, lr: torch.Tensor) -> None:
        if lr.dim() != 4 or lr.size(1) != IN_CHANNELS:
            raise ValueError(
                f"entree LR attendue [N,{IN_CHANNELS},H,W], recue {tuple(lr.shape)}"
            )

    def _forward_residual(self, lr: torch.Tensor) -> torch.Tensor:
        stem_features = self.stem(lr)
        features = stem_features
        for block in self.blocks:
            features = block(features)

        features = self.trunk(features) + stem_features
        return self.pixel_shuffle(self.reconstruction(features))

    def predict_residual(self, lr: torch.Tensor) -> torch.Tensor:
        """Renvoie le residu RGB HR predit, sans la baseline externe."""
        self._validate_lr(lr)
        return self._forward_residual(lr)

    def forward(self, lr: torch.Tensor, baseline_hr: torch.Tensor) -> torch.Tensor:
        self._validate_lr(lr)
        expected_shape = (
            lr.size(0),
            IN_CHANNELS,
            lr.size(2) * SCALE_FACTOR,
            lr.size(3) * SCALE_FACTOR,
        )
        if tuple(baseline_hr.shape) != expected_shape:
            raise ValueError(
                f"baseline HR attendue {expected_shape}, recue {tuple(baseline_hr.shape)}"
            )
        return baseline_hr + self._forward_residual(lr)


def count_parameters(model: nn.Module) -> int:
    """Compte les parametres entrainables du modele."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _self_test() -> None:
    torch.manual_seed(0)
    model = M3GSS_v2()

    batch, height, width = 2, 12, 16
    lr = torch.rand(batch, IN_CHANNELS, height, width)
    baseline_hr = torch.rand(
        batch,
        IN_CHANNELS,
        height * SCALE_FACTOR,
        width * SCALE_FACTOR,
    )
    target_hr = torch.rand_like(baseline_hr)

    with torch.no_grad():
        output = model(lr, baseline_hr)
        residual = model.predict_residual(lr)

    expected_output_shape = (batch, IN_CHANNELS, height * 2, width * 2)
    assert tuple(output.shape) == expected_output_shape
    assert tuple(residual.shape) == expected_output_shape
    assert torch.equal(output, baseline_hr)
    assert torch.equal(residual, torch.zeros_like(residual))

    loss = F.l1_loss(model(lr, baseline_hr), target_hr)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)

    parameter_count = count_parameters(model)
    assert parameter_count == 50_148

    print("M3GSS_v2 self-test: forward/backward OK")
    print(f"LR             : {tuple(lr.shape)}")
    print(f"Baseline HR    : {tuple(baseline_hr.shape)}")
    print(f"Output HR      : {tuple(output.shape)}")
    print(f"Parametres     : {parameter_count:,}")


if __name__ == "__main__":
    _self_test()