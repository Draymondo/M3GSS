"""Metriques de qualite M3GSS — replication exacte de `src/metrics/QualityMetrics.cpp`.

Les definitions sont STRICTEMENT identiques a celles du C++ :
  * MSE   : moyenne des carres d'ecart sur tous les octets (tous canaux)
  * PSNR  : 10 * log10(255^2 / MSE), MAX = 255 (images 8 bits),
            +infini si MSE == 0 (cas explicite, images identiques)
  * SSIM  : Wang, Bovik, Sheikh & Simoncelli (2004)
            - fenetre gaussienne 11x11, sigma = 1.5
            - C1 = (0.01*255)^2 = 6.5025, C2 = (0.03*255)^2 = 58.5225
            - convolution separable (horizontale puis verticale), bords reproduits
            - variance = max(0, E[x^2] - mu^2)
            - moyenne globale sur la region valide et tous les canaux
            - fenetre retrecie si l'image est plus petite (au moins 1 position)

Fideltite numerique : les cumuls sont effectues en float64 dans le MEME ordre
que le C++ (parcours des taps k croissant, puis y croissant, puis x croissant),
afin d'obtenir des valeurs identiques bit a bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "QualityMetrics",
    "MetricError",
    "psnr_from_mse",
    "compute_mse",
    "compute_ssim",
    "compute",
    "C1",
    "C2",
    "SSIM_SIGMA",
    "SSIM_MAX_RADIUS",
]

# Constantes identiques au C++ (QualityMetrics.cpp)
C1 = 6.5025
C2 = 58.5225
SSIM_SIGMA = 1.5
SSIM_MAX_RADIUS = 5


class MetricError(ValueError):
    """Metriques impossibles a calculer (images incompatibles ou invalides)."""


@dataclass
class QualityMetrics:
    """Resultat structure, aligne sur m3gss::metrics::QualityMetrics."""

    mse: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    identical: bool = False


def _as_float_image(image: np.ndarray, name: str) -> np.ndarray:
    """Valide et convertit en float64 [H, W, C]."""
    arr = np.asarray(image)
    if arr.size == 0:
        raise MetricError(f"{name} : image vide")
    if arr.ndim != 3:
        raise MetricError(f"{name} : attendu [H,W,C], recu {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1 or arr.shape[2] < 1:
        raise MetricError(f"{name} : dimension nulle ({arr.shape})")
    out = arr.astype(np.float64)
    if not np.isfinite(out).all():
        raise MetricError(f"{name} : donnees non finies")
    return out


def psnr_from_mse(mse: float) -> float:
    """PSNR en dB a partir du MSE. MSE <= 0 -> +infini (images identiques)."""
    if mse <= 0.0:
        return math.inf
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def compute_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Erreur quadratique moyenne sur tous les octets (tous canaux)."""
    ref = _as_float_image(reference, "reference")
    cand = _as_float_image(candidate, "candidate")
    if ref.shape != cand.shape:
        raise MetricError(
            f"dimensions/canaux incompatibles : {ref.shape} vs {cand.shape}"
        )
    # Les carres d'ecart d'images 8 bits sont des entiers : la somme reste
    # exacte en float64 quel que soit l'ordre de sommation.
    diff = ref - cand
    return float(np.sum(diff * diff) / diff.size)


def _gauss_kernel(radius: int) -> list[float]:
    kernel = []
    total = 0.0
    for t in range(-radius, radius + 1):
        weight = math.exp(-(float(t) * float(t)) / (2.0 * SSIM_SIGMA * SSIM_SIGMA))
        kernel.append(weight)
        total += weight
    return [w / total for w in kernel]


def _sep_gaussian(data: np.ndarray, scratch: np.ndarray, width: int,
                  height: int, kernel: list[float]) -> None:
    """Convolution gaussienne separable, bords reproduits, ordre C++.

    Passe horizontale (data -> scratch) puis verticale (scratch -> data).
    """
    radius = (len(kernel) - 1) // 2

    # horizontale
    for y in range(height):
        row = data[y * width : (y + 1) * width]
        out = np.zeros(width, dtype=np.float64)
        for k in range(-radius, radius + 1):
            idx = np.clip(np.arange(width) + k, 0, width - 1)
            out += kernel[k + radius] * row[idx]
        scratch[y * width : (y + 1) * width] = out

    # verticale
    for x in range(width):
        col = scratch[x::width]
        res = np.zeros(height, dtype=np.float64)
        for k in range(-radius, radius + 1):
            idx = np.clip(np.arange(height) + k, 0, height - 1)
            res += kernel[k + radius] * col[idx]
        data[x::width] = res


def compute_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """SSIM global (Wang 2004), conventions identiques au C++."""
    ref = _as_float_image(reference, "reference")
    cand = _as_float_image(candidate, "candidate")
    if ref.shape != cand.shape:
        raise MetricError(
            f"dimensions/canaux incompatibles : {ref.shape} vs {cand.shape}"
        )

    height, width, channels = ref.shape
    plane_size = width * height

    radius = min(SSIM_MAX_RADIUS, (min(width, height) - 1) // 2)
    kernel = _gauss_kernel(radius)

    plane_x = np.zeros(plane_size, dtype=np.float64)
    plane_y = np.zeros(plane_size, dtype=np.float64)
    moment_x2 = np.zeros(plane_size, dtype=np.float64)
    moment_y2 = np.zeros(plane_size, dtype=np.float64)
    moment_xy = np.zeros(plane_size, dtype=np.float64)
    scratch = np.zeros(plane_size, dtype=np.float64)

    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    ssim_sum = 0.0
    sample_count = 0

    for c in range(channels):
        plane_x[:] = ref_flat[c::channels]
        plane_y[:] = cand_flat[c::channels]
        np.square(plane_x, out=moment_x2)
        np.square(plane_y, out=moment_y2)
        np.multiply(plane_x, plane_y, out=moment_xy)

        _sep_gaussian(plane_x, scratch, width, height, kernel)
        _sep_gaussian(moment_x2, scratch, width, height, kernel)
        _sep_gaussian(plane_y, scratch, width, height, kernel)
        _sep_gaussian(moment_y2, scratch, width, height, kernel)
        _sep_gaussian(moment_xy, scratch, width, height, kernel)

        for y in range(radius, height - radius):
            base = y * width
            for x in range(radius, width - radius):
                i = base + x
                mu_x = plane_x[i]
                mu_y = plane_y[i]
                var_x = max(0.0, moment_x2[i] - mu_x * mu_x)
                var_y = max(0.0, moment_y2[i] - mu_y * mu_y)
                covariance = moment_xy[i] - mu_x * mu_y

                numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * covariance + C2)
                denominator = (
                    mu_x * mu_x + mu_y * mu_y + C1
                ) * (var_x + var_y + C2)

                ssim_sum += numerator / denominator
                sample_count += 1

    if sample_count == 0:
        return 0.0
    return ssim_sum / float(sample_count)


def compute(reference: np.ndarray, candidate: np.ndarray,
            metrics: QualityMetrics | None = None) -> QualityMetrics:
    """Calcule MSE, PSNR et SSIM (API proche de m3gss::metrics::compute).

    Leve MetricError si les images sont incompatibles ou invalides.
    """
    result = metrics if metrics is not None else QualityMetrics()
    result.mse = compute_mse(reference, candidate)
    result.identical = result.mse == 0.0
    result.psnr = psnr_from_mse(result.mse)
    result.ssim = compute_ssim(reference, candidate)
    return result
