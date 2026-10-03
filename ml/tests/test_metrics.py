"""Tests des metriques Python (replication des definitions C++ QualityMetrics)."""

import math

import numpy as np
import pytest

from metrics import (
    C1,
    C2,
    MetricError,
    QualityMetrics,
    compute,
    compute_mse,
    compute_ssim,
    psnr_from_mse,
)


def pattern(width, height, channels, salt):
    ys = np.arange(height)
    xs = np.arange(width)
    img = np.zeros((height, width, channels), dtype=np.uint8)
    for y in ys:
        for x in xs:
            for c in range(channels):
                img[y, x, c] = (int(x) * 7 + int(y) * 13 + c * 29 + salt * 31) % 256
    return img


def constant(width, height, channels, value):
    return np.full((height, width, channels), value, dtype=np.uint8)


# --- Cas A : images identiques ---
def test_a_identical_images():
    ref = pattern(16, 16, 3, 1)
    metrics = compute(ref, ref)
    assert metrics.mse == 0.0
    assert metrics.identical is True
    assert math.isinf(metrics.psnr) and metrics.psnr > 0.0
    assert abs(metrics.ssim - 1.0) <= 1e-9


def test_a_single_channel_identical():
    ref = pattern(16, 16, 1, 7)
    metrics = compute(ref, ref)
    assert metrics.mse == 0.0
    assert abs(metrics.ssim - 1.0) <= 1e-9


# --- Cas B : difference connue (MSE verifie a la main) ---
def test_b_constant_offset_one():
    ref = constant(8, 8, 1, 100)
    cand = constant(8, 8, 1, 101)
    metrics = compute(ref, cand)
    assert abs(metrics.mse - 1.0) <= 1e-12
    assert abs(metrics.psnr - 10.0 * math.log10(255.0 * 255.0)) <= 1e-9
    assert metrics.identical is False


def test_b_hand_computed_mse():
    # ecarts 0, 2, 4, 6 -> (0 + 4 + 16 + 36) / 4 = 14
    ref = np.array([[0, 10], [20, 30]], dtype=np.uint8)[:, :, None]
    cand = np.array([[0, 8], [24, 24]], dtype=np.uint8)[:, :, None]
    metrics = compute(ref, cand)
    assert abs(metrics.mse - 14.0) <= 1e-12
    assert abs(metrics.psnr - 10.0 * math.log10(255.0 * 255.0 / 14.0)) <= 1e-9


# --- Cas C : images differentes ---
def test_c_different_images_are_sane():
    a = pattern(16, 16, 3, 1)
    b = pattern(16, 16, 3, 4)
    metrics = compute(a, b)
    assert metrics.mse > 0.0
    assert math.isfinite(metrics.psnr)
    assert math.isfinite(metrics.ssim)
    assert -1.0 - 1e-6 <= metrics.ssim <= 1.0 + 1e-6
    assert 0.0 <= metrics.psnr <= 100.0


def test_c_ssim_is_symmetric():
    a = pattern(16, 16, 3, 1)
    b = pattern(16, 16, 3, 4)
    assert abs(compute(a, b).ssim - compute(b, a).ssim) <= 1e-12


# --- SSIM analytique : conventions identiques au C++ ---
def test_ssim_analytic_constant_images():
    """Constantes 100 vs 110 : variances et covariance nulles -> valeur fermee."""
    ref = constant(8, 8, 1, 100)
    cand = constant(8, 8, 1, 110)
    expected = (2.0 * 100.0 * 110.0 + C1) / (100.0 * 100.0 + 110.0 * 110.0 + C1)
    assert abs(compute_ssim(ref, cand) - expected) <= 1e-6


def test_ssim_constants_match_cpp():
    assert C1 == 6.5025
    assert C2 == 58.5225


# --- psnr_from_mse : cas limites (identiques au C++) ---
def test_psnr_from_mse_edges():
    assert math.isinf(psnr_from_mse(0.0))
    assert abs(psnr_from_mse(65025.0) - 0.0) <= 1e-9
    assert abs(psnr_from_mse(1.0) - 10.0 * math.log10(65025.0)) <= 1e-9


# --- Cas D : entrees invalides ---
def test_d_incompatible_dimensions():
    ref = pattern(8, 8, 3, 1)
    bad = pattern(9, 8, 3, 1)
    with pytest.raises(MetricError):
        compute(ref, bad)


def test_d_incompatible_channels():
    ref = pattern(8, 8, 3, 1)
    bad = pattern(8, 8, 1, 1)
    with pytest.raises(MetricError):
        compute(ref, bad)


def test_d_empty_image():
    empty = np.zeros((0, 8, 3), dtype=np.uint8)
    ref = pattern(8, 8, 3, 1)
    with pytest.raises(MetricError):
        compute(empty, ref)
    with pytest.raises(MetricError):
        compute(ref, empty)


def test_d_non_finite_values():
    ref = pattern(8, 8, 3, 1).astype(np.float64)
    bad = ref.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(MetricError):
        compute(ref, bad)

    inf = ref.copy()
    inf[1, 1, 1] = np.inf
    with pytest.raises(MetricError):
        compute(ref, inf)


def test_d_metrics_untouched_on_error():
    """En cas d'erreur, l'objet QualityMetrics fourni n'est pas modifie.

    Equivalent Python du "reset" du C++ : l'API leve une exception explicite
    et laisse l'objet fourni intact.
    """
    ref = pattern(8, 8, 3, 1)
    metrics = QualityMetrics(mse=42.0, psnr=7.0, ssim=0.5, identical=True)
    with pytest.raises(MetricError):
        compute(ref, pattern(9, 8, 3, 1), metrics)
    assert metrics.mse == 42.0
    assert metrics.psnr == 7.0
    assert metrics.ssim == 0.5
    assert metrics.identical is True
