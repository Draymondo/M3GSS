"""Tests de l'infrastructure d'evaluation M3GSS v0.

Couvre : invariant zero-init, pipeline HR/LR/bicubique/modele, determinisme,
cross-check Python/C++ et chargement fonctionnel du checkpoint smoke.
"""

import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from evaluate import (
    build_zero_init_model,
    evaluate_image,
    evaluate_synthetic_set,
    load_model_from_checkpoint,
)
from m3gss_v0.bicubic import area_downscale, catmull_rom_upscale
from m3gss_v0.synthetic import make_synthetic_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_EXE = PROJECT_ROOT / "build" / "Release" / "M3GSS.exe"
SMOKE_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "smoke_test.pt"


# --- Zero-init : M3GSS(non entrainé) == bicubique, pixel par pixel ---
def test_zero_init_equals_bicubic_pixel_by_pixel():
    hr = make_synthetic_image(100, size=192)
    model = build_zero_init_model()

    bicubic = catmull_rom_upscale(area_downscale(hr, 96, 96), 192, 192)

    tensor = torch.from_numpy(
        np.ascontiguousarray(
            (bicubic.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1)
        )
    ).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    restored = np.round(
        np.clip(out.squeeze(0).numpy().transpose(1, 2, 0), 0.0, 1.0) * np.float32(255.0)
    ).astype(np.uint8)

    assert restored.shape == bicubic.shape
    assert restored.dtype == bicubic.dtype
    assert np.array_equal(restored, bicubic), (
        f"{int((restored != bicubic).sum())} pixels differents"
    )
    assert int((restored != bicubic).sum()) == 0


def test_zero_init_metrics_identical_to_baseline():
    """M3GSS zero-init produit exactement le bicubique.

    Consequence : les metriques de M3GSS vs HR sont IDENTIQUES a celles de la
    baseline. Elles ne valent PAS 0 : elles mesurent l'ecart au HR, pas l'ecart
    entre M3GSS et le bicubique (qui, lui, est nul — cf. test precedent).
    """
    hr = make_synthetic_image(101, size=192)
    evaluation = evaluate_image(hr, model=build_zero_init_model())

    assert evaluation.model.mse == evaluation.baseline.mse
    assert evaluation.model.psnr == evaluation.baseline.psnr
    assert evaluation.model.ssim == evaluation.baseline.ssim
    assert evaluation.baseline.mse > 0.0


# --- Pipeline d'evaluation ---
def test_evaluate_image_structure():
    hr = make_synthetic_image(102, size=192)
    evaluation = evaluate_image(hr, model=None, name="syn")

    assert evaluation.name == "syn"
    assert (evaluation.width, evaluation.height) == (192, 192)
    assert (evaluation.lr_width, evaluation.lr_height) == (96, 96)
    assert evaluation.model is None
    assert evaluation.baseline.mse > 0.0
    assert math.isfinite(evaluation.baseline.psnr)
    assert -1.0 <= evaluation.baseline.ssim <= 1.0


def test_baseline_is_worse_than_reference():
    """Le bicubique ne peut pas etre meilleur que l'original."""
    hr = make_synthetic_image(103, size=192)
    evaluation = evaluate_image(hr, model=None)
    assert evaluation.baseline.mse > 0.0
    assert evaluation.baseline.psnr < math.inf


# --- Determinisme ---
def test_evaluation_is_deterministic():
    hr = make_synthetic_image(104, size=192)
    first = evaluate_image(hr, model=build_zero_init_model())
    second = evaluate_image(hr, model=build_zero_init_model())
    assert first.baseline.mse == second.baseline.mse
    assert first.baseline.psnr == second.baseline.psnr
    assert first.baseline.ssim == second.baseline.ssim
    assert first.model.mse == second.model.mse
    assert first.model.ssim == second.model.ssim


def test_synthetic_set_is_deterministic():
    a = evaluate_synthetic_set(num_images=2, size=192, image_offset=100,
                               model=build_zero_init_model())
    b = evaluate_synthetic_set(num_images=2, size=192, image_offset=100,
                               model=build_zero_init_model())
    assert a.mean_baseline_mse == b.mean_baseline_mse
    assert a.mean_baseline_psnr == b.mean_baseline_psnr
    assert a.mean_baseline_ssim == b.mean_baseline_ssim
    assert a.mean_model_mse == b.mean_model_mse
    assert a.mean_model_ssim == b.mean_model_ssim
    assert [e.name for e in a.images] == [e.name for e in b.images]


# --- Cross-check Python / C++ ---
def _parse_cpp_metrics(stdout: str) -> dict:
    return {
        "mse": float(re.search(r"MSE\s+:\s+([0-9.eE+-]+)", stdout).group(1)),
        "psnr": float(re.search(r"PSNR\s+:\s+([0-9.eE+-]+)", stdout).group(1)),
        "ssim": float(re.search(r"SSIM\s+:\s+([0-9.eE+-]+)", stdout).group(1)),
    }


def test_cross_check_python_vs_cpp(tmp_path):
    """Compare les metriques Python et celles affichees par le binaire C++.

    Le binaire C++ imprime 6 chiffres significatifs (setprecision(6)) : la
    precision de la comparaison est donc limitee par l'affichage, pas par le
    calcul. On verifie aussi l'egalite pixel par pixel du bicubique.
    """
    if not CPP_EXE.is_file():
        pytest.skip("binaire C++ absent (build non realise)")

    hr = make_synthetic_image(100, size=192)
    in_path = tmp_path / "xcheck_in.png"
    out_path = tmp_path / "xcheck_out.png"
    Image.fromarray(hr, mode="RGB").save(in_path)

    proc = subprocess.run(
        [str(CPP_EXE), "upscale", str(in_path), str(out_path), "0.5"],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr

    bicubic = catmull_rom_upscale(area_downscale(hr, 96, 96), 192, 192)
    cpp_output = np.asarray(Image.open(out_path).convert("RGB"))

    # 1) le bicubique Python est identique au bicubique C++ (pixel par pixel)
    assert np.array_equal(cpp_output, bicubic)

    # 2) les metriques Python concordent avec celles affichees par le C++
    cpp = _parse_cpp_metrics(proc.stdout)
    py = evaluate_image(hr, model=None).baseline
    for name, py_value, cpp_value in (
        ("MSE", py.mse, cpp["mse"]),
        ("PSNR", py.psnr, cpp["psnr"]),
        ("SSIM", py.ssim, cpp["ssim"]),
    ):
        rel = abs(py_value - cpp_value) / max(abs(cpp_value), 1e-12)
        assert rel < 1e-5, f"{name} : python={py_value} cpp={cpp_value} rel={rel}"

    # 3) comparaison independante de l'affichage : metriques recalculees en
    #    Python sur la sortie C++ => identiques (images pixel-identiques)
    from metrics import compute
    recomputed = compute(hr, cpp_output)
    assert recomputed.mse == py.mse
    assert recomputed.psnr == py.psnr
    assert recomputed.ssim == py.ssim


# --- Checkpoint smoke : test FONCTIONNEL uniquement ---
def test_smoke_checkpoint_loads_and_infers():
    if not SMOKE_CKPT.is_file():
        pytest.skip("checkpoint smoke_test.pt absent")

    hr = make_synthetic_image(105, size=192)
    model = load_model_from_checkpoint(str(SMOKE_CKPT))

    evaluation = evaluate_image(hr, model=model, model_tag="smoke")
    assert evaluation.model is not None
    assert math.isfinite(evaluation.model.mse)
    assert math.isfinite(evaluation.model.psnr)
    assert -1.0 <= evaluation.model.ssim <= 1.0
    # test fonctionnel : l'inference fonctionne et est deterministe
    again = evaluate_image(hr, model=model, model_tag="smoke")
    assert again.model.mse == evaluation.model.mse
    assert again.model.ssim == evaluation.model.ssim


def test_smoke_checkpoint_differs_from_zero_init():
    """Le checkpoint n'est PAS identique au zero-init (il a ete entraine)."""
    if not SMOKE_CKPT.is_file():
        pytest.skip("checkpoint smoke_test.pt absent")

    hr = make_synthetic_image(105, size=192)
    zero = evaluate_image(hr, model=build_zero_init_model())
    trained = evaluate_image(hr, model=load_model_from_checkpoint(str(SMOKE_CKPT)))
    assert trained.model.mse != zero.model.mse
