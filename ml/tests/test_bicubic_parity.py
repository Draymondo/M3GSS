"""Parité C++ / Python du pipeline baseline (area reduction + bicubique).

Compare la sortie du vrai binaire C++ `M3GSS upscale` (BaselineBicubicUpscaler)
avec la reproduction Python de `m3gss_v0.bicubic` sur une image deterministe.

Metrique : dimensions, difference absolue maximale, erreur moyenne, MSE.
Cible : difference maximale <= 1 niveau sur 8 bits.

Si > 1, le test echoue et la cause doit etre identifiee (pas masquee).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from m3gss_v0.bicubic import baseline_upscale, make_test_image

# Resolutions du test (paires -> lround exact)
HR_W, HR_H = 64, 48

CANDIDATE_EXES = [
    Path(__file__).resolve().parents[2] / "build" / "Release" / "M3GSS.exe",
    Path(__file__).resolve().parents[2] / "build" / "Debug" / "M3GSS.exe",
]


def _find_exe() -> Path | None:
    for p in CANDIDATE_EXES:
        if p.is_file():
            return p
    return None


@pytest.fixture(scope="module")
def parity_results(tmp_path_factory) -> dict:
    """Tourne le C++ puis le Python sur la meme image et calcule l'ecart."""
    exe = _find_exe()
    if exe is None:
        pytest.skip("binaire M3GSS.exe introuvable (build C++ absent)")

    work = tmp_path_factory.mktemp("parity")
    in_path = work / "parity_in.png"
    out_cpp_path = work / "parity_out_cpp.png"

    # 1. image de test deterministique -> PNG
    hr = make_test_image(width=HR_W, height=HR_H)
    assert hr.shape == (HR_H, HR_W, 3)
    Image.fromarray(hr, mode="RGB").save(in_path)

    # 2. execution reelle du baseline C++
    proc = subprocess.run(
        [str(exe), "upscale", str(in_path), str(out_cpp_path), "0.5"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"M3GSS upscale a echoue (rc={proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert out_cpp_path.is_file(), "le C++ n'a pas produit la sortie PNG"

    out_cpp = np.asarray(Image.open(out_cpp_path).convert("RGB"), dtype=np.int16)

    # 3. reproduction Python (area reduction 2x puis bicubique)
    out_py = baseline_upscale(hr, factor=0.5).astype(np.int16)

    diff = np.abs(out_cpp - out_py)
    return {
        "hr": hr,
        "out_cpp": out_cpp,
        "out_py": out_py,
        "max_abs": int(diff.max()),
        "mean_abs": float(diff.mean()),
        "mse": float(np.mean(diff.astype(np.float64) ** 2)),
        "different_pixels": int(np.count_nonzero(diff)),
        "total_values": int(diff.size),
    }


def test_dimensions_identiques(parity_results):
    r = parity_results
    assert r["out_cpp"].shape == r["hr"].shape, (
        f"forme C++ {r['out_cpp'].shape} != HR {r['hr'].shape}"
    )
    assert r["out_py"].shape == r["hr"].shape, (
        f"forme Python {r['out_py'].shape} != HR {r['hr'].shape}"
    )
    assert r["out_cpp"].shape == r["out_py"].shape


def test_parity_max_abs_le_1(parity_results):
    r = parity_results
    print(
        "\n[PARIÉTÉ C++/Python] "
        f"max_abs={r['max_abs']}  mean_abs={r['mean_abs']:.6f}  "
        f"mse={r['mse']:.6f}  pixels_diff={r['different_pixels']}/{r['total_values']}"
    )
    assert r["max_abs"] <= 1, (
        f"parité > 1 niveau : max={r['max_abs']}, mean={r['mean_abs']:.6f}, "
        f"mse={r['mse']:.6f}, pixels_diff={r['different_pixels']}/{r['total_values']}. "
        "NE PAS relacher ce seuil sans identifier la cause "
        "(coordonnées / bords / arrondi / float vs double / conversion uint8)."
    )


def test_parity_mean_tres_faible(parity_results):
    r = parity_results
    # erreur moyenne tres basse (la plupart des pixels sont identiques)
    assert r["mean_abs"] <= 0.05, f"erreur moyenne trop elevee : {r['mean_abs']}"
