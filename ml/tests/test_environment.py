"""Tests d'environnement M3GSS (Phase 4) - aucun entrainement, aucun dataset.

Verifie uniquement que la chaine d'outils ML est utilisable sur cette machine :
imports, tenseurs, convolution, detection CPU/GPU et versions.
"""

import importlib

import pytest


def test_import_numpy():
    import numpy as np

    assert np.__version__


def test_import_pillow():
    from PIL import Image

    assert Image is not None


def test_import_pytorch():
    torch = pytest.importorskip("torch", reason="PyTorch non installe dans ce venv")
    assert torch.__version__


def test_tensor_creation():
    torch = pytest.importorskip("torch")
    t = torch.zeros(2, 3, 4, 4)
    assert tuple(t.shape) == (2, 3, 4, 4)
    assert t.dtype == torch.float32
    assert float(t.sum()) == 0.0


def test_small_convolution():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    x = torch.randn(1, 3, 16, 16)
    y = conv(x)
    assert tuple(y.shape) == (1, 8, 16, 16)
    assert torch.isfinite(y).all()


def test_cpu_detection():
    torch = pytest.importorskip("torch")
    device = torch.device("cpu")
    x = torch.ones(4, device=device)
    assert x.device.type == "cpu"


def test_gpu_detection():
    """Presence GPU : informationnelle (iGPU Intel = non exploitable pour CUDA)."""
    torch = pytest.importorskip("torch")
    print("cuda_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    # Ne echoue pas : le T470s n'a pas de GPU NVIDIA, c'est attendu.
    assert isinstance(torch.cuda.is_available(), bool)


def test_print_versions():
    torch = pytest.importorskip("torch")
    import numpy as np
    import PIL

    print("torch:", torch.__version__)
    print("numpy:", np.__version__)
    print("pillow:", PIL.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    assert True


def test_m3gss_v0_package_importable():
    module = importlib.import_module("m3gss_v0")
    assert module is not None
