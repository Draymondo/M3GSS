"""Tests du pipeline de patches M3GSS v0 (HR 96x96 / LR 48x48, facteur 2).

Images synthetiques generees localement uniquement — aucun dataset externe.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from m3gss_v0.bicubic import (  # noqa: E402
    area_downscale,
    catmull_rom_upscale,
    make_test_image,
)
from m3gss_v0.model import M3GSS_v0_32x8  # noqa: E402
from m3gss_v0.patches import (  # noqa: E402
    PATCH_SIZE_HR,
    PATCH_SIZE_LR,
    PatchCoords,
    PatchValidationError,
    from_tensor,
    grid_coords,
    make_patch,
    sample_coords,
    to_tensor,
    validate_coords,
)

IMG_W, IMG_H = 256, 192
COORD = PatchCoords(16, 24)


@pytest.fixture(scope="module")
def image() -> np.ndarray:
    return make_test_image(width=IMG_W, height=IMG_H)


@pytest.fixture(scope="module")
def sample(image):
    return make_patch(image, COORD)


# --- Test A : un patch HR 96x96 donne bien un LR 48x48 ---
def test_a_lr_is_half_size(sample):
    assert sample.hr.shape == (PATCH_SIZE_HR, PATCH_SIZE_HR, 3)
    assert sample.lr.shape == (PATCH_SIZE_LR, PATCH_SIZE_LR, 3)


# --- Test B : le bicubique reconstruit fait 96x96 ---
def test_b_bicubic_restored_to_hr_size(sample):
    assert sample.bicubic.shape == (PATCH_SIZE_HR, PATCH_SIZE_HR, 3)


# --- Test C : les canaux restent RGB ---
def test_c_channels_stay_rgb(sample):
    for arr in (sample.hr, sample.lr, sample.bicubic):
        assert arr.shape[2] == 3


# --- Test D : les valeurs sont valides ---
def test_d_values_valid(sample):
    for name in ("hr", "lr", "bicubic"):
        arr = getattr(sample, name)
        assert arr.dtype == np.uint8, f"{name} : dtype {arr.dtype}"
        assert arr.min() >= 0 and arr.max() <= 255, f"{name} hors [0,255]"


# --- Test E : uint8 -> float32 -> uint8 sans aucune perte ---
def test_e_tensor_roundtrip_lossless():
    # image contenant les 256 niveaux dans chaque canal
    ramp = np.tile(np.arange(256, dtype=np.uint8), (2, 1))          # (2,256)
    img = np.stack([ramp, ramp[::-1], ramp], axis=-1)               # (2,256,3)

    tensor = to_tensor(img)
    assert tensor.shape == (3, 2, 256)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0

    back = from_tensor(tensor)
    assert np.array_equal(back, img), "conversion aller-retour non fidele"


# --- Test F : le pipeline est deterministe ---
def test_f_pipeline_is_deterministic(image):
    a = make_patch(image, COORD)
    b = make_patch(image, COORD)
    assert np.array_equal(a.hr, b.hr)
    assert np.array_equal(a.lr, b.lr)
    assert np.array_equal(a.bicubic, b.bicubic)
    assert a.coords == b.coords


def test_f_sampling_is_deterministic_by_index(image):
    for index in (0, 7, 123):
        assert sample_coords(index, IMG_W, IMG_H) == sample_coords(
            index, IMG_W, IMG_H
        )


# --- Test G : image trop petite rejetee proprement ---
def test_g_too_small_image_rejected():
    small = make_test_image(width=64, height=48)
    with pytest.raises(PatchValidationError) as exc:
        make_patch(small, PatchCoords(0, 0))
    assert "trop petite" in str(exc.value)

    with pytest.raises(PatchValidationError):
        grid_coords(95, 200)


# --- Test H : coordonnees invalides rejetees ---
@pytest.mark.parametrize(
    "coords",
    [
        PatchCoords(-2, 0),        # negatives
        PatchCoords(0, -2),
        PatchCoords(3, 0),         # non alignees sur le facteur 2
        PatchCoords(0, 5),
        PatchCoords(IMG_W, 0),     # hors borne horizontale
        PatchCoords(0, IMG_H),     # hors borne verticale
        PatchCoords(IMG_W - PATCH_SIZE_HR + 2, 0),  # deborde de 2 px
    ],
)
def test_h_invalid_coords_rejected(coords):
    with pytest.raises(PatchValidationError):
        validate_coords(coords, IMG_W, IMG_H)


def test_h_edge_patch_exactly_at_border_is_valid():
    validate_coords(PatchCoords(0, 0), IMG_W, IMG_H)
    validate_coords(PatchCoords(IMG_W - PATCH_SIZE_HR, IMG_H - PATCH_SIZE_HR),
                    IMG_W, IMG_H)


# --- Test I : le bicubique du pipeline == bicubic.py ---
def test_i_pipeline_matches_bicubic_module(sample):
    lr_ref = area_downscale(sample.hr, PATCH_SIZE_LR, PATCH_SIZE_LR)
    bic_ref = catmull_rom_upscale(lr_ref, PATCH_SIZE_HR, PATCH_SIZE_HR)
    assert np.array_equal(sample.lr, lr_ref)
    assert np.array_equal(sample.bicubic, bic_ref)


def test_i_patch_content_matches_image_region(image, sample):
    region = image[
        COORD.y : COORD.y + PATCH_SIZE_HR,
        COORD.x : COORD.x + PATCH_SIZE_HR,
    ]
    assert np.array_equal(sample.hr, region)


# --- Conversion tensor : formes et bornes ---
def test_tensor_conversion_shapes_and_range(image):
    sample_img = make_patch(image, COORD).bicubic
    t = to_tensor(sample_img)
    assert t.shape == (3, PATCH_SIZE_HR, PATCH_SIZE_HR)
    assert float(t.min()) >= 0.0
    assert float(t.max()) <= 1.0

    back = from_tensor(t)
    assert back.shape == sample_img.shape
    assert back.dtype == np.uint8


# --- Test d'integration : HR -> LR -> bicubique -> modele ---
def test_integration_model_with_zero_init_matches_bicubic(image):
    sample = make_patch(image, COORD)

    tensor = to_tensor(sample.bicubic).unsqueeze(0)     # [1,3,96,96]

    model = M3GSS_v0_32x8()
    model.eval()
    with torch.no_grad():
        out = model(tensor)

    assert out.shape == (1, 3, PATCH_SIZE_HR, PATCH_SIZE_HR)
    assert out.shape[1] == 3
    assert torch.isfinite(out).all()

    # tail zero-init -> sortie identique bit a bit au bicubique
    assert torch.equal(out, tensor), "sortie != bicubique a l'initialisation"

    residual = model.predict_residual(tensor)
    assert residual.abs().max().item() == 0.0
