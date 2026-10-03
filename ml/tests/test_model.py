"""Tests du modele M3GSS_v0_32x8 — aucun entrainement, uniquement des forwards."""

import pytest

torch = pytest.importorskip("torch")

from m3gss_v0.model import (  # noqa: E402
    BLOCKS,
    CHANNELS,
    M3GSS_v0_32x8,
    count_parameters,
    expected_parameter_count,
)


def test_model_instantiable():
    assert isinstance(M3GSS_v0_32x8(), M3GSS_v0_32x8)


def test_rgb_input_forward():
    model = M3GSS_v0_32x8()
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 3, 32, 32)


def test_output_same_resolution():
    model = M3GSS_v0_32x8()
    model.eval()
    for h, w in [(16, 16), (48, 64), (96, 96)]:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape, f"forme incoherente pour {h}x{w}"


def test_output_finite():
    model = M3GSS_v0_32x8()
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all()


def test_parameter_count_recomputed_by_code():
    """Le compte attendu est RECALCULE depuis l'architecture, pas recopie."""
    expected = expected_parameter_count(CHANNELS, BLOCKS)
    model = M3GSS_v0_32x8(channels=CHANNELS, blocks=BLOCKS)
    actual = count_parameters(model)

    assert actual == expected, (
        f"compte du modele {actual} != compte recalcule {expected}"
    )
    # et le total reste coherent avec la cible ~159 000
    assert abs(actual - 159_000) <= 1_000, f"compte inattendu : {actual}"


def test_expected_count_is_architecture_formula():
    """Le calcul attendu doit dependre de la configuration (non fige)."""
    base = expected_parameter_count(32, 8, 3)
    assert base == expected_parameter_count(32, 8, 3)
    # plus de canaux -> plus de parametres
    assert expected_parameter_count(64, 8, 3) > base
    # plus de blocs -> plus de parametres
    assert expected_parameter_count(32, 10, 3) > base


def test_no_batchnorm_no_attention():
    import torch.nn as nn

    model = M3GSS_v0_32x8()
    for module in model.modules():
        assert not isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm),
        ), f"couche de normalisation trouvee : {type(module).__name__}"


def _reference_forward(model, x, use_skip):
    """Recalcul independant du forward, avec ou sans skip global."""
    head = model.head_act(model.head_conv(x))
    feats = head
    for blk in model.blocks:
        feats = blk.conv2(blk.act(blk.conv1(feats))) + feats
    trunk = model.trunk(feats)
    merged = trunk + head if use_skip else trunk
    return x + model.tail(merged)


def test_global_skip_is_implemented_and_active():
    """Le skip global (head -> trunk) existe et modifie reellement le calcul.

    On rend le tail non nul pour observer le residu, puis on compare avec un
    recalcul independant AVEC et SANS skip global.
    """
    torch.manual_seed(1234)
    model = M3GSS_v0_32x8()
    model.eval()
    with torch.no_grad():
        model.tail.weight.normal_(0.0, 0.1)
        model.tail.bias.normal_(0.0, 0.1)

    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        out = model(x)
        ref_skip = _reference_forward(model, x, use_skip=True)
        ref_noskip = _reference_forward(model, x, use_skip=False)

    assert torch.allclose(out, ref_skip, atol=1e-6), (
        "la sortie ne correspond pas au calcul AVEC skip global"
    )
    assert not torch.allclose(out, ref_noskip, atol=1e-6), (
        "la sortie est identique SANS skip global : le skip n'agit pas"
    )


def test_zero_init_residual_is_null():
    """Propriete fondamentale : a l'init, sortie == bicubique (execution reelle)."""
    torch.manual_seed(0)
    model = M3GSS_v0_32x8()
    model.eval()

    bicubic = torch.randn(2, 3, 40, 40)
    with torch.no_grad():
        out = model(bicubic)
        residual = model.predict_residual(bicubic)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
    assert torch.allclose(out, bicubic, atol=1e-6)
    assert residual.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_zero_init_holds_for_several_shapes():
    model = M3GSS_v0_32x8()
    model.eval()
    for shape in [(1, 3, 8, 8), (3, 3, 64, 48), (1, 3, 96, 96)]:
        x = torch.randn(*shape)
        with torch.no_grad():
            out = model(x)
        assert torch.equal(out, x), f"sortie != entree pour {shape}"
