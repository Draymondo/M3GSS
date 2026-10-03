"""Tests du pipeline d'entrainement M3GSS v0 (smoke test CPU).

Ces tests ne dependent d'aucune amelioration visuelle : ils verifient que la
chaine HR -> LR -> bicubique -> modele -> L1 -> backward -> Adam fonctionne.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from m3gss_v0.model import M3GSS_v0_32x8
from m3gss_v0.synthetic import build_smoke_dataset, make_synthetic_image
from train import (
    SmokeConfig,
    load_checkpoint,
    run_smoke_test,
    save_checkpoint,
    set_seed,
)


@pytest.fixture(scope="module")
def samples():
    return build_smoke_dataset(num_images=2, samples_per_image=2, seed=1)


# --- A. dataset synthetique deterministe ---
def test_a_synthetic_images_are_deterministic():
    a = make_synthetic_image(3, size=192)
    b = make_synthetic_image(3, size=192)
    assert np.array_equal(a, b)
    assert a.dtype == np.uint8
    assert a.shape == (192, 192, 3)


def test_a_synthetic_dataset_is_deterministic():
    s1 = build_smoke_dataset(num_images=2, samples_per_image=2, seed=7)
    s2 = build_smoke_dataset(num_images=2, samples_per_image=2, seed=7)
    assert len(s1) == len(s2) == 4
    for x, y in zip(s1, s2):
        assert x.name == y.name
        assert x.coords == y.coords
        assert torch.equal(x.bicubic, y.bicubic)
        assert torch.equal(x.target, y.target)


def test_a_different_index_gives_different_image():
    assert not np.array_equal(
        make_synthetic_image(0, size=192), make_synthetic_image(1, size=192)
    )


# --- B. dimensions correctes ---
def test_b_dataset_shapes(samples):
    for s in samples:
        assert s.bicubic.shape == (3, 96, 96)
        assert s.target.shape == (3, 96, 96)
        assert s.bicubic.dtype == torch.float32
        assert float(s.bicubic.min()) >= 0.0
        assert float(s.bicubic.max()) <= 1.0
        assert s.coords.x % 2 == 0 and s.coords.y % 2 == 0


# --- C. forward fonctionnel ---
def test_c_forward_works(samples):
    set_seed(0)
    sample = samples[0]
    model = M3GSS_v0_32x8()
    model.eval()
    with torch.no_grad():
        out = model(sample.bicubic.unsqueeze(0))
    assert out.shape == (1, 3, 96, 96)
    assert torch.isfinite(out).all()


# --- D. loss finie ---
def test_d_loss_is_finite(samples):
    set_seed(0)
    sample = samples[0]
    model = M3GSS_v0_32x8()
    out = model(sample.bicubic.unsqueeze(0))
    loss = nn.L1Loss()(out, sample.target.unsqueeze(0))
    assert math.isfinite(float(loss.detach()))
    assert float(loss.detach()) >= 0.0


# --- E + F. backward et gradients non nuls ---
def test_e_f_backward_and_gradients(samples):
    set_seed(0)
    sample = samples[0]
    model = M3GSS_v0_32x8()
    out = model(sample.bicubic.unsqueeze(0))
    loss = nn.L1Loss()(out, sample.target.unsqueeze(0))
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "aucun gradient calcule"
    total_norm = math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads))
    assert total_norm > 0.0, "gradients tous nuls"
    assert all(torch.isfinite(g).all() for g in grads)


# --- G. optimizer.step() modifie les parametres ---
def test_g_optimizer_step_changes_parameters(samples):
    """optimizer.step() modifie reellement les poids.

    Comportement attendu, propre au zero-init : au point d'initialisation le
    gradient qui traverse la derniere convolution est exactement nul (ses poids
    sont nuls), donc SEUL le tail bouge au 1er step. Ce n'est pas un bug mais la
    consequence directe de tail zero-init. Au 2e step, tail n'etant plus nul,
    le gradient se propage aux couches en amont.
    """
    set_seed(0)
    sample = samples[0]
    model = M3GSS_v0_32x8()
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def one_step() -> None:
        optimizer.zero_grad()
        out = model(sample.bicubic.unsqueeze(0))
        loss = nn.L1Loss()(out, sample.target.unsqueeze(0))
        loss.backward()
        optimizer.step()

    def changed_count() -> int:
        return sum(
            1
            for name, p in model.named_parameters()
            if not torch.equal(p.detach(), snapshot[name])
        )

    one_step()
    first = changed_count()
    assert first >= 1, "aucun parametre modifie au 1er step"
    assert not torch.equal(snapshot["tail.weight"], model.tail.weight.detach())

    one_step()
    second = changed_count()
    assert second > first, (
        f"gradient non propage apres le 2e step : {first} -> {second} tenseurs"
    )


# --- H + I + J. checkpoint ---
def _make_trained_model_and_optimizer(samples):
    set_seed(0)
    sample = samples[0]
    model = M3GSS_v0_32x8()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    out = model(sample.bicubic.unsqueeze(0))
    loss = nn.L1Loss()(out, sample.target.unsqueeze(0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model, optimizer, float(loss.detach())


def test_h_i_j_checkpoint_roundtrip(samples, tmp_path):
    model, optimizer, loss_value = _make_trained_model_and_optimizer(samples)

    # H : sauvegardable (le dossier parent est cree automatiquement)
    path = tmp_path / "ckpt_dir" / "smoke.pt"
    cfg = SmokeConfig(seed=42, steps=2)
    save_checkpoint(path, model, optimizer, cfg, loss_value, step=2)
    assert path.is_file()

    # I : rechargeable
    ckpt = load_checkpoint(path)
    for key in (
        "state_dict",
        "optimizer_state_dict",
        "step",
        "seed",
        "config",
        "final_loss",
    ):
        assert key in ckpt, f"cle manquante : {key}"
    assert ckpt["step"] == 2
    assert ckpt["seed"] == 42
    assert math.isfinite(ckpt["final_loss"])

    # J : le modele recharge produit des valeurs finies
    reloaded = M3GSS_v0_32x8()
    reloaded.load_state_dict(ckpt["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        out = reloaded(samples[0].bicubic.unsqueeze(0))
    assert out.shape == (1, 3, 96, 96)
    assert torch.isfinite(out).all()


# --- Garde-fous d'architecture ---
def test_pre_step_guard_zero_init(samples):
    """Avant entrainement : sortie == bicubique, residu nul (tail zero-init)."""
    model = M3GSS_v0_32x8()
    model.eval()
    bic = samples[0].bicubic.unsqueeze(0)
    with torch.no_grad():
        out = model(bic)
    assert torch.equal(out, bic)
    with torch.no_grad():
        assert float(model.predict_residual(bic).abs().max()) == 0.0


def test_post_step_residual_becomes_non_zero(samples):
    """Apres une mise a jour : le residu n'est plus nul (le reseau a appris)."""
    model, _optimizer, _loss = _make_trained_model_and_optimizer(samples)
    bic = samples[0].bicubic.unsqueeze(0)
    with torch.no_grad():
        residual = model.predict_residual(bic)
    assert float(residual.abs().max()) > 0.0


# --- Integration : le smoke test complet ---
def test_smoke_test_runs_end_to_end(tmp_path):
    cfg = SmokeConfig(seed=99, steps=3, batch_size=1, num_images=2,
                      samples_per_image=1)
    result = run_smoke_test(cfg, tmp_path / "run.pt")

    assert len(result.losses) == 3
    assert len(result.residual_max) == 3
    assert len(result.grad_norms) == 3
    assert all(math.isfinite(v) for v in result.losses)
    assert all(v > 0.0 for v in result.grad_norms)
    assert math.isfinite(result.initial_loss)
    assert math.isfinite(result.final_loss)
    assert result.modified_parameters == result.total_parameter_tensors > 0
    assert result.total_parameters == 158_979
    assert result.elapsed_seconds > 0.0
    assert (tmp_path / "run.pt").is_file()
