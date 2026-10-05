"""Tests d'integration du pipeline d'entrainement M3GSS V2."""

import random

import pytest
import torch

from m3gss_v0.model import M3GSS_v0_32x8
from m3gss_v0.synthetic import make_synthetic_image
from train import (
    DEFAULT_BEST,
    DEFAULT_CHECKPOINT,
    DEFAULT_V2_BEST,
    DEFAULT_V2_CHECKPOINT,
    M3GSSV1Loss,
    build_model,
    make_synthetic_batch_v2,
    make_training_pair,
    load_checkpoint,
    load_v2_checkpoint,
    save_checkpoint,
    save_v2_checkpoint,
    validate_v2_checkpoint_path,
)
from m3gss_v2.model import M3GSS_v2


def test_v1_default_selection_training_and_checkpoint_still_work(tmp_path):
    model = build_model("v1")
    assert isinstance(model, M3GSS_v0_32x8)

    hr_image = make_synthetic_image(11, size=96)
    bicubic, target = make_training_pair(hr_image, random.Random(3))
    output = model(bicubic.unsqueeze(0))
    loss, _, _ = M3GSSV1Loss()(output, target.unsqueeze(0))
    loss.backward()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "m3gss_v1_roundtrip.pt"
    save_checkpoint(checkpoint_path, model, optimizer, 2, float(loss.detach()), 17)
    reloaded_model = build_model("v1")
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-4)
    step, saved_loss, seed = load_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
        torch.device("cpu"),
    )

    assert output.shape == (1, 3, 96, 96)
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert step == 2
    assert saved_loss == float(loss.detach())
    assert seed == 17
    for original, reloaded in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.equal(original, reloaded)
    assert DEFAULT_CHECKPOINT.name == "m3gss_v1_big_latest.pt"
    assert DEFAULT_BEST.name == "m3gss_v1_big_best.pt"


def test_v2_synthetic_batch_forward_loss_and_backward():
    lr, baseline_hr, target_hr = make_synthetic_batch_v2(1, random.Random(7))
    model = build_model("v2")
    assert isinstance(model, M3GSS_v2)
    assert lr.shape == (1, 3, 48, 48)
    assert baseline_hr.shape == target_hr.shape == (1, 3, 96, 96)

    output = model(lr, baseline_hr)
    assert output.shape == (1, 3, 96, 96)
    loss, charbonnier, edge = M3GSSV1Loss()(output, target_hr)
    assert torch.isfinite(loss)
    assert torch.isfinite(charbonnier)
    assert torch.isfinite(edge)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.parameters()
    )


def test_v2_checkpoint_metadata_save_and_reload(tmp_path):
    model = build_model("v2")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    checkpoint_path = tmp_path / "m3gss_v2_test.pt"

    save_v2_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=0,
        loss=0.25,
        seed=42,
        dataset="synthetic",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["architecture"] == "M3GSS_v2"
    assert checkpoint["scale"] == 2
    assert checkpoint["patch_size_hr"] == 96
    assert checkpoint["patch_size_lr"] == 48
    assert checkpoint["loss_version"] == "v1_charbonnier_edge"
    assert checkpoint["num_parameters"] == 50_148
    assert checkpoint["dataset"] == "synthetic"
    assert checkpoint["step"] == 0
    assert "optimizer_state_dict" in checkpoint

    reloaded_model = build_model("v2")
    reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=1e-4)
    step, loss, seed = load_v2_checkpoint(
        checkpoint_path,
        reloaded_model,
        reloaded_optimizer,
        torch.device("cpu"),
    )
    assert (step, loss, seed) == (0, 0.25, 42)
    for original, reloaded in zip(model.parameters(), reloaded_model.parameters()):
        assert torch.equal(original, reloaded)


def test_v1_checkpoint_is_rejected_by_v2_loader(tmp_path):
    v1_model = build_model("v1")
    v1_path = tmp_path / "legacy_checkpoint.pt"
    torch.save({"state_dict": v1_model.state_dict()}, v1_path)

    v2_model = build_model("v2")
    optimizer = torch.optim.Adam(v2_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="architecture M3GSS_v2 requise"):
        load_v2_checkpoint(v1_path, v2_model, optimizer, torch.device("cpu"))
    with pytest.raises(ValueError, match="n'est pas M3GSS_v2"):
        validate_v2_checkpoint_path(v1_path)


def test_v2_paths_cannot_overwrite_v0_or_v1_checkpoints():
    assert DEFAULT_V2_CHECKPOINT.name == "m3gss_v2_latest.pt"
    assert DEFAULT_V2_BEST.name == "m3gss_v2_best.pt"
    protected_names = (
        "m3gss_v0_big_latest.pt",
        "m3gss_v0_big_best.pt",
        "m3gss_v0_div2k_500.pt",
        "m3gss_v1_big_latest.pt",
        "m3gss_v1_big_best.pt",
    )
    for name in protected_names:
        with pytest.raises(ValueError, match="protege contre V2"):
            validate_v2_checkpoint_path(DEFAULT_V2_CHECKPOINT.parent / name)