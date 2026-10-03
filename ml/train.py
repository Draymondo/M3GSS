"""M3GSS v0 — entrainement. ETAPE 4 : SMOKE TEST controle uniquement.

Commande :

    .\\.venv\\Scripts\\python.exe train.py --smoke-test

Objectif : prouver que la chaine complete fonctionne REELLEMENT :

    HR -> LR (area x2) -> bicubique Catmull-Rom -> M3GSS -> L1
        -> backward -> Adam -> mise a jour des poids

Contraintes respectees : CPU uniquement, aucun GPU requis, aucun
telechargement, aucun dataset externe, batch 1-2, ~10-20 steps.
L'entrainement reel (DIV2K / Flickr2K) n'est PAS implemente.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from m3gss_v0.model import M3GSS_v0_32x8, count_parameters
from m3gss_v0.synthetic import TrainingSample, build_smoke_dataset

__all__ = [
    "SmokeConfig",
    "SmokeResult",
    "set_seed",
    "compute_psnr",
    "run_smoke_test",
    "save_checkpoint",
    "load_checkpoint",
    "main",
]

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoints" / "smoke_test.pt"


@dataclass
class SmokeConfig:
    """Configuration du smoke test (toute petite, pour CPU)."""

    seed: int = 1234
    steps: int = 15
    batch_size: int = 1
    lr: float = 1e-4
    num_images: int = 4
    samples_per_image: int = 2
    patch_size: int = 96
    base_size: int = 192


@dataclass
class SmokeResult:
    steps: list = field(default_factory=list)
    losses: list = field(default_factory=list)
    residual_max: list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)
    initial_loss: float = 0.0
    final_loss: float = 0.0
    loss_delta: float = 0.0
    modified_parameters: int = 0
    total_parameters: int = 0
    total_parameter_tensors: int = 0
    val_psnr_before: float = 0.0
    val_psnr_after: float = 0.0
    elapsed_seconds: float = 0.0
    checkpoint: str = ""


def set_seed(seed: int) -> None:
    """Seed fixe pour Python, NumPy et PyTorch -> reproductibilite totale."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR en dB sur des tenseurs normalises [0,1] (MAX = 1.0)."""
    diff = pred.detach().double() - target.detach().double()
    mse = float(torch.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _batch_from(samples, indices) -> tuple[torch.Tensor, torch.Tensor]:
    bicubic = torch.stack([samples[i].bicubic for i in indices], dim=0)
    target = torch.stack([samples[i].target for i in indices], dim=0)
    return bicubic, target


def _grad_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().double().pow(2).sum())
    return math.sqrt(total)


def _validate_pre_step(model: nn.Module, bicubic: torch.Tensor) -> float:
    """GARDE-FOU AVANT le premier optimizer.step().

    Avec tail zero-init, la sortie doit etre strictement identique au bicubique
    et le residu strictement nul. Cela detecte une erreur d'architecture.
    """
    with torch.no_grad():
        out = model(bicubic)
        residual = model.predict_residual(bicubic)

    if not torch.equal(out, bicubic):
        raise RuntimeError(
            "GARDE-FOU : la sortie differe du bicubique avant entrainement "
            f"(ecart max {float((out - bicubic).abs().max()):.6e})"
        )
    if float(residual.abs().max()) != 0.0:
        raise RuntimeError(
            "GARDE-FOU : le residu initial n'est pas nul "
            f"({float(residual.abs().max()):.6e})"
        )
    return float(residual.abs().max())


def _validate_post_step(model, snapshot, bicubic, loss_value) -> float:
    """GARDE-FOU APRES le premier optimizer.step().

    Les parametres doivent avoir reellement change, le residu ne doit plus
    etre nul, et la loss doit etre finie.
    On n'exige PAS que la loss diminue.
    """
    changed = sum(
        1
        for name, param in model.named_parameters()
        if not torch.equal(param.detach(), snapshot[name])
    )
    if changed == 0:
        raise RuntimeError(
            "GARDE-FOU : aucun parametre n'a change apres optimizer.step()"
        )
    if not math.isfinite(loss_value):
        raise RuntimeError(f"GARDE-FOU : loss non finie ({loss_value})")

    with torch.no_grad():
        residual_max = float(model.predict_residual(bicubic).abs().max())
    if residual_max == 0.0:
        raise RuntimeError(
            "GARDE-FOU : le residu est encore nul apres une mise a jour des poids"
        )
    return residual_max


def run_smoke_test(cfg: SmokeConfig, checkpoint_path: Path | None = None) -> SmokeResult:
    """Execute le smoke test complet et renvoie toutes les mesures."""
    start = time.perf_counter()
    set_seed(cfg.seed)

    train_samples = build_smoke_dataset(
        num_images=cfg.num_images,
        samples_per_image=cfg.samples_per_image,
        patch_size=cfg.patch_size,
        base_size=cfg.base_size,
        seed=cfg.seed,
        image_offset=0,
    )
    # Validation synthetique DISJOINTE (images d'index differents)
    val_samples = build_smoke_dataset(
        num_images=1,
        samples_per_image=1,
        patch_size=cfg.patch_size,
        base_size=cfg.base_size,
        seed=cfg.seed + 1,
        image_offset=50,
    )
    val_bicubic, val_target = _batch_from(val_samples, [0])

    model = M3GSS_v0_32x8()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    l1 = nn.L1Loss()

    result = SmokeResult(
        total_parameters=count_parameters(model),
        total_parameter_tensors=len(list(model.named_parameters())),
    )

    first_bicubic, first_target = _batch_from(train_samples, [0])
    _validate_pre_step(model, first_bicubic)
    result.val_psnr_before = compute_psnr(model(val_bicubic), val_target)

    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}

    # NOTE zero-init : au 1er step, tail.poids est nul donc dL/d(merged) = 0 :
    # le gradient ne traverse PAS les couches en amont et seul le tail bouge.
    # Ce comportement est attendu (il vient du zero-init, pas d'un bug).
    # Au 2e step, tail n'etant plus nul, le gradient se propage normalement.
    n_train = len(train_samples)
    for step in range(1, cfg.steps + 1):
        indices = [
            (step * cfg.batch_size + k) % n_train for k in range(cfg.batch_size)
        ]
        bicubic, target = _batch_from(train_samples, indices)

        out = model(bicubic)
        loss = l1(out, target)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = _grad_norm(model)
        optimizer.step()

        with torch.no_grad():
            residual_max = float(model.predict_residual(bicubic).abs().max())

        if step == 1:
            residual_max = _validate_post_step(model, snapshot, bicubic, float(loss.detach()))
            result.initial_loss = float(loss.detach())

        result.steps.append(step)
        result.losses.append(float(loss.detach()))
        result.residual_max.append(residual_max)
        result.grad_norms.append(grad_norm)

        print(
            f"step {step:3d}/{cfg.steps} | "
            f"L1 {float(loss.detach()):.6f} | "
            f"residu max {residual_max:.6e} | "
            f"grad norm {grad_norm:.6e}"
        )

    result.final_loss = result.losses[-1]
    result.loss_delta = result.final_loss - result.initial_loss
    result.modified_parameters = sum(
        1
        for name, param in model.named_parameters()
        if not torch.equal(param.detach(), snapshot[name])
    )

    with torch.no_grad():
        result.val_psnr_after = compute_psnr(model(val_bicubic), val_target)

    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path, model, optimizer, cfg, result.final_loss, len(result.steps)
        )
        result.checkpoint = str(checkpoint_path)

    result.elapsed_seconds = time.perf_counter() - start
    return result


def save_checkpoint(path, model, optimizer, cfg, final_loss, step) -> None:
    """Sauvegarde le checkpoint final (dossier cree automatiquement)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "seed": cfg.seed,
            "config": asdict(cfg),
            "final_loss": final_loss,
        },
        path,
    )


def load_checkpoint(path):
    """Recharge un checkpoint sur CPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="M3GSS v0 - entrainement")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="smoke test CPU sur donnees synthetiques (seul mode implemente)",
    )
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH))
    args = parser.parse_args()

    if not args.smoke_test:
        print(
            "Aucun entrainement reel n'est implemente.\n"
            "Utilisez :  python train.py --smoke-test"
        )
        return 2

    cfg = SmokeConfig(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print("=== M3GSS v0 - SMOKE TEST (CPU, donnees synthetiques) ===")
    print(f"config : {asdict(cfg)}")
    print()

    result = run_smoke_test(cfg, Path(args.checkpoint))

    print()
    print("=== Bilan ===")
    print(f"loss initiale        : {result.initial_loss:.6f}")
    print(f"loss finale          : {result.final_loss:.6f}")
    print(f"variation loss       : {result.loss_delta:+.6f}")
    print(f"parametres modifies  : {result.modified_parameters}/{result.total_parameter_tensors} tenseurs ({result.total_parameters} elements)")

    print(f"PSNR val avant       : {result.val_psnr_before:.4f} dB")
    print(f"PSNR val apres       : {result.val_psnr_after:.4f} dB")
    print(f"temps total          : {result.elapsed_seconds:.3f} s")
    print(f"checkpoint           : {result.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
