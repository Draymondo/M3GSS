from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from m3gss_v0.bicubic import area_downscale, catmull_rom_upscale
from m3gss_v0.model import M3GSS_v0_32x8, count_parameters
from m3gss_v2.model import M3GSS_v2, count_parameters as count_parameters_v2


ROOT = Path(__file__).resolve().parent

DIV2K = Path(r"D:\M3GSS_OFFLINE\datasets\DIV2K\train_HR")
FLICKR2K = Path(r"D:\M3GSS_OFFLINE\datasets\Flickr2K\HR\Flickr2K")

# Noms V1 : un run V1 ne doit jamais ecraser un checkpoint V0.
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "m3gss_v1_big_latest.pt"
DEFAULT_BEST = ROOT / "checkpoints" / "m3gss_v1_big_best.pt"
DEFAULT_V2_CHECKPOINT = ROOT / "checkpoints" / "m3gss_v2_latest.pt"
DEFAULT_V2_BEST = ROOT / "checkpoints" / "m3gss_v2_best.pt"
DEFAULT_V2_SMOKE_CHECKPOINT = ROOT / "checkpoints" / "m3gss_v2_smoke.pt"

PATCH_SIZE = 96
SCALE = 2
SAVE_EVERY = 500

# ---------------------------------------------------------------------------
# Pertes M3GSS V1 -- constantes faciles a modifier
# ---------------------------------------------------------------------------
CHARBONNIER_EPS = 1e-3   # stabilise sqrt(...) autour de 0 (valeur usuelle en SR)
EDGE_WEIGHT = 0.1         # poids faible : la loss reste principalement fidele
LOSS_VERSION = "v1_charbonnier_edge"


class CharbonnierLoss(nn.Module):
    """Charbonnier : sqrt((pred - target)^2 + eps^2), differentiable partout."""

    def __init__(
        self,
        eps: float = CHARBONNIER_EPS,
    ) -> None:

        super().__init__()

        self.eps_sq = float(eps) ** 2

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        diff = prediction - target

        return torch.sqrt(
            diff * diff + self.eps_sq
        ).mean()


class EdgeGradientLoss(nn.Module):
    """L1 sur les gradients horizontaux et verticaux (differences finies).

    Les bords sont ignores (tranches 1:) : pas de bord torique, aucune
    dependance, differentiable et rapide sur CPU comme sur CUDA.
    """

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if (
            prediction.shape[-2] < 2
            or prediction.shape[-1] < 2
        ):

            return (
                prediction - target
            ).abs().mean() * 0.0

        pred_gx = prediction[..., :, 1:] - prediction[..., :, :-1]
        pred_gy = prediction[..., 1:, :] - prediction[..., :-1, :]

        targ_gx = target[..., :, 1:] - target[..., :, :-1]
        targ_gy = target[..., 1:, :] - target[..., :-1, :]

        return (
            (pred_gx - targ_gx).abs().mean()
            + (pred_gy - targ_gy).abs().mean()
        )


class M3GSSV1Loss(nn.Module):
    """Loss V1 : Charbonnier + EDGE_WEIGHT * Edge/Gradient.

    Retourne (total, charbonnier, edge) pour l'affichage et le suivi.
    """

    def __init__(
        self,
        eps: float = CHARBONNIER_EPS,
        edge_weight: float = EDGE_WEIGHT,
    ) -> None:

        super().__init__()

        self.charbonnier = CharbonnierLoss(eps)
        self.edge = EdgeGradientLoss()
        self.edge_weight = edge_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ):

        charbonnier = self.charbonnier(prediction, target)
        edge = self.edge(prediction, target)
        total = charbonnier + self.edge_weight * edge

        return total, charbonnier, edge


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(model_name: str) -> nn.Module:
    """Construit explicitement V1 ou V2, sans changer le choix par defaut."""
    if model_name == "v1":
        return M3GSS_v0_32x8()
    if model_name == "v2":
        return M3GSS_v2()
    raise ValueError(f"Modele inconnu : {model_name}")


def count_model_parameters(model_name: str, model: nn.Module) -> int:
    if model_name == "v1":
        return count_parameters(model)
    if model_name == "v2":
        return count_parameters_v2(model)
    raise ValueError(f"Modele inconnu : {model_name}")


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def make_training_pair(
    image: np.ndarray,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:

    h, w = image.shape[:2]

    if h < PATCH_SIZE or w < PATCH_SIZE:
        raise ValueError(f"Image trop petite: {image.shape}")

    x = rng.randint(0, w - PATCH_SIZE)
    y = rng.randint(0, h - PATCH_SIZE)

    hr = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

    lr_w = PATCH_SIZE // SCALE
    lr_h = PATCH_SIZE // SCALE

    lr = area_downscale(hr, lr_w, lr_h)
    bicubic = catmull_rom_upscale(lr, PATCH_SIZE, PATCH_SIZE)

    bicubic_t = torch.from_numpy(
        np.ascontiguousarray(bicubic)
    ).permute(2, 0, 1).float() / 255.0

    hr_t = torch.from_numpy(
        np.ascontiguousarray(hr)
    ).permute(2, 0, 1).float() / 255.0

    return bicubic_t, hr_t


def make_training_pair_v2(
    image: np.ndarray,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construit LR 48, baseline Catmull-Rom HR 96 et cible HR 96."""
    h, w = image.shape[:2]
    if h < PATCH_SIZE or w < PATCH_SIZE:
        raise ValueError(f"Image trop petite: {image.shape}")

    x = rng.randint(0, w - PATCH_SIZE)
    y = rng.randint(0, h - PATCH_SIZE)
    hr = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

    lr_size = PATCH_SIZE // SCALE
    lr = area_downscale(hr, lr_size, lr_size)
    baseline_hr = catmull_rom_upscale(lr, PATCH_SIZE, PATCH_SIZE)

    def image_to_tensor(patch: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.ascontiguousarray(patch))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )

    return image_to_tensor(lr), image_to_tensor(baseline_hr), image_to_tensor(hr)


def make_batch(
    paths: list[Path],
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:

    inputs = []
    targets = []

    for _ in range(batch_size):
        path = paths[rng.randrange(len(paths))]
        image = load_image(path)

        inp, target = make_training_pair(
            image,
            rng,
        )

        inputs.append(inp)
        targets.append(target)

    return torch.stack(inputs), torch.stack(targets)


def make_batch_v2(
    paths: list[Path],
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lr_inputs = []
    baselines = []
    targets = []

    for _ in range(batch_size):
        path = paths[rng.randrange(len(paths))]
        image = load_image(path)
        lr, baseline_hr, target_hr = make_training_pair_v2(image, rng)
        lr_inputs.append(lr)
        baselines.append(baseline_hr)
        targets.append(target_hr)

    return torch.stack(lr_inputs), torch.stack(baselines), torch.stack(targets)


def make_synthetic_batch_v2(
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construit un batch V2 en memoire, sans dataset ni fichier image."""
    from m3gss_v0.synthetic import make_synthetic_image

    lr_inputs = []
    baselines = []
    targets = []
    for _ in range(batch_size):
        image_index = rng.randrange(10_000)
        image = make_synthetic_image(image_index, size=PATCH_SIZE)
        lr, baseline_hr, target_hr = make_training_pair_v2(image, rng)
        lr_inputs.append(lr)
        baselines.append(baseline_hr)
        targets.append(target_hr)

    return torch.stack(lr_inputs), torch.stack(baselines), torch.stack(targets)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    seed: int,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "seed": seed,
            "patch_size": PATCH_SIZE,
            "scale": SCALE,
            "dataset": "DIV2K_train_HR + Flickr2K_HR",
            "loss_version": LOSS_VERSION,
            "charbonnier_eps": CHARBONNIER_EPS,
            "edge_weight": EDGE_WEIGHT,
        },
        path,
    )


PROTECTED_CHECKPOINT_PREFIXES = ("m3gss_v0_", "m3gss_v1_")


def validate_v2_checkpoint_path(path: Path) -> None:
    """Refuse les chemins V0/V1 et les fichiers existants non identifies V2."""
    path = Path(path)
    if path.name.casefold().startswith(PROTECTED_CHECKPOINT_PREFIXES):
        raise ValueError(f"Chemin de checkpoint protege contre V2 : {path}")

    if path.exists():
        try:
            existing = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                f"Refus d'ecraser un checkpoint existant non valide V2 : {path}"
            ) from exc
        if not isinstance(existing, dict) or existing.get("architecture") != "M3GSS_v2":
            raise ValueError(
                f"Refus d'ecraser un checkpoint qui n'est pas M3GSS_v2 : {path}"
            )


def save_v2_checkpoint(
    path: Path,
    model: M3GSS_v2,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    seed: int,
    dataset: str,
) -> None:
    validate_v2_checkpoint_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "M3GSS_v2",
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "loss": loss,
            "seed": seed,
            "scale": SCALE,
            "patch_size_hr": PATCH_SIZE,
            "patch_size_lr": PATCH_SIZE // SCALE,
            "loss_version": LOSS_VERSION,
            "num_parameters": count_parameters_v2(model),
            "dataset": dataset,
        },
        path,
    )


def load_v2_checkpoint(
    path: Path,
    model: M3GSS_v2,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "M3GSS_v2":
        raise ValueError(f"Checkpoint refuse : architecture M3GSS_v2 requise ({path})")

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["step"]),
        float(checkpoint["loss"]),
        int(checkpoint.get("seed", 1234)),
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, int]:

    print(f"Chargement du checkpoint : {path}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    step = int(checkpoint["step"])
    loss = float(checkpoint["loss"])
    seed = int(checkpoint.get("seed", 1234))

    print(f"Reprise a l'etape : {step}")
    print(f"Derniere loss      : {loss:.6f}")
    print()

    return step, loss, seed


def main() -> int:

    parser = argparse.ArgumentParser(
        description="Entrainement M3GSS V1 ou V2 (Charbonnier + Edge)"
    )

    parser.add_argument(
        "--model",
        choices=("v1", "v2"),
        default="v1",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="V2 seulement : tres court test sur batches synthetiques en memoire",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.smoke_test and args.model != "v2":
        parser.error("--smoke-test est reserve a --model v2")
    if args.smoke_test and args.resume:
        parser.error("--resume ne peut pas etre combine avec --smoke-test")
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps et --batch-size doivent etre positifs")

    if args.checkpoint is None:
        if args.smoke_test:
            checkpoint = DEFAULT_V2_SMOKE_CHECKPOINT
        elif args.model == "v2":
            checkpoint = DEFAULT_V2_CHECKPOINT
        else:
            checkpoint = DEFAULT_CHECKPOINT
    else:
        checkpoint = Path(args.checkpoint)

    if args.model == "v2":
        validate_v2_checkpoint_path(checkpoint)
        if not args.smoke_test:
            validate_v2_checkpoint_path(DEFAULT_V2_BEST)
        if checkpoint.resolve() == DEFAULT_V2_BEST.resolve():
            parser.error("--checkpoint doit etre distinct du checkpoint V2 best")

    paths: list[Path] = []
    dataset_name = "synthetic (generated in memory)" if args.smoke_test else (
        "DIV2K_train_HR + Flickr2K_HR"
    )

    if not args.smoke_test:
        if not DIV2K.exists():
            raise FileNotFoundError(
                f"Dataset DIV2K introuvable : {DIV2K}"
            )

        if not FLICKR2K.exists():
            raise FileNotFoundError(
                f"Dataset Flickr2K introuvable : {FLICKR2K}"
            )

        div2k_paths = sorted(DIV2K.rglob("*.png"))
        flickr2k_paths = sorted(FLICKR2K.rglob("*.png"))

        if not div2k_paths:
            raise RuntimeError(f"Aucune image PNG dans {DIV2K}")

        if not flickr2k_paths:
            raise RuntimeError(f"Aucune image PNG dans {FLICKR2K}")

        paths = div2k_paths + flickr2k_paths

    if not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError(
            "CUDA n'est pas disponible."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== M3GSS {args.model.upper()} - ENTRAINEMENT ===")
    print()
    if args.smoke_test:
        print("Dataset         : synthetique (en memoire)")
    else:
        print(f"DIV2K images    : {len(div2k_paths)}")
        print(f"Flickr2K images : {len(flickr2k_paths)}")
        print(f"Total images    : {len(paths)}")
    print()
    print(
        f"GPU             : {torch.cuda.get_device_name(0)}"
        if device.type == "cuda"
        else "GPU             : indisponible (smoke test CPU)"
    )
    print(f"PyTorch         : {torch.__version__}")
    print(f"CUDA            : {torch.version.cuda}")
    print(f"Patch HR        : {PATCH_SIZE}x{PATCH_SIZE}")
    print(
        f"Patch LR        : "
        f"{PATCH_SIZE // SCALE}x{PATCH_SIZE // SCALE}"
    )
    print(f"Steps           : {args.steps}")
    print(f"Batch           : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"Loss            : {LOSS_VERSION}")
    print(f"Charbonnier eps : {CHARBONNIER_EPS}")
    print(f"Edge weight     : {EDGE_WEIGHT}")
    print()

    model = build_model(args.model).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = M3GSSV1Loss()

    start_step = 0
    resume_seed = args.seed

    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint}"
            )

        if args.model == "v2":
            start_step, _, resume_seed = load_v2_checkpoint(
                checkpoint, model, optimizer, device
            )
        else:
            start_step, _, resume_seed = load_checkpoint(
                checkpoint,
                model,
                optimizer,
                device,
            )

        rng = random.Random(resume_seed)

    else:
        rng = random.Random(args.seed)

    print(f"Parametres      : {count_model_parameters(args.model, model):,}")
    print(
        f"Debut effectif  : "
        f"{start_step + 1}"
    )
    print(
        f"Fin demandee    : "
        f"{args.steps}"
    )
    print()
    print("Demarrage...")
    print()

    start = time.perf_counter()

    running_loss = 0.0
    running_charbonnier = 0.0
    running_edge = 0.0
    loss_count = 0

    model.train()

    for step in range(
        start_step + 1,
        args.steps + 1,
    ):

        if args.model == "v1":
            model_input, target = make_batch(paths, args.batch_size, rng)
            model_input = model_input.to(device, non_blocking=True)
        elif args.smoke_test:
            lr_input, bicubic, target = make_synthetic_batch_v2(args.batch_size, rng)
            lr_input = lr_input.to(device, non_blocking=True)
            bicubic = bicubic.to(device, non_blocking=True)
        else:
            lr_input, bicubic, target = make_batch_v2(paths, args.batch_size, rng)
            lr_input = lr_input.to(device, non_blocking=True)
            bicubic = bicubic.to(device, non_blocking=True)

        target = target.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if args.model == "v1":
            output = model(model_input)
        else:
            output = model(lr_input, bicubic)

        loss, charbonnier_t, edge_t = criterion(
            output,
            target,
        )

        loss.backward()

        optimizer.step()

        loss_value = float(
            loss.detach()
        )
        charbonnier_value = float(
            charbonnier_t.detach()
        )
        edge_value = float(
            edge_t.detach()
        )

        running_loss += loss_value
        running_charbonnier += charbonnier_value
        running_edge += edge_value
        loss_count += 1

        if (
            step == start_step + 1
            or step % 25 == 0
            or step == args.steps
        ):

            elapsed = (
                time.perf_counter()
                - start
            )

            current_steps = (
                step - start_step
            )

            steps_per_sec = (
                current_steps / elapsed
            )

            print(
                f"step {step:5d}/{args.steps} | "
                f"Charbonnier {charbonnier_value:.6f} | "
                f"Edge {edge_value:.6f} | "
                f"Total {loss_value:.6f} | "
                f"{steps_per_sec:.2f} step/s"
            )

        if step % SAVE_EVERY == 0:

            checkpoint_loss = (
                running_loss / loss_count
            )

            if args.model == "v2":
                save_v2_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    step,
                    checkpoint_loss,
                    resume_seed,
                    dataset_name,
                )
            else:
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    step,
                    checkpoint_loss,
                    resume_seed,
                )

            print(
                f"  -> checkpoint sauvegarde : "
                f"step {step}"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    final_loss = (
        running_loss / loss_count
    )

    final_charbonnier = (
        running_charbonnier / loss_count
    )

    final_edge = (
        running_edge / loss_count
    )

    if args.model == "v2":
        save_v2_checkpoint(
            checkpoint,
            model,
            optimizer,
            args.steps,
            final_loss,
            resume_seed,
            dataset_name,
        )
        if not args.smoke_test:
            save_v2_checkpoint(
                DEFAULT_V2_BEST,
                model,
                optimizer,
                args.steps,
                final_loss,
                resume_seed,
                dataset_name,
            )
        best_path = None if args.smoke_test else DEFAULT_V2_BEST
    else:
        save_checkpoint(
            checkpoint,
            model,
            optimizer,
            args.steps,
            final_loss,
            resume_seed,
        )
        best_path = Path(DEFAULT_BEST)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "step": args.steps,
                "loss": final_loss,
                "patch_size": PATCH_SIZE,
                "scale": SCALE,
                "dataset": "DIV2K_train_HR + Flickr2K_HR",
                "loss_version": LOSS_VERSION,
            },
            best_path,
        )

    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )

    print()
    print(f"=== ENTRAINEMENT M3GSS {args.model.upper()} TERMINE ===")
    print(f"Charbonnier    : {final_charbonnier:.6f}")
    print(f"Edge/Gradient  : {final_edge:.6f}")
    print(f"Loss totale    : {final_loss:.6f}")
    print(f"Temps total  : {elapsed:.2f} s")
    print(
        f"Vitesse      : "
        f"{(args.steps - start_step) / elapsed:.2f} step/s"
    )
    print(f"VRAM max     : {peak_vram:.1f} MiB")
    print(f"Checkpoint   : {checkpoint}")
    if best_path is not None:
        print(f"Modele final : {best_path}")
    if args.smoke_test:
        print("Mode smoke test : aucun checkpoint best V2 n'a ete modifie")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())