from __future__ import annotations

import argparse
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from m3gss_v0.bicubic import area_downscale, catmull_rom_upscale
from m3gss_v0.model import M3GSS_v0_32x8, count_parameters


ROOT = Path(__file__).resolve().parent

DIV2K = Path(r"D:\M3GSS_OFFLINE\datasets\DIV2K\train_HR")
FLICKR2K = Path(r"D:\M3GSS_OFFLINE\datasets\Flickr2K\HR\Flickr2K")

# Noms V1 : un run V1 ne doit jamais ecraser un checkpoint V0.
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "m3gss_v1_big_latest.pt"
DEFAULT_BEST = ROOT / "checkpoints" / "m3gss_v1_big_best.pt"

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


def load_image(
    source: Path | tuple[Path, str],
    zip_file: zipfile.ZipFile | None = None,
) -> np.ndarray:
    """Charge une image depuis un fichier PNG ou directement depuis un ZIP."""
    if isinstance(source, tuple):
        if zip_file is None:
            raise RuntimeError("Lecteur ZIP Flickr2K manquant.")
        zip_path, member = source
        with zip_file.open(member) as fp:
            with Image.open(fp) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)

    with Image.open(source) as img:
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


def make_batch(
    paths: list[Path | tuple[Path, str]],
    batch_size: int,
    rng: random.Random,
    zip_file: zipfile.ZipFile | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    inputs = []
    targets = []

    for _ in range(batch_size):
        path = paths[rng.randrange(len(paths))]
        image = load_image(path, zip_file)

        inp, target = make_training_pair(
            image,
            rng,
        )

        inputs.append(inp)
        targets.append(target)

    return torch.stack(inputs), torch.stack(targets)


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
        description="M3GSS V1 - gros entrainement DIV2K + Flickr2K (Charbonnier + Edge)"
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
        "--div2k",
        type=str,
        default=str(DIV2K),
        help="Dossier DIV2K contenant les PNG d'entrainement.",
    )

    parser.add_argument(
        "--flickr2k",
        type=str,
        default=str(FLICKR2K),
        help="Dossier Flickr2K contenant les PNG d'entrainement.",
    )

    parser.add_argument(
        "--flickr2k-zip",
        type=str,
        default=None,
        help="Archive Flickr2K.zip a lire directement sans extraction.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    div2k_root = Path(args.div2k)
    flickr2k_root = Path(args.flickr2k)
    flickr2k_zip_path = (
        Path(args.flickr2k_zip)
        if args.flickr2k_zip
        else None
    )

    if not div2k_root.exists():
        raise FileNotFoundError(
            f"Dataset DIV2K introuvable : {div2k_root}"
        )

    if flickr2k_zip_path is None and not flickr2k_root.exists():
        raise FileNotFoundError(
            f"Dataset Flickr2K introuvable : {flickr2k_root}"
        )

    div2k_paths = sorted(
        div2k_root.rglob("*.png")
    )

    if flickr2k_zip_path is not None:
        if not flickr2k_zip_path.exists():
            raise FileNotFoundError(
                f"Archive Flickr2K introuvable : {flickr2k_zip_path}"
            )

        with zipfile.ZipFile(flickr2k_zip_path, "r") as zf:
            flickr2k_members = sorted(
                name for name in zf.namelist()
                if name.lower().endswith(".png")
            )

        if not flickr2k_members:
            raise RuntimeError(
                f"Aucune image PNG dans {flickr2k_zip_path}"
            )

        flickr2k_paths = [
            (flickr2k_zip_path, member)
            for member in flickr2k_members
        ]
    else:
        flickr2k_paths = sorted(
            flickr2k_root.rglob("*.png")
        )

        if not flickr2k_paths:
            raise RuntimeError(
                f"Aucune image PNG dans {flickr2k_root}"
            )

    paths = div2k_paths + flickr2k_paths

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA n'est pas disponible."
        )

    device = torch.device("cuda")

    print("=== M3GSS V1 - GROS ENTRAINEMENT ===")
    print()
    print(f"DIV2K images    : {len(div2k_paths)}")
    print(f"Flickr2K images : {len(flickr2k_paths)}")
    print(f"Total images    : {len(paths)}")
    print()
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
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

    model = M3GSS_v0_32x8().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    criterion = M3GSSV1Loss()

    start_step = 0
    resume_seed = args.seed

    checkpoint = Path(args.checkpoint)

    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint}"
            )

        start_step, _, resume_seed = load_checkpoint(
            checkpoint,
            model,
            optimizer,
            device,
        )

        rng = random.Random(resume_seed)

    else:
        rng = random.Random(args.seed)

    print(f"Parametres      : {count_parameters(model):,}")
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

    flickr_zip_file = (
        zipfile.ZipFile(flickr2k_zip_path, "r")
        if flickr2k_zip_path is not None
        else None
    )

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

        bicubic, target = make_batch(
            paths,
            args.batch_size,
            rng,
            flickr_zip_file,
        )

        bicubic = bicubic.to(
            device,
            non_blocking=True,
        )

        target = target.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(bicubic)

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

    if flickr_zip_file is not None:
        flickr_zip_file.close()

    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        args.steps,
        final_loss,
        resume_seed,
    )

    best_path = Path(
        DEFAULT_BEST
    )

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
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print()
    print("=== GROS ENTRAINEMENT TERMINE ===")
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
    print(f"Modele final : {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())