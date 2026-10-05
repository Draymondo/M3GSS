"""Validation qualité V1/V2 sur DIV2K validation avec baseline Catmull-Rom."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = ROOT / "ml"
sys.path.insert(0, str(ML_ROOT))

from metrics import C1, C2, SSIM_MAX_RADIUS, SSIM_SIGMA, compute_mse, psnr_from_mse
from m3gss_v0.bicubic import catmull_rom_upscale, clamp_to_byte
from m3gss_v0.model import M3GSS_v0_32x8
from m3gss_v0.patches import from_tensor, to_tensor
from m3gss_v2.model import M3GSS_v2

DEFAULT_DATASET = Path(r"D:\M3GSS_OFFLINE\datasets\DIV2K\valid_HR")
DEFAULT_V1_CHECKPOINT = ML_ROOT / "checkpoints" / "m3gss_v1_big_best.pt"
DEFAULT_V2_CHECKPOINT = ML_ROOT / "checkpoints" / "m3gss_v2_best.pt"
DEFAULT_OUTPUT_DIR = ML_ROOT / "visual_results" / "v2_validation"
EXPECTED_IMAGE_COUNT = 100
VISUAL_COUNT = 5
V1_TILE_SIZE = 512
V1_TILE_HALO = 20


def area_downscale_x2(image: np.ndarray) -> np.ndarray:
    """Reduction RGB x2 vectorisee, alignee sur area_downscale pour le facteur 2."""
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Image RGB attendue [H,W,3], recue {source.shape}")
    height, width, _ = source.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"Dimensions paires requises pour une reduction x2: {width}x{height}")

    source_float = source.astype(np.float32, copy=False)
    accumulated = source_float[0::2, 0::2] + source_float[0::2, 1::2]
    accumulated = accumulated + source_float[1::2, 0::2]
    accumulated = accumulated + source_float[1::2, 1::2]
    low_resolution = clamp_to_byte(accumulated * np.float32(0.25))

    expected_shape = (height // 2, width // 2, 3)
    if low_resolution.shape != expected_shape:
        raise RuntimeError(
            f"Forme LR incorrecte: {low_resolution.shape} != {expected_shape}"
        )
    return low_resolution


def compute_ssim_batch(
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, float]:
    """SSIM de plusieurs images via convolutions gaussiennes separables.

    Suit la definition de metrics.py : RGB, plage 0..255, noyau gaussien
    11x11 (sigma 1.5), constantes C1/C2 identiques et moyenne sur la zone
    valide, sans les cinq pixels de bord.
    """
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError(f"Reference attendue [H,W,3], recue {reference.shape}")
    if not candidates:
        return {}

    height, width, _ = reference.shape
    radius = SSIM_MAX_RADIUS
    if min(height, width) < 2 * radius + 1:
        raise ValueError("SSIM exige une image d'au moins 11x11 pixels")
    if any(candidate.shape != reference.shape for candidate in candidates.values()):
        raise ValueError("Reference et candidates SSIM doivent avoir la meme forme")

    names = list(candidates)
    candidate_array = np.stack([candidates[name] for name in names])
    reference_tensor = torch.from_numpy(
        np.ascontiguousarray(reference.transpose(2, 0, 1))
    ).to(device=device, dtype=torch.float32).unsqueeze(0)
    candidate_tensor = torch.from_numpy(
        np.ascontiguousarray(candidate_array.transpose(0, 3, 1, 2))
    ).to(device=device, dtype=torch.float32)

    reference_batch = reference_tensor.expand(len(names), -1, -1, -1)
    moments = torch.cat(
        (
            reference_batch,
            candidate_tensor,
            reference_batch.square(),
            candidate_tensor.square(),
            reference_batch * candidate_tensor,
        ),
        dim=1,
    )

    positions = torch.arange(
        -radius, radius + 1, dtype=torch.float32, device=device
    )
    kernel_1d = torch.exp(-(positions.square()) / (2.0 * SSIM_SIGMA**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    groups = moments.shape[1]
    horizontal = kernel_1d.view(1, 1, 1, -1).expand(groups, 1, 1, -1).contiguous()
    vertical = kernel_1d.view(1, 1, -1, 1).expand(groups, 1, -1, 1).contiguous()

    filtered = F.conv2d(moments, horizontal, groups=groups)
    filtered = F.conv2d(filtered, vertical, groups=groups)
    mean_reference, mean_candidate, mean_reference_sq, mean_candidate_sq, mean_product = (
        filtered.split(3, dim=1)
    )

    variance_reference = (mean_reference_sq - mean_reference.square()).clamp_min(0.0)
    variance_candidate = (mean_candidate_sq - mean_candidate.square()).clamp_min(0.0)
    covariance = mean_product - mean_reference * mean_candidate
    numerator = (2.0 * mean_reference * mean_candidate + C1) * (2.0 * covariance + C2)
    denominator = (
        mean_reference.square() + mean_candidate.square() + C1
    ) * (variance_reference + variance_candidate + C2)
    scores = (numerator / denominator).mean(dim=(1, 2, 3)).detach().cpu().tolist()
    return dict(zip(names, scores))


def load_strict_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    expected_architecture: str | None = None,
) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise RuntimeError(
            f"Checkpoint invalide : cle 'state_dict' absente ({checkpoint_path})"
        )
    if (
        expected_architecture is not None
        and checkpoint.get("architecture") != expected_architecture
    ):
        raise RuntimeError(
            f"Architecture attendue {expected_architecture}, obtenue "
            f"{checkpoint.get('architecture')!r} ({checkpoint_path})"
        )

    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Poids incompatibles avec le modele {type(model).__name__} "
            f"dans {checkpoint_path} : {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return checkpoint


def run_v1_tiled(model: M3GSS_v0_32x8, bicubic_hr: torch.Tensor) -> torch.Tensor:
    """Infere V1 en tuiles avec halo couvrant son champ receptif."""
    _, _, height, width = bicubic_hr.shape
    output = torch.empty_like(bicubic_hr)

    for top in range(0, height, V1_TILE_SIZE):
        bottom = min(top + V1_TILE_SIZE, height)
        tile_top = max(0, top - V1_TILE_HALO)
        tile_bottom = min(height, bottom + V1_TILE_HALO)

        for left in range(0, width, V1_TILE_SIZE):
            right = min(left + V1_TILE_SIZE, width)
            tile_left = max(0, left - V1_TILE_HALO)
            tile_right = min(width, right + V1_TILE_HALO)

            tile = bicubic_hr[:, :, tile_top:tile_bottom, tile_left:tile_right]
            tile_output = model(tile)
            crop_top = top - tile_top
            crop_left = left - tile_left
            output[:, :, top:bottom, left:right] = tile_output[
                :, :, crop_top : crop_top + bottom - top,
                crop_left : crop_left + right - left,
            ]

    return output


def save_comparison(
    output_path: Path,
    catmull_rom: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    reference: np.ndarray,
) -> None:
    panels = (
        ("Catmull-Rom", catmull_rom),
        ("V1", v1),
        ("V2", v2),
        ("HR reference", reference),
    )
    height, width = reference.shape[:2]
    label_height = 36
    comparison = Image.new(
        "RGB", (width * len(panels), height + label_height), (0, 0, 0)
    )
    draw = ImageDraw.Draw(comparison)

    for index, (label, pixels) in enumerate(panels):
        x = index * width
        draw.text((x + 10, 10), label, fill=(255, 255, 255))
        comparison.paste(Image.fromarray(pixels, mode="RGB"), (x, label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(output_path, format="PNG")


def evaluate(args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"Dataset introuvable : {args.dataset}")

    image_paths = sorted(
        path
        for path in args.dataset.rglob("*")
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if len(image_paths) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            f"DIV2K validation doit contenir exactement {EXPECTED_IMAGE_COUNT} PNG, "
            f"{len(image_paths)} trouves dans {args.dataset}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Evaluation qualité M3GSS V1/V2 sur DIV2K validation ===")
    print(f"Dataset : {args.dataset}")
    print(f"Images  : {len(image_paths)}")
    print(f"Device  : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    v1_model = M3GSS_v0_32x8()
    v2_model = M3GSS_v2()
    load_strict_checkpoint(args.v1_checkpoint, v1_model, device)
    v2_checkpoint = load_strict_checkpoint(
        args.v2_checkpoint,
        v2_model,
        device,
        expected_architecture="M3GSS_v2",
    )
    if v2_checkpoint.get("num_parameters") != sum(
        parameter.numel() for parameter in v2_model.parameters()
    ):
        raise RuntimeError("Le nombre de parametres du checkpoint V2 est incoherent")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_indices = set(
        np.linspace(0, len(image_paths) - 1, VISUAL_COUNT, dtype=np.int64).tolist()
    )
    scores = {
        name: {"psnr": [], "ssim": []}
        for name in ("Catmull-Rom", "V1", "V2")
    }

    with torch.inference_mode():
        for index, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                reference = np.asarray(image.convert("RGB"), dtype=np.uint8)

            height = (reference.shape[0] // 2) * 2
            width = (reference.shape[1] // 2) * 2
            reference = np.ascontiguousarray(reference[:height, :width])

            lr = area_downscale_x2(reference)
            catmull_rom = catmull_rom_upscale(lr, width, height)

            lr_tensor = to_tensor(lr).unsqueeze(0).to(device)
            baseline_tensor = to_tensor(catmull_rom).unsqueeze(0).to(device)

            v1_tensor = run_v1_tiled(v1_model, baseline_tensor)
            v2_tensor = v2_model(lr_tensor, baseline_tensor)
            expected_shape = (1, 3, height, width)
            if tuple(v1_tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Sortie V1 inattendue pour {image_path.name}: {tuple(v1_tensor.shape)}"
                )
            if tuple(v2_tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"Sortie V2 inattendue pour {image_path.name}: {tuple(v2_tensor.shape)}"
                )
            if not torch.isfinite(v1_tensor).all().item() or not torch.isfinite(
                v2_tensor
            ).all().item():
                raise RuntimeError(f"Sortie NaN/Inf pour {image_path.name}")

            v1_image = from_tensor(v1_tensor.squeeze(0))
            v2_image = from_tensor(v2_tensor.squeeze(0))
            candidates = {
                "Catmull-Rom": catmull_rom,
                "V1": v1_image,
                "V2": v2_image,
            }
            ssim_scores = compute_ssim_batch(reference, candidates, device)
            for name, candidate in candidates.items():
                mse = compute_mse(reference, candidate)
                scores[name]["psnr"].append(psnr_from_mse(mse))
                scores[name]["ssim"].append(ssim_scores[name])

            if index in visual_indices:
                save_comparison(
                    args.output_dir / f"{index + 1:03d}_{image_path.stem}_comparison.png",
                    catmull_rom,
                    v1_image,
                    v2_image,
                    reference,
                )

            if (index + 1) % 5 == 0 or index + 1 == len(image_paths):
                print(f"Progression : {index + 1}/{len(image_paths)} images")

            del lr_tensor, baseline_tensor, v1_tensor, v2_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    mean_psnr = {
        name: float(np.mean(values["psnr"])) for name, values in scores.items()
    }
    mean_ssim = {
        name: float(np.mean(values["ssim"])) for name, values in scores.items()
    }

    print()
    print("=== Moyennes DIV2K validation ===")
    print(f"Images evaluees : {len(image_paths)}")
    for name in scores:
        print(f"PSNR {name:12s}: {mean_psnr[name]:.4f} dB")
    for name in scores:
        print(f"SSIM {name:12s}: {mean_ssim[name]:.6f}")
    print(f"Gain V1 vs Catmull-Rom : {mean_psnr['V1'] - mean_psnr['Catmull-Rom']:+.4f} dB")
    print(f"Gain V2 vs Catmull-Rom : {mean_psnr['V2'] - mean_psnr['Catmull-Rom']:+.4f} dB")
    print(f"Difference V2 vs V1    : {mean_psnr['V2'] - mean_psnr['V1']:+.4f} dB")
    print(f"Comparaisons visuelles : {args.output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Catmull-Rom, M3GSS V1 et V2 sur DIV2K validation"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--v1-checkpoint", type=Path, default=DEFAULT_V1_CHECKPOINT)
    parser.add_argument(
        "--v2-checkpoint",
        type=Path,
        default=DEFAULT_V2_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())