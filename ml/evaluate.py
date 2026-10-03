"""Infrastructure d'evaluation M3GSS v0 — synthetique, deterministe, CPU.

Chaine evaluee (aucun dataset externe, aucun GPU) :

    HR uint8 [H,W,3]
      -> area_downscale x2  (Resampler, parite C++ verifiee)
      -> LR uint8
      -> Catmull-Rom bicubique (parite C++ verifiee)
      -> bicubique uint8 [H,W,3]
      -> float32 [1,3,H,W] dans [0,1]
      -> M3GSS_v0
      -> uint8 [H,W,3]

Puis metriques (definitions identiques au C++, voir metrics.py) :
    - HR vs bicubique  (baseline de reference)
    - HR vs M3GSS      (modele evalue)

Avec un modele a zero-init, la sortie M3GSS est IDENTIQUE au bicubique :
MSE = 0, PSNR = +inf, SSIM = 1. C'est l'invariant de l'architecture.

Commande :
    .\\.venv\\Scripts\\python.exe evaluate.py
    .\\.venv\\Scripts\\python.exe evaluate.py --checkpoint checkpoints/smoke_test.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import torch

from metrics import QualityMetrics, compute
from m3gss_v0.bicubic import area_downscale, catmull_rom_upscale
from m3gss_v0.model import M3GSS_v0_32x8
from m3gss_v0.patches import from_tensor, to_tensor
from m3gss_v0.synthetic import make_synthetic_image

__all__ = [
    "ImageEvaluation",
    "SetEvaluation",
    "evaluate_image",
    "evaluate_synthetic_set",
    "build_zero_init_model",
    "load_model_from_checkpoint",
]


@dataclass
class ImageEvaluation:
    """Resultat pour une image."""

    name: str
    width: int
    height: int
    lr_width: int
    lr_height: int
    model_tag: str
    baseline: QualityMetrics
    model: QualityMetrics | None = None


@dataclass
class SetEvaluation:
    """Resultat agrege pour un ensemble d'images."""

    model_tag: str = "zero-init"
    images: list = field(default_factory=list)
    mean_baseline_mse: float = 0.0
    mean_baseline_psnr: float = 0.0
    mean_baseline_ssim: float = 0.0
    mean_model_mse: float = 0.0
    mean_model_psnr: float = 0.0
    mean_model_ssim: float = 0.0


def build_zero_init_model() -> M3GSS_v0_32x8:
    """Modele M3GSS non entrainable (tail zero-init) -> sortie = bicubique."""
    model = M3GSS_v0_32x8()
    model.eval()
    return model


def load_model_from_checkpoint(path: str) -> M3GSS_v0_32x8:
    """Charge un checkpoint (usage FONCTIONNEL uniquement, pas une mesure)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = M3GSS_v0_32x8()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _predict_uint8(model: M3GSS_v0_32x8, image_uint8: np.ndarray) -> np.ndarray:
    """Entree uint8 [H,W,C] -> sortie uint8 [H,W,C] (aller-retour sans perte)."""
    tensor = to_tensor(image_uint8).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
    return from_tensor(out.squeeze(0))


def evaluate_image(
    hr_image: np.ndarray,
    model: torch.nn.Module | None = None,
    model_tag: str = "zero-init",
    factor: float = 0.5,
    name: str = "image",
) -> ImageEvaluation:
    """Evalue une image HR : baseline bicubique puis (optionnel) M3GSS."""
    hr = np.asarray(hr_image)
    if hr.ndim != 3:
        raise ValueError(f"image attendue [H,W,C], recu {hr.shape}")
    height, width = hr.shape[0], hr.shape[1]

    low_w = max(1, int(np.floor(width * factor + 0.5)))
    low_h = max(1, int(np.floor(height * factor + 0.5)))

    lr = area_downscale(hr, low_w, low_h)
    bicubic = catmull_rom_upscale(lr, width, height)

    baseline = compute(hr, bicubic)

    model_metrics = None
    if model is not None:
        prediction = _predict_uint8(model, bicubic)
        model_metrics = compute(hr, prediction)

    return ImageEvaluation(
        name=name,
        width=width,
        height=height,
        lr_width=low_w,
        lr_height=low_h,
        model_tag=model_tag,
        baseline=baseline,
        model=model_metrics,
    )


def evaluate_synthetic_set(
    num_images: int = 6,
    size: int = 192,
    image_offset: int = 100,
    model: torch.nn.Module | None = None,
    model_tag: str = "zero-init",
    factor: float = 0.5,
) -> SetEvaluation:
    """Evalue un ensemble synthetique deterministe (memes motifs que l'entrainement)."""
    result = SetEvaluation(model_tag=model_tag)
    evaluations = []

    for i in range(num_images):
        image = make_synthetic_image(image_offset + i, size=size)
        evaluations.append(
            evaluate_image(
                image, model=model, model_tag=model_tag, factor=factor,
                name=f"syn{image_offset + i:03d}_{size}",
            )
        )

    result.images = evaluations
    n = len(evaluations)
    result.mean_baseline_mse = sum(e.baseline.mse for e in evaluations) / n
    result.mean_baseline_psnr = sum(e.baseline.psnr for e in evaluations) / n
    result.mean_baseline_ssim = sum(e.baseline.ssim for e in evaluations) / n
    if model is not None:
        result.mean_model_mse = sum(e.model.mse for e in evaluations) / n
        result.mean_model_psnr = sum(e.model.psnr for e in evaluations) / n
        result.mean_model_ssim = sum(e.model.ssim for e in evaluations) / n
    return result


def _format(value: float) -> str:
    return "inf" if np.isinf(value) else f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="M3GSS v0 - evaluation")
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--offset", type=int, default=100)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint optionnel : test FONCTIONNEL uniquement",
    )
    args = parser.parse_args()

    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint)
        tag = "smoke_test.pt (FONCTIONNEL, PAS une mesure de qualite)"
    else:
        model = build_zero_init_model()
        tag = "zero-init (invariant : sortie == bicubique)"

    print("=== M3GSS v0 - evaluation synthetique (CPU, deterministe) ===")
    print(f"images : {args.num_images} | taille : {args.size}x{args.size} "
          f"| offset : {args.offset} | modele : {tag}")
    print()

    result = evaluate_synthetic_set(
        num_images=args.num_images, size=args.size, image_offset=args.offset,
        model=model, model_tag=tag,
    )

    for e in result.images:
        print(f"{e.name:<16} HR {e.width}x{e.height} | LR {e.lr_width}x{e.lr_height}")
        print(
            f"{'':16} baseline  MSE {_format(e.baseline.mse):>12}"
            f" | PSNR {_format(e.baseline.psnr):>12} dB"
            f" | SSIM {e.baseline.ssim:.6f}"
        )
        if e.model is not None:
            print(
                f"{'':16} M3GSS     MSE {_format(e.model.mse):>12}"
                f" | PSNR {_format(e.model.psnr):>12} dB"
                f" | SSIM {e.model.ssim:.6f}"
            )

    print()
    print("=== Moyennes ===")
    print(f"baseline : MSE {_format(result.mean_baseline_mse):>12} "
          f"| PSNR {_format(result.mean_baseline_psnr):>12} dB "
          f"| SSIM {result.mean_baseline_ssim:.6f}")
    if model is not None:
        print(f"M3GSS    : MSE {_format(result.mean_model_mse):>12} "
              f"| PSNR {_format(result.mean_model_psnr):>12} dB "
              f"| SSIM {result.mean_model_ssim:.6f}")
    print()
    print("Rappel : mesures sur images synthetiques, hors distribution de jeu.")
    print("Le checkpoint smoke_test (si utilise) est un test fonctionnel, "
          "PAS une mesure de qualite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
