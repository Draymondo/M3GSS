from pathlib import Path
import math
import argparse
import gc

import numpy as np
import torch
from PIL import Image

from m3gss_v0.bicubic import area_downscale, catmull_rom_upscale
from m3gss_v0.model import M3GSS_v0_32x8


ROOT = Path(__file__).resolve().parent

DATASET = Path(r"D:\M3GSS_OFFLINE\datasets\DIV2K\valid_HR")
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "m3gss_v0_big_best.pt"

PATCH_SIZE = 96
SCALE = 2


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mse = np.mean((a - b) ** 2)

    if mse == 0:
        return float("inf")

    return 10.0 * math.log10((255.0 ** 2) / mse)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluation M3GSS v0"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATASET}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}"
        )

    paths = sorted(DATASET.rglob("*.png"))

    if not paths:
        raise RuntimeError(
            f"Aucune image PNG dans {DATASET}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible.")

    device = torch.device("cuda")

    print("=== EVALUATION M3GSS v0 ===")
    print(f"Dataset    : {DATASET}")
    print(f"Images     : {len(paths)}")
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint : {checkpoint_path}")
    print()

    model = M3GSS_v0_32x8().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    bicubic_scores = []
    m3gss_scores = []

    for index, path in enumerate(paths, 1):

        with Image.open(path) as img:
            hr = np.asarray(
                img.convert("RGB"),
                dtype=np.uint8,
            )

        h, w = hr.shape[:2]

        h2 = (h // 2) * 2
        w2 = (w // 2) * 2

        hr = hr[:h2, :w2]

        lr = area_downscale(
            hr,
            w2 // SCALE,
            h2 // SCALE,
        )

        bicubic = catmull_rom_upscale(
            lr,
            w2,
            h2,
        )

        bicubic_tensor = torch.from_numpy(
            np.ascontiguousarray(bicubic)
        ).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        with torch.inference_mode():
            output = model(
                bicubic_tensor.to(device)
            )

            output = (
                output.squeeze(0)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

        bicubic_score = psnr(hr, bicubic)
        m3gss_score = psnr(hr, output)

        bicubic_scores.append(bicubic_score)
        m3gss_scores.append(m3gss_score)

        del bicubic_tensor
        del output

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        if (
            index == 1
            or index % 10 == 0
            or index == len(paths)
        ):
            print(
                f"{index:3d}/{len(paths)} | "
                f"Bicubique {bicubic_score:.3f} dB | "
                f"M3GSS {m3gss_score:.3f} dB"
            )

    bicubic_avg = float(
        np.mean(bicubic_scores)
    )

    m3gss_avg = float(
        np.mean(m3gss_scores)
    )

    gain = m3gss_avg - bicubic_avg

    print()
    print("=== RESULTAT FINAL ===")
    print(
        f"PSNR bicubique : {bicubic_avg:.4f} dB"
    )
    print(
        f"PSNR M3GSS     : {m3gss_avg:.4f} dB"
    )
    print(
        f"GAIN M3GSS     : {gain:+.4f} dB"
    )

    if gain > 0:
        print(
            "M3GSS apporte un gain PSNR mesurable."
        )
    elif gain < 0:
        print(
            "M3GSS est actuellement inférieur au bicubique."
        )
    else:
        print(
            "M3GSS est équivalent au bicubique."
        )

    print()
    print("Evaluation terminee.")


if __name__ == "__main__":
    main()