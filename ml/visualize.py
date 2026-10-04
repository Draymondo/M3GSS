from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw

from m3gss_v0.bicubic import area_downscale, catmull_rom_upscale
from m3gss_v0.model import M3GSS_v0_32x8


ROOT = Path(__file__).resolve().parent

DATASET = Path(r"D:\M3GSS_OFFLINE\datasets\DIV2K\valid_HR")
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "m3gss_v0_big_best.pt"
OUTPUT_DIR = ROOT / "visual_results"

PATCH_SIZE = 96
SCALE = 2


def pil_to_tensor(img):
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(tensor):
    arr = (
        tensor.squeeze(0)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255.0
    ).round().astype(np.uint8)

    return Image.fromarray(arr)


def make_m3gss_output(model, bicubic, device):
    width = bicubic.width
    height = bicubic.height 

    # Padding reflect pour obtenir des dimensions multiples de 96.
    pad_w = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
    pad_h = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE

    padded = Image.fromarray(
        np.pad(
            np.asarray(bicubic),
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )
    )

    padded_w, padded_h = padded.size

    output = Image.new("RGB", (padded_w, padded_h))

    with torch.inference_mode():
        for y in range(0, padded_h, PATCH_SIZE):
            for x in range(0, padded_w, PATCH_SIZE):
                patch = padded.crop(
                    (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
                )

                tensor = pil_to_tensor(patch).to(device)

                result = model(tensor)

                result_img = tensor_to_pil(result)

                output.paste(result_img, (x, y))

    return output.crop((0, 0, width, height))


def add_label(img, text):
    result = img.copy()

    draw = ImageDraw.Draw(result)

    draw.rectangle(
        (0, 0, result.width, 38),
        fill=(0, 0, 0),
    )

    draw.text(
        (10, 8),
        text,
        fill=(255, 255, 255),
    )

    return result


def create_comparison(hr, bicubic, m3gss):
    # Même taille pour les trois images.
    width, height = hr.size

    bicubic = bicubic.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    m3gss = m3gss.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    hr = add_label(hr, "ORIGINAL HR")
    bicubic = add_label(bicubic, "BICUBIQUE x2")
    m3gss = add_label(m3gss, "M3GSS v0")

    comparison = Image.new(
        "RGB",
        (width * 3, height),
        (20, 20, 20),
    )

    comparison.paste(hr, (0, 0))
    comparison.paste(bicubic, (width, 0))
    comparison.paste(m3gss, (width * 2, 0))

    return comparison


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== VISUALISATION M3GSS v0 ===")
    print(f"Dataset    : {DATASET}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"GPU        : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    print(f"Sortie     : {OUTPUT_DIR}")
    print()

    model = M3GSS_v0_32x8()

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    images = sorted(DATASET.glob("*.png"))

    if not images:
        images = sorted(DATASET.rglob("*.png"))

    images = images[:args.count]

    print(f"Images selectionnees : {len(images)}")
    print()

    for index, image_path in enumerate(images, 1):

        print(
            f"{index}/{len(images)} | "
            f"{image_path.name}"
        )

        hr = Image.open(image_path).convert("RGB")

        # Même protocole que l'évaluation.
        width = (hr.width // 2) * 2
        height = (hr.height // 2) * 2

        hr = hr.crop((0, 0, width, height))

        lr = area_downscale(
            hr,
            width // SCALE,
            height // SCALE,
        )
        bicubic = catmull_rom_upscale(
            lr,
            width,
            height,
        )

        bicubic = Image.fromarray(
            bicubic.astype(np.uint8),
            mode="RGB",
)

        m3gss = make_m3gss_output(
            model,
            bicubic,
            device,
        )

        comparison = create_comparison(
            hr,
            bicubic,
            m3gss,
        )

        output_path = (
            OUTPUT_DIR
            / f"{index:02d}_{image_path.stem}_comparison.png"
        )

        comparison.save(
            output_path,
            format="PNG",
        )

        print(f"    -> {output_path}")

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    print()
    print("=== TERMINE ===")
    print(f"Resultats : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()