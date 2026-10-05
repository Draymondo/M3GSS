import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.m3gss_v0.model import M3GSS_v0_32x8


CHECKPOINT = ROOT / "ml" / "checkpoints" / "m3gss_v1_big_best.pt"

LR_W, LR_H = 960, 540
HR_W, HR_H = 1920, 1080

WARMUP = 20
ITERATIONS = 100


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible.")

    device = torch.device("cuda")

    print("=== M3GSS V1 — GTX 970 inference benchmark ===")
    print("PyTorch :", torch.__version__)
    print("GPU     :", torch.cuda.get_device_name(0))
    print("Input   :", f"{LR_W}x{LR_H}")
    print("Network :", f"{HR_W}x{HR_H}")
    print("Warmup  :", WARMUP)
    print("Tests   :", ITERATIONS)
    print()

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    model = M3GSS_v0_32x8()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)

    # Image LR synthétique.
    lr = torch.rand(
        1, 3, LR_H, LR_W,
        dtype=torch.float32,
        device=device,
    )

    # Même principe que le pipeline M3GSS :
    # LR -> bicubique -> entrée réseau à résolution cible.
    bicubic = F.interpolate(
        lr,
        size=(HR_H, HR_W),
        mode="bicubic",
        align_corners=False,
    )

    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = model(bicubic)

        torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(ITERATIONS):
            _ = model(bicubic)

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    total_ms = elapsed * 1000.0
    avg_ms = total_ms / ITERATIONS
    fps = 1000.0 / avg_ms

    print("Résultats")
    print("---------")
    print(f"Temps total : {total_ms:.2f} ms")
    print(f"Latence     : {avg_ms:.3f} ms/frame")
    print(f"FPS théorique réseau : {fps:.2f}")
    print(f"VRAM allouée : {torch.cuda.memory_allocated() / 1024**2:.1f} MiB")
    print(f"VRAM réservée : {torch.cuda.memory_reserved() / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
