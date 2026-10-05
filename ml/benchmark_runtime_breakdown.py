import time
import torch
import torch.nn.functional as F
from m3gss_v0.model import M3GSS_v0_32x8

CHECKPOINT = "ml/checkpoints/m3gss_v1_big_best.pt"

W_LR, H_LR = 960, 540
W_HR, H_HR = 1920, 1080

WARMUP = 20
ITERATIONS = 100

checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model = M3GSS_v0_32x8().cuda().eval()
model.load_state_dict(checkpoint["state_dict"])

lr = torch.rand(
    1, 3, H_LR, W_LR,
    device="cuda",
    dtype=torch.float32,
)

with torch.inference_mode():

    # Prépare l'entrée HR une seule fois
    hr = F.interpolate(
        lr,
        size=(H_HR, W_HR),
        mode="bicubic",
        align_corners=False,
    )

    # ---------------------------------------------------------
    # 1. Bicubic seul
    # ---------------------------------------------------------
    for _ in range(WARMUP):
        F.interpolate(
            lr,
            size=(H_HR, W_HR),
            mode="bicubic",
            align_corners=False,
        )

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(ITERATIONS):
        F.interpolate(
            lr,
            size=(H_HR, W_HR),
            mode="bicubic",
            align_corners=False,
        )

    torch.cuda.synchronize()

    bicubic_ms = (
        (time.perf_counter() - start)
        * 1000.0
        / ITERATIONS
    )

    # ---------------------------------------------------------
    # 2. Réseau seul à 1920x1080
    # ---------------------------------------------------------
    for _ in range(WARMUP):
        model(hr)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(ITERATIONS):
        model(hr)

    torch.cuda.synchronize()

    network_hr_ms = (
        (time.perf_counter() - start)
        * 1000.0
        / ITERATIONS
    )

    # ---------------------------------------------------------
    # 3. Réseau à 960x540
    # ---------------------------------------------------------
    for _ in range(WARMUP):
        model(lr)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(ITERATIONS):
        model(lr)

    torch.cuda.synchronize()

    network_lr_ms = (
        (time.perf_counter() - start)
        * 1000.0
        / ITERATIONS
    )

print("=== M3GSS V1 — Décomposition du coût runtime ===")
print("GPU:", torch.cuda.get_device_name(0))
print()
print(f"Bicubic 960x540 -> 1920x1080 : {bicubic_ms:.3f} ms")
print(f"Réseau à 1920x1080           : {network_hr_ms:.3f} ms")
print(f"Réseau à 960x540             : {network_lr_ms:.3f} ms")
print()
print(f"Pipeline actuel              : {bicubic_ms + network_hr_ms:.3f} ms")
print(f"Réseau basse résolution      : {network_lr_ms:.3f} ms")
print()
print("Benchmark terminé.")
