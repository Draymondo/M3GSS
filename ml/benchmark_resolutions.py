import time
import torch
from m3gss_v0.model import M3GSS_v0_32x8

CHECKPOINT = "ml/checkpoints/m3gss_v1_big_best.pt"

RESOLUTIONS = [
    (1920, 1080),
    (1600, 900),
    (1280, 720),
    (960, 540),
]

WARMUP = 10
ITERATIONS = 30


checkpoint = torch.load(CHECKPOINT, map_location="cpu")

model = M3GSS_v0_32x8().cuda().eval()
model.load_state_dict(checkpoint["state_dict"])


print("=== M3GSS V1 — Benchmark multi-résolutions GTX 970 ===")
print("GPU:", torch.cuda.get_device_name(0))
print()

with torch.inference_mode():

    for width, height in RESOLUTIONS:

        x = torch.rand(
            1, 3, height, width,
            device="cuda",
            dtype=torch.float32,
        )

        for _ in range(WARMUP):
            model(x)

        torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(ITERATIONS):
            model(x)

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        ms = elapsed * 1000.0 / ITERATIONS
        fps = 1000.0 / ms

        print(
            f"{width}x{height} : "
            f"{ms:.3f} ms/frame | "
            f"{fps:.2f} FPS"
        )

        del x

print()
print("Benchmark terminé.")
