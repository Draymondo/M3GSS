import time
import torch
from m3gss_v0.model import M3GSS_v0_32x8

checkpoint = torch.load(
    "ml/checkpoints/m3gss_v1_big_best.pt",
    map_location="cpu"
)

model = M3GSS_v0_32x8().cuda().eval()
model.load_state_dict(checkpoint["state_dict"])

x = torch.rand(1, 3, 1080, 1920, device="cuda")

def benchmark(model, x, iterations=30):
    with torch.inference_mode():
        for _ in range(10):
            model(x)

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            model(x)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed * 1000 / iterations


print("=== FP32 ===")
fp32_ms = benchmark(model, x)
print(f"FP32 : {fp32_ms:.3f} ms")
print(f"FPS  : {1000 / fp32_ms:.2f}")

model.half()
x16 = x.half()

print()
print("=== FP16 ===")
fp16_ms = benchmark(model, x16)
print(f"FP16 : {fp16_ms:.3f} ms")
print(f"FPS  : {1000 / fp16_ms:.2f}")

print()
print(f"Accélération FP16 : {fp32_ms / fp16_ms:.2f}x")
