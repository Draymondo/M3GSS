"""Benchmark CUDA du modele M3GSS V2 a 960x540 -> 1920x1080."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.m3gss_v2.model import M3GSS_v2

LR_H, LR_W = 540, 960
HR_H, HR_W = 1080, 1920
BATCH_SIZE = 1
WARMUP_ITERATIONS = 20
MEASURED_ITERATIONS = 100
V1_REFERENCE_MS = 311.2
V1_REFERENCE_FPS = 3.21


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA n'est pas disponible dans cet environnement.")

    device = torch.device("cuda")
    print("=== M3GSS V2 — Benchmark runtime CUDA ===")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(f"GPU     : {torch.cuda.get_device_name(device)}")
    print(f"Device  : {device}")
    print("Dtype   : FP32")
    print(f"LR      : [{BATCH_SIZE}, 3, {LR_H}, {LR_W}]")
    print(f"HR      : [{BATCH_SIZE}, 3, {HR_H}, {HR_W}]")
    print(f"Warm-up : {WARMUP_ITERATIONS} iterations")
    print(f"Mesure  : {MEASURED_ITERATIONS} iterations")
    print()

    model = M3GSS_v2().to(device=device, dtype=torch.float32).eval()

    # Les donnees sont allouees avant le chronometrage.
    lr = torch.rand(
        BATCH_SIZE, 3, LR_H, LR_W, device=device, dtype=torch.float32
    )
    baseline_hr = torch.rand(
        BATCH_SIZE, 3, HR_H, HR_W, device=device, dtype=torch.float32
    )

    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            output = model(lr, baseline_hr)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(MEASURED_ITERATIONS)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(MEASURED_ITERATIONS)
    ]

    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        for index in range(MEASURED_ITERATIONS):
            start_events[index].record()
            output = model(lr, baseline_hr)
            end_events[index].record()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    frame_times_ms = [
        start_event.elapsed_time(end_event)
        for start_event, end_event in zip(start_events, end_events)
    ]
    average_ms = elapsed_ms / MEASURED_ITERATIONS
    fps = 1000.0 / average_ms

    expected_shape = (BATCH_SIZE, 3, HR_H, HR_W)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Forme de sortie incorrecte : {tuple(output.shape)} != {expected_shape}"
        )
    if not torch.isfinite(output).all().item():
        raise RuntimeError("La sortie contient des valeurs NaN ou Inf.")

    allocated_mib = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mib = torch.cuda.memory_reserved(device) / (1024**2)
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    speedup = V1_REFERENCE_MS / average_ms
    latency_reduction = (1.0 - average_ms / V1_REFERENCE_MS) * 100.0
    fps_multiplier = fps / V1_REFERENCE_FPS

    print("Résultats V2")
    print("-------------")
    print(f"Temps moyen/frame : {average_ms:.3f} ms (boucle synchronisée)")
    print(f"FPS théorique     : {fps:.2f}")
    print(f"Min CUDA          : {min(frame_times_ms):.3f} ms")
    print(f"Max CUDA          : {max(frame_times_ms):.3f} ms")
    print(f"Sortie            : {tuple(output.shape)}")
    print("Valeurs finies    : oui")
    print()
    print("Mémoire CUDA")
    print("------------")
    print(f"Allouée           : {allocated_mib:.1f} MiB")
    print(f"Réservée          : {reserved_mib:.1f} MiB")
    print(f"Pic alloué        : {peak_mib:.1f} MiB")
    print()
    print("V1 reference:")
    print(f"{V1_REFERENCE_MS:.1f} ms")
    print(f"{V1_REFERENCE_FPS:.2f} FPS")
    print()
    print(f"Speedup V2 vs V1 : {speedup:.2f}x")
    print(f"Réduction latence: {latency_reduction:.2f}%")
    print(f"FPS multiplié par: {fps_multiplier:.2f}x")


if __name__ == "__main__":
    main()