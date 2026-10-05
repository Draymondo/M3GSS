"""Prototype CUDA Catmull-Rom compatible avec la reference NumPy du projet."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from m3gss_v0.bicubic import (
    _inv_sum,
    _weights,
    _coords,
    catmull_rom_upscale,
)

SCALE = 2
CHANNELS = 3
BENCHMARK_LR_HEIGHT = 540
BENCHMARK_LR_WIDTH = 960
WARMUP_ITERATIONS = 5
BENCHMARK_ITERATIONS = 50


@dataclass
class AxisPlan:
    indices: torch.Tensor
    weights: torch.Tensor
    inverse_weight_sum: torch.Tensor


@dataclass
class CatmullRomPlan:
    source_height: int
    source_width: int
    destination_height: int
    destination_width: int
    horizontal: AxisPlan
    vertical: AxisPlan
    device: torch.device

    @classmethod
    def prepare(
        cls,
        source_height: int,
        source_width: int,
        destination_height: int,
        destination_width: int,
        device: torch.device,
    ) -> CatmullRomPlan:
        if min(source_height, source_width, destination_height, destination_width) < 1:
            raise ValueError("Les dimensions doivent etre positives")

        def prepare_axis(source_size: int, destination_size: int) -> AxisPlan:
            centers, bases = _coords(source_size, destination_size)
            weights, indices = _weights(centers, bases, source_size)
            inverse_weight_sum = _inv_sum(weights)
            return AxisPlan(
                indices=torch.from_numpy(np.ascontiguousarray(indices.T)).to(
                    device=device, dtype=torch.long
                ),
                weights=torch.from_numpy(np.ascontiguousarray(weights.T)).to(
                    device=device, dtype=torch.float32
                ),
                inverse_weight_sum=torch.from_numpy(
                    np.ascontiguousarray(inverse_weight_sum)
                ).to(device=device, dtype=torch.float32),
            )

        return cls(
            source_height=source_height,
            source_width=source_width,
            destination_height=destination_height,
            destination_width=destination_width,
            horizontal=prepare_axis(source_width, destination_width),
            vertical=prepare_axis(source_height, destination_height),
            device=device,
        )


def _separable_axis(
    source: torch.Tensor,
    axis_plan: AxisPlan,
    dimension: int,
    vertical: bool,
) -> torch.Tensor:
    accumulator = None
    for tap in range(4):
        sample = torch.index_select(source, dimension, axis_plan.indices[tap])
        if vertical:
            weight = axis_plan.weights[tap].view(1, 1, -1, 1)
            inverse_sum = axis_plan.inverse_weight_sum.view(1, 1, -1, 1)
        else:
            weight = axis_plan.weights[tap].view(1, 1, 1, -1)
            inverse_sum = axis_plan.inverse_weight_sum.view(1, 1, 1, -1)

        weighted_sample = weight * sample
        accumulator = (
            weighted_sample
            if accumulator is None
            else accumulator + weighted_sample
        )

    return accumulator * inverse_sum


def catmull_rom_upscale_cuda(
    lr_pixels_nchw: torch.Tensor,
    plan: CatmullRomPlan,
) -> torch.Tensor:
    """Upscale GPU FP32 pixels [N,3,H,W] en [N,3,2H,2W], valeurs en [0,1].

    Les pixels d'entree sont des valeurs uint8 converties en float32, dans
    l'echelle [0,255]. La sortie applique le meme +0.5, clamp et quantification
    uint8 que clamp_to_byte, puis renvoie le baseline float32 normalise [0,1].
    """
    if lr_pixels_nchw.ndim != 4 or lr_pixels_nchw.shape[1] != CHANNELS:
        raise ValueError("Entree attendue [N,3,H,W]")
    if lr_pixels_nchw.dtype != torch.float32:
        raise TypeError("L'entree CUDA doit etre en float32")
    if lr_pixels_nchw.device.type != plan.device.type or (
        plan.device.index is not None
        and lr_pixels_nchw.device.index != plan.device.index
    ):
        raise ValueError(f"Entree sur {lr_pixels_nchw.device}, plan sur {plan.device}")
    if tuple(lr_pixels_nchw.shape[-2:]) != (
        plan.source_height,
        plan.source_width,
    ):
        raise ValueError("La forme LR ne correspond pas au plan Catmull-Rom")

    horizontal = _separable_axis(
        lr_pixels_nchw,
        plan.horizontal,
        dimension=3,
        vertical=False,
    )
    high_resolution_pixels = _separable_axis(
        horizontal,
        plan.vertical,
        dimension=2,
        vertical=True,
    )

    rounded = high_resolution_pixels + 0.5
    quantized = rounded.clamp(0.0, 255.0).to(torch.uint8)
    return quantized.to(torch.float32).div_(255.0)


def _make_synthetic_image(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height, width, CHANNELS), dtype=np.uint8)
    image[:, width // 3 : width // 3 + 2] = 255
    image[height // 2 : height // 2 + 2, :] = 0
    return image


def _tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    return (
        image[0]
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )


def validate_parity(
    lr_image: np.ndarray,
    device: torch.device,
    label: str,
) -> dict[str, float | int | tuple[int, ...] | str]:
    height, width, channels = lr_image.shape
    if channels != CHANNELS:
        raise ValueError("Les tests de parite exigent une image RGB")

    destination_height = height * SCALE
    destination_width = width * SCALE
    plan = CatmullRomPlan.prepare(
        height, width, destination_height, destination_width, device
    )
    input_tensor = (
        torch.from_numpy(np.ascontiguousarray(lr_image.transpose(2, 0, 1)))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    with torch.inference_mode():
        cuda_float = catmull_rom_upscale_cuda(input_tensor, plan)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    cpu_output = catmull_rom_upscale(lr_image, destination_width, destination_height)
    cuda_output = _tensor_to_uint8_hwc(cuda_float)
    difference = np.abs(cpu_output.astype(np.int16) - cuda_output.astype(np.int16))
    maximum_error = int(difference.max())
    mean_error = float(difference.mean())
    different_pixels = int(np.count_nonzero(np.any(difference != 0, axis=2)))
    if cuda_output.shape != cpu_output.shape:
        raise RuntimeError(
            f"Sortie CUDA {cuda_output.shape} != sortie CPU {cpu_output.shape}"
        )
    if cuda_output.min() < 0 or cuda_output.max() > 255:
        raise RuntimeError("La sortie quantifiee CUDA sort de la plage [0,255]")
    if not torch.isfinite(cuda_float).all().item():
        raise RuntimeError("La sortie CUDA contient NaN/Inf")

    result = {
        "label": label,
        "shape": cuda_output.shape,
        "dtype": str(cuda_float.dtype),
        "max_abs_error": maximum_error,
        "mean_abs_error": mean_error,
        "different_pixels": different_pixels,
        "different_values": int(np.count_nonzero(difference)),
    }
    return result


def _to_gpu_pixels(lr_image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(np.ascontiguousarray(lr_image.transpose(2, 0, 1)))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def benchmark_large(device: torch.device) -> None:
    lr_image = _make_synthetic_image(
        BENCHMARK_LR_HEIGHT, BENCHMARK_LR_WIDTH, seed=20261005
    )

    torch.cuda.synchronize(device)
    preparation_start = time.perf_counter()
    plan = CatmullRomPlan.prepare(
        BENCHMARK_LR_HEIGHT,
        BENCHMARK_LR_WIDTH,
        BENCHMARK_LR_HEIGHT * SCALE,
        BENCHMARK_LR_WIDTH * SCALE,
        device,
    )
    torch.cuda.synchronize(device)
    preparation_ms = (time.perf_counter() - preparation_start) * 1000.0

    torch.cuda.synchronize(device)
    upload_start = time.perf_counter()
    lr_gpu = _to_gpu_pixels(lr_image, device)
    torch.cuda.synchronize(device)
    upload_ms = (time.perf_counter() - upload_start) * 1000.0

    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            output = catmull_rom_upscale_cuda(lr_gpu, plan)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    start_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(BENCHMARK_ITERATIONS)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True) for _ in range(BENCHMARK_ITERATIONS)
    ]

    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    with torch.inference_mode():
        for index in range(BENCHMARK_ITERATIONS):
            start_events[index].record()
            output = catmull_rom_upscale_cuda(lr_gpu, plan)
            end_events[index].record()
    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - total_start) * 1000.0

    event_times = [
        start.elapsed_time(end)
        for start, end in zip(start_events, end_events)
    ]
    cuda_average_ms = float(np.mean(event_times))
    synchronized_loop_average_ms = total_ms / BENCHMARK_ITERATIONS
    comparable_total_ms = upload_ms + cuda_average_ms
    speedup = 335.7 / comparable_total_ms if comparable_total_ms > 0 else 0.0
    allocated_mib = torch.cuda.memory_allocated(device) / (1024**2)
    reserved_mib = torch.cuda.memory_reserved(device) / (1024**2)
    peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)

    expected_shape = (1, CHANNELS, BENCHMARK_LR_HEIGHT * 2, BENCHMARK_LR_WIDTH * 2)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Forme benchmark incorrecte: {tuple(output.shape)}")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Sortie benchmark NaN/Inf")

    print()
    print("=== Benchmark CUDA Catmull-Rom 960x540 -> 1920x1080 ===")
    print(f"GPU                          : {torch.cuda.get_device_name(device)}")
    print(f"Dtype                        : {lr_gpu.dtype}")
    print(f"Préparation indices/poids    : {preparation_ms:.3f} ms (one-shot)")
    print(f"Upload LR CPU->CUDA          : {upload_ms:.3f} ms/frame")
    print(f"Catmull-Rom CUDA moyenne     : {cuda_average_ms:.3f} ms/frame (CUDA events)")
    print(f"Boucle synchronisée moyenne  : {synchronized_loop_average_ms:.3f} ms/frame")
    print(f"Total comparable/frame       : {comparable_total_ms:.3f} ms (upload + CUDA, plan amorti)")
    print(f"Référence CPU annoncée       : 335.700 ms/frame")
    print(f"Speedup incluant upload      : {speedup:.2f}x")
    print(f"Sortie                       : {tuple(output.shape)} float32 [0,1]")
    print(f"VRAM allouée                 : {allocated_mib:.1f} MiB")
    print(f"VRAM réservée                : {reserved_mib:.1f} MiB")
    print(f"Pic VRAM allouée             : {peak_allocated_mib:.1f} MiB")
    print(f"Pic VRAM réservée            : {peak_reserved_mib:.1f} MiB")
    print(f"Iterations mesurées          : {BENCHMARK_ITERATIONS}")
    del lr_gpu, output, plan


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA est requis pour ce prototype")
    device = torch.device("cuda")
    print("=== Parité Catmull-Rom CPU/CUDA ===")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    tests = (
        (17, 23, 1, "petite image"),
        (48, 48, 2, "patch 48x48"),
        (BENCHMARK_LR_HEIGHT, BENCHMARK_LR_WIDTH, 3, "résolution benchmark"),
    )
    for height, width, seed, label in tests:
        lr_image = _make_synthetic_image(height, width, seed)
        result = validate_parity(lr_image, device, label)
        print(
            f"{result['label']}: shape={result['shape']}, dtype={result['dtype']}, "
            f"max_abs={result['max_abs_error']}, "
            f"mean_abs={result['mean_abs_error']:.6f}, "
            f"pixels_diff={result['different_pixels']}, "
            f"values_diff={result['different_values']}"
        )

    benchmark_large(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())