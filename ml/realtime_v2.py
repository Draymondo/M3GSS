"""Prototype de pipeline M3GSS V2 sur une sequence d'images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from catmull_rom_cuda import CatmullRomPlan, catmull_rom_upscale_cuda
from m3gss_v0.bicubic import catmull_rom_upscale, make_test_image
from m3gss_v0.patches import from_tensor, to_tensor
from m3gss_v2.model import M3GSS_v2, count_parameters

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "m3gss_v2_best.pt"
LR_WIDTH, LR_HEIGHT = 960, 540
HR_WIDTH, HR_HEIGHT = 1920, 1080
WARMUP_ITERATIONS = 3
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CPU_CUDA_PARITY_TOLERANCE = 0


def load_model(device: torch.device) -> M3GSS_v2:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint V2 introuvable : {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or checkpoint.get("architecture") != "M3GSS_v2":
        raise RuntimeError("Le checkpoint ne declare pas l'architecture M3GSS_v2")
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError("Le checkpoint V2 ne contient pas de state_dict valide")

    model = M3GSS_v2()
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Poids du checkpoint incompatibles avec M3GSS_v2 : {exc}") from exc

    model.to(device=device, dtype=torch.float32)
    model.eval()
    if model.training:
        raise RuntimeError("Le modele doit etre en mode eval()")
    return model


def save_comparison(path: Path, baseline: np.ndarray, output: np.ndarray) -> None:
    label_height = 36
    comparison = Image.new(
        "RGB", (HR_WIDTH * 2, HR_HEIGHT + label_height), (0, 0, 0)
    )
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), "Catmull-Rom", fill=(255, 255, 255))
    draw.text((HR_WIDTH + 10, 10), "M3GSS V2", fill=(255, 255, 255))
    comparison.paste(Image.fromarray(baseline, mode="RGB"), (0, label_height))
    comparison.paste(Image.fromarray(output, mode="RGB"), (HR_WIDTH, label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(path, format="PNG")


def list_images(input_dir: Path, output_dir: Path | None) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Dossier d'entree introuvable : {input_dir}")

    output_root = output_dir.resolve() if output_dir is not None else None
    paths = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if output_root is not None:
            try:
                path.resolve().relative_to(output_root)
            except ValueError:
                pass
            else:
                continue
        paths.append(path)
    return paths


def verify_cuda_parity(device: torch.device) -> dict[str, float | int]:
    lr_image = make_test_image(width=48, height=48)
    cpu_baseline = catmull_rom_upscale(lr_image, 96, 96)
    plan = CatmullRomPlan.prepare(48, 48, 96, 96, device)
    lr_cpu = to_tensor(lr_image).unsqueeze(0)

    torch.cuda.synchronize(device)
    lr_gpu = lr_cpu.to(device=device, dtype=torch.float32)
    lr_pixels_gpu = lr_gpu * 255.0
    with torch.inference_mode():
        cuda_baseline = catmull_rom_upscale_cuda(lr_pixels_gpu, plan)
    torch.cuda.synchronize(device)

    cuda_baseline_image = from_tensor(cuda_baseline.squeeze(0))
    difference = np.abs(
        cpu_baseline.astype(np.int16) - cuda_baseline_image.astype(np.int16)
    )
    result = {
        "max_abs": int(difference.max()),
        "mean_abs": float(difference.mean()),
        "different_pixels": int(np.count_nonzero(np.any(difference != 0, axis=2))),
    }
    print(
        "Parité CPU/CUDA (48x48 -> 96x96): "
        f"max_abs={result['max_abs']}, mean_abs={result['mean_abs']:.6f}, "
        f"pixels_diff={result['different_pixels']}"
    )
    if result["max_abs"] > CPU_CUDA_PARITY_TOLERANCE:
        raise RuntimeError("Parité Catmull-Rom CPU/CUDA échouée; benchmark interrompu")
    del plan, lr_cpu, lr_gpu, lr_pixels_gpu, cuda_baseline
    return result


def run(args: argparse.Namespace) -> None:
    image_paths = list_images(args.input, args.output)
    if args.max_frames is not None:
        image_paths = image_paths[:args.max_frames]
    if not image_paths:
        raise RuntimeError(f"Aucune image a traiter dans {args.input}")

    if args.catmull_rom_backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Le backend Catmull-Rom CUDA exige un GPU CUDA disponible")
    if args.resize_backend == "cuda" and (
        not torch.cuda.is_available() or args.catmull_rom_backend != "cuda"
    ):
        raise RuntimeError(
            "Le resize CUDA exige CUDA et --catmull-rom-backend cuda"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    parity_result = None

    if args.catmull_rom_backend == "cuda":
        parity_result = verify_cuda_parity(device)

    catmull_plan = None
    plan_prepare_ms = 0.0
    if args.catmull_rom_backend == "cuda":
        torch.cuda.synchronize(device)
        plan_start = time.perf_counter()
        catmull_plan = CatmullRomPlan.prepare(
            LR_HEIGHT, LR_WIDTH, HR_HEIGHT, HR_WIDTH, device
        )
        torch.cuda.synchronize(device)
        plan_prepare_ms = (time.perf_counter() - plan_start) * 1000.0

    print("=== M3GSS V2 — Pipeline sur séquence d'images ===")
    print(f"GPU          : {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'indisponible'}")
    print(f"Device       : {device}")
    print(f"Backend Catmull-Rom : {args.catmull_rom_backend.upper()}")
    print(f"Backend resize LR : {args.resize_backend.upper()}")
    print("Dtype        : FP32")
    print(f"Checkpoint   : {CHECKPOINT_PATH}")
    print(f"Paramètres   : {count_parameters(model):,}")
    print(f"Mode eval    : {not model.training}")
    print(f"Inference mode: torch.inference_mode()")
    print(f"Images       : {len(image_paths)}")
    print(f"Sortie       : {args.output if args.output else 'non sauvegardee'}")
    print(f"Warm-up GPU  : {WARMUP_ITERATIONS} iterations")
    if catmull_plan is not None:
        print(f"Préparation plan CUDA : {plan_prepare_ms:.3f} ms (one-shot)")

    if device.type == "cuda":
        warm_source = torch.zeros(1, 3, HR_HEIGHT, HR_WIDTH, device=device)
        warm_lr = torch.zeros(1, 3, LR_HEIGHT, LR_WIDTH, device=device)
        with torch.inference_mode():
            for _ in range(WARMUP_ITERATIONS):
                if args.resize_backend == "cuda":
                    warm_pixels = F.interpolate(
                        warm_source,
                        size=(LR_HEIGHT, LR_WIDTH),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ).round().clamp(0.0, 255.0)
                    warm_lr = warm_pixels / 255.0
                if args.catmull_rom_backend == "cuda":
                    warm_input_pixels = (
                        warm_pixels
                        if args.resize_backend == "cuda"
                        else warm_lr * 255.0
                    )
                    warm_baseline = catmull_rom_upscale_cuda(
                        warm_input_pixels, catmull_plan
                    )
                else:
                    warm_baseline = torch.zeros(
                        1, 3, HR_HEIGHT, HR_WIDTH, device=device
                    )
                model(warm_lr, warm_baseline)
        torch.cuda.synchronize(device)
        del warm_source, warm_lr, warm_baseline
        if args.resize_backend == "cuda":
            del warm_pixels, warm_input_pixels
        torch.cuda.reset_peak_memory_stats(device)

    timings = {
        "read_decode": [],
        "rgb_conversion": [],
        "resize_cpu": [],
        "tensor_cpu": [],
        "resize_gpu": [],
        "preprocessing": [],
        "transfer": [],
        "catmull_rom": [],
        "network": [],
        "post_gpu_quantize": [],
        "post_gpu_layout": [],
        "post_gpu_to_cpu": [],
        "post_cpu_convert": [],
        "postprocessing": [],
        "pipeline": [],
        "end_to_end": [],
    }
    benchmark_start = time.perf_counter()

    for index, image_path in enumerate(image_paths, start=1):
        frame_start = time.perf_counter()
        read_start = time.perf_counter()
        with Image.open(image_path) as source_image:
            source_image.load()
            decoded_image = source_image.copy()
        timings["read_decode"].append((time.perf_counter() - read_start) * 1000.0)

        pipeline_start = time.perf_counter()
        rgb_start = time.perf_counter()
        source_rgb = decoded_image.convert("RGB")
        rgb_ms = (time.perf_counter() - rgb_start) * 1000.0
        timings["rgb_conversion"].append(rgb_ms)

        lr_image = None
        lr_tensor_cpu = None
        source_tensor_cpu = None
        if args.resize_backend == "pil":
            resize_start = time.perf_counter()
            lr_pil = source_rgb.resize(
                (LR_WIDTH, LR_HEIGHT),
                resample=Image.Resampling.LANCZOS,
            )
            lr_image = np.asarray(lr_pil, dtype=np.uint8)
            resize_cpu_ms = (time.perf_counter() - resize_start) * 1000.0
            if lr_image.shape != (LR_HEIGHT, LR_WIDTH, 3):
                raise RuntimeError(
                    f"Dimensions LR incorrectes pour {image_path.name}: {lr_image.shape}"
                )

            tensor_start = time.perf_counter()
            lr_tensor_cpu = to_tensor(lr_image).unsqueeze(0)
            tensor_cpu_ms = (time.perf_counter() - tensor_start) * 1000.0
            lr_pixels_gpu = None
        else:
            resize_cpu_ms = 0.0
            tensor_start = time.perf_counter()
            source_array = np.array(source_rgb, dtype=np.uint8, copy=True)
            source_tensor_cpu = torch.from_numpy(source_array).unsqueeze(0)
            tensor_cpu_ms = (time.perf_counter() - tensor_start) * 1000.0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_start = time.perf_counter()
        if args.resize_backend == "pil":
            lr_tensor = lr_tensor_cpu.to(device=device, dtype=torch.float32)
        else:
            source_tensor_gpu = source_tensor_cpu.to(device=device)
            source_tensor_gpu = source_tensor_gpu.permute(0, 3, 1, 2)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        transfer_ms = (time.perf_counter() - transfer_start) * 1000.0
        timings["preprocessing"].append(rgb_ms + resize_cpu_ms + tensor_cpu_ms)
        timings["resize_cpu"].append(resize_cpu_ms)
        timings["tensor_cpu"].append(tensor_cpu_ms)

        resize_gpu_ms = 0.0
        if args.resize_backend == "cuda":
            torch.cuda.synchronize(device)
            resize_gpu_start = time.perf_counter()
            resized_pixels = F.interpolate(
                source_tensor_gpu.float(),
                size=(LR_HEIGHT, LR_WIDTH),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).round().clamp(0.0, 255.0)
            lr_pixels_gpu = resized_pixels
            lr_tensor = lr_pixels_gpu / 255.0
            torch.cuda.synchronize(device)
            resize_gpu_ms = (time.perf_counter() - resize_gpu_start) * 1000.0
        timings["resize_gpu"].append(resize_gpu_ms)
        baseline_image = None

        if args.catmull_rom_backend == "cuda":
            torch.cuda.synchronize(device)
            catmull_start = time.perf_counter()
            if lr_pixels_gpu is None:
                lr_pixels_gpu = lr_tensor * 255.0
            with torch.inference_mode():
                baseline_tensor = catmull_rom_upscale_cuda(
                    lr_pixels_gpu, catmull_plan
                )
            torch.cuda.synchronize(device)
            catmull_ms = (time.perf_counter() - catmull_start) * 1000.0
        else:
            catmull_start = time.perf_counter()
            baseline_image = catmull_rom_upscale(lr_image, HR_WIDTH, HR_HEIGHT)
            if baseline_image.shape != (HR_HEIGHT, HR_WIDTH, 3):
                raise RuntimeError(
                    f"Dimensions Catmull-Rom incorrectes pour {image_path.name}: "
                    f"{baseline_image.shape}"
                )
            baseline_tensor_cpu = to_tensor(baseline_image).unsqueeze(0)
            catmull_ms = (time.perf_counter() - catmull_start) * 1000.0

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            baseline_transfer_start = time.perf_counter()
            baseline_tensor = baseline_tensor_cpu.to(
                device=device, dtype=torch.float32
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            transfer_ms += (time.perf_counter() - baseline_transfer_start) * 1000.0

        timings["transfer"].append(transfer_ms)
        timings["catmull_rom"].append(catmull_ms)

        expected_baseline_shape = (1, 3, HR_HEIGHT, HR_WIDTH)
        if tuple(baseline_tensor.shape) != expected_baseline_shape:
            raise RuntimeError(
                f"Dimensions baseline incorrectes pour {image_path.name}: "
                f"{tuple(baseline_tensor.shape)}"
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        network_start = time.perf_counter()
        with torch.inference_mode():
            if not torch.is_inference_mode_enabled():
                raise RuntimeError("L'inference doit utiliser torch.inference_mode()")
            output_tensor = model(lr_tensor, baseline_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings["network"].append((time.perf_counter() - network_start) * 1000.0)

        expected_output_shape = (1, 3, HR_HEIGHT, HR_WIDTH)
        if tuple(output_tensor.shape) != expected_output_shape:
            raise RuntimeError(
                f"Dimensions de sortie incorrectes pour {image_path.name}: "
                f"{tuple(output_tensor.shape)}"
            )
        if not torch.isfinite(output_tensor).all().item():
            raise RuntimeError(f"Sortie NaN/Inf pour {image_path.name}")

        postprocessing_start = time.perf_counter()
        if args.postprocess_backend == "gpu":
            if device.type != "cuda":
                raise RuntimeError("Le postprocessing GPU exige CUDA")

            torch.cuda.synchronize(device)
            quantize_start = time.perf_counter()
            output_bytes = (
                output_tensor[0]
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
            )
            torch.cuda.synchronize(device)
            timings["post_gpu_quantize"].append(
                (time.perf_counter() - quantize_start) * 1000.0
            )

            layout_start = time.perf_counter()
            output_hwc_gpu = output_bytes.permute(1, 2, 0).contiguous()
            torch.cuda.synchronize(device)
            timings["post_gpu_layout"].append(
                (time.perf_counter() - layout_start) * 1000.0
            )

            transfer_start = time.perf_counter()
            output_image = output_hwc_gpu.cpu().numpy()
            torch.cuda.synchronize(device)
            timings["post_gpu_to_cpu"].append(
                (time.perf_counter() - transfer_start) * 1000.0
            )
            timings["post_cpu_convert"].append(0.0)
            del output_bytes, output_hwc_gpu
        else:
            cpu_convert_start = time.perf_counter()
            output_image = from_tensor(output_tensor.squeeze(0))
            timings["post_cpu_convert"].append(
                (time.perf_counter() - cpu_convert_start) * 1000.0
            )
            timings["post_gpu_quantize"].append(0.0)
            timings["post_gpu_layout"].append(0.0)
            timings["post_gpu_to_cpu"].append(0.0)
        if output_image.shape != (HR_HEIGHT, HR_WIDTH, 3):
            raise RuntimeError(
                f"Dimensions post-traitement incorrectes pour {image_path.name}: "
                f"{output_image.shape}"
            )
        if args.output is not None:
            if baseline_image is None:
                baseline_image = from_tensor(baseline_tensor.squeeze(0))
            output_path = args.output / f"{index:05d}_{image_path.stem}_comparison.png"
            save_comparison(output_path, baseline_image, output_image)
        timings["postprocessing"].append(
            (time.perf_counter() - postprocessing_start) * 1000.0
        )

        timings["pipeline"].append((time.perf_counter() - pipeline_start) * 1000.0)
        timings["end_to_end"].append((time.perf_counter() - frame_start) * 1000.0)
        if index % 5 == 0 or index == len(image_paths):
            print(f"Progression : {index}/{len(image_paths)} — {image_path.name}")

        del decoded_image, lr_tensor, baseline_tensor, output_tensor
        if args.catmull_rom_backend == "cuda":
            del lr_pixels_gpu
        if lr_tensor_cpu is not None:
            del lr_tensor_cpu
        if source_tensor_cpu is not None:
            del source_tensor_cpu, source_tensor_gpu

    total_elapsed = time.perf_counter() - benchmark_start
    frame_count = len(image_paths)
    average_pipeline_ms = float(np.mean(timings["pipeline"]))
    pipeline_fps = 1000.0 / average_pipeline_ms if average_pipeline_ms > 0 else 0.0
    effective_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0

    print()
    print(f"=== Résumé M3GSS V2 — Catmull-Rom {args.catmull_rom_backend.upper()} ===")
    print(f"Frames traitées            : {frame_count}")
    print(f"Lecture/décodage moyen     : {np.mean(timings['read_decode']):.3f} ms/frame (hors latence pipeline)")
    print(f"Conversion RGB moyenne     : {np.mean(timings['rgb_conversion']):.3f} ms/frame")
    print(f"Resize CPU moyen           : {np.mean(timings['resize_cpu']):.3f} ms/frame")
    print(f"Tensor CPU moyen           : {np.mean(timings['tensor_cpu']):.3f} ms/frame")
    print(f"Preprocessing moyen        : {np.mean(timings['preprocessing']):.3f} ms/frame (RGB + resize LR + tenseur CPU)")
    print(f"Transfert CPU→GPU moyen    : {np.mean(timings['transfer']):.3f} ms/frame")
    print(f"Resize GPU moyen           : {np.mean(timings['resize_gpu']):.3f} ms/frame")
    print(f"Catmull-Rom {args.catmull_rom_backend.upper()} moyen : {np.mean(timings['catmull_rom']):.3f} ms/frame")
    print(f"Temps réseau moyen         : {np.mean(timings['network']):.3f} ms/frame")
    print(f"Temps postprocessing moyen : {np.mean(timings['postprocessing']):.3f} ms/frame")
    print(f"Post GPU quantification    : {np.mean(timings['post_gpu_quantize']):.3f} ms/frame")
    print(f"Post GPU HWC layout        : {np.mean(timings['post_gpu_layout']):.3f} ms/frame")
    print(f"Post GPU→CPU               : {np.mean(timings['post_gpu_to_cpu']):.3f} ms/frame")
    print(f"Post CPU conversion        : {np.mean(timings['post_cpu_convert']):.3f} ms/frame")
    print(f"Backend postprocessing     : {args.postprocess_backend.upper()}")
    print(f"Latence pipeline moyenne   : {average_pipeline_ms:.3f} ms/frame (hors lecture disque)")
    print(f"FPS théorique pipeline     : {pipeline_fps:.3f}")
    print(f"Latence moyenne complète   : {np.mean(timings['end_to_end']):.3f} ms/frame (avec lecture disque)")
    print(f"Temps total benchmark      : {total_elapsed:.3f} s")
    print(f"FPS effectif end-to-end    : {effective_fps:.3f}")
    print(f"Prétraitement LR           : RGB -> {LR_WIDTH}x{LR_HEIGHT} (Pillow LANCZOS)")
    print(f"Sortie vérifiée            : {HR_WIDTH}x{HR_HEIGHT}, valeurs finies")
    print(f"Modèle eval/inference_mode : {not model.training}/True")
    if parity_result is not None:
        print(
            "Parité CPU/CUDA (48x48)  : "
            f"max_abs={parity_result['max_abs']}, "
            f"mean_abs={parity_result['mean_abs']:.6f}, "
            f"pixels_diff={parity_result['different_pixels']}"
        )
    if args.output is not None:
        print(f"Comparaisons sauvegardées  : {args.output}")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
        print(f"VRAM allouée               : {allocated:.1f} MiB")
        print(f"VRAM réservée              : {reserved:.1f} MiB")
        print(f"Pic VRAM allouée           : {peak_allocated:.1f} MiB")
        print(f"Pic VRAM réservée          : {peak_reserved:.1f} MiB")
    else:
        print("VRAM GPU                   : indisponible (exécution CPU)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline M3GSS V2 sur un dossier de frames image"
    )
    parser.add_argument("--input", required=True, type=Path, help="Dossier source d'images")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Dossier optionnel pour les comparaisons Catmull-Rom/V2",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Nombre maximal d'images à traiter",
    )
    parser.add_argument(
        "--catmull-rom-backend",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Backend explicite Catmull-Rom (défaut: cpu pour préserver le comportement)",
    )
    parser.add_argument(
        "--postprocess-backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Quantification/layout de sortie CPU ou GPU (défaut: cpu)",
    )
    parser.add_argument(
        "--resize-backend",
        choices=("pil", "cuda"),
        default="pil",
        help="Resize LR PIL LANCZOS ou PyTorch CUDA bicubic antialias (défaut: pil)",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames doit etre superieur ou egal a 1")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())