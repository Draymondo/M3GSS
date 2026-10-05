"""Viewer interactif M3GSS V2 par capture d'une region du bureau Windows."""

from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageGrab, ImageTk

from catmull_rom_cuda import CatmullRomPlan, catmull_rom_upscale_cuda
from realtime_v2 import CHECKPOINT_PATH, LR_HEIGHT, LR_WIDTH, load_model
from realtime_v2 import HR_HEIGHT, HR_WIDTH
from realtime_v2 import verify_cuda_parity
from m3gss_v2.model import count_parameters

FPS_WINDOW = 60


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    try:
        coordinates = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bbox attendu sous la forme left,top,right,bottom") from exc
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("bbox attendu sous la forme left,top,right,bottom")
    left, top, right, bottom = coordinates
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise argparse.ArgumentTypeError("bbox invalide")
    return coordinates


def parse_dimensions(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("taille attendue sous la forme WIDTHxHEIGHT") from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("les dimensions doivent etre positives")
    return width, height


class RealtimeCaptureViewer:
    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        preview_size: tuple[int, int],
        max_frames: int | None,
        start_mode: str,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Le viewer M3GSS V2 exige CUDA")

        self.device = torch.device("cuda")
        self.bbox = bbox
        self.preview_size = preview_size
        self.max_frames = max_frames
        self.frame_count = 0
        self.mode = start_mode
        self.frame_durations: deque[float] = deque(maxlen=FPS_WINDOW)
        self.latencies: deque[float] = deque(maxlen=FPS_WINDOW)
        self.first_display_time: float | None = None
        self.last_display_time: float | None = None
        self.stage_samples: dict[str, list[float]] = {
            name: []
            for name in (
                "capture",
                "rgb_resize_cpu",
                "tensor_cpu",
                "transfer",
                "resize_gpu",
                "catmull_rom",
                "network",
                "post_quantize",
                "post_layout",
                "post_to_cpu",
                "display",
                "total",
            )
        }

        self.model = load_model(self.device)
        self.plan = CatmullRomPlan.prepare(
            LR_HEIGHT, LR_WIDTH, HR_HEIGHT, HR_WIDTH, self.device
        )
        self.parity = verify_cuda_parity(self.device)
        self._warm_up()

        self.root = tk.Tk()
        self.root.title("M3GSS V2 — capture interactive")
        preview_width, preview_height = self.preview_size
        left, top, right, _bottom = self.bbox
        screen_width = self.root.winfo_screenwidth()
        window_left = right if right + preview_width <= screen_width else max(0, left)
        self.root.geometry(
            f"{preview_width}x{preview_height + 58}+{window_left}+{max(0, top)}"
        )
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress-m>", self.toggle_m3gss)
        self.root.bind("<KeyPress-b>", self.toggle_baseline)
        self.root.bind("<KeyPress-Escape>", self.close)

        toolbar = ttk.Frame(self.root, padding=(8, 5))
        toolbar.pack(fill="x")
        self.status = ttk.Label(toolbar, text="Initialisation…")
        self.status.pack(side="left", fill="x", expand=True)
        ttk.Label(toolbar, text="M: source/V2 | B: baseline/V2 | Esc: quitter").pack(
            side="right"
        )
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(fill="both", expand=True)
        self.photo = None
        self.running = True
        self.root.after(1, self.process_next_frame)

        print(f"GPU                : {torch.cuda.get_device_name(self.device)}")
        print(f"Checkpoint         : {CHECKPOINT_PATH}")
        print(f"Paramètres         : {count_parameters(self.model):,}")
        print(f"Capture ROI        : {self.bbox[2] - self.bbox[0]}x{self.bbox[3] - self.bbox[1]} @ {self.bbox}")
        print(f"Résolution V2      : {LR_WIDTH}x{LR_HEIGHT} -> {HR_WIDTH}x{HR_HEIGHT}")
        print(f"Parité CPU/CUDA    : max={self.parity['max_abs']}, pixels_diff={self.parity['different_pixels']}")
        print(f"Mode initial       : {self.mode}")

    def _warm_up(self) -> None:
        warm_lr_pixels = torch.zeros(
            1, 3, LR_HEIGHT, LR_WIDTH, device=self.device, dtype=torch.float32
        )
        warm_lr = warm_lr_pixels / 255.0
        with torch.inference_mode():
            for _ in range(5):
                baseline = catmull_rom_upscale_cuda(warm_lr_pixels, self.plan)
                self.model(warm_lr, baseline)
        torch.cuda.synchronize(self.device)
        del warm_lr_pixels, warm_lr, baseline
        torch.cuda.reset_peak_memory_stats(self.device)

    def toggle_m3gss(self, _event=None) -> None:
        self.mode = "source" if self.mode != "source" else "v2"

    def toggle_baseline(self, _event=None) -> None:
        self.mode = "baseline" if self.mode != "baseline" else "v2"

    def close(self, _event=None) -> None:
        self.running = False
        if self.root.winfo_exists():
            self.root.destroy()

    def _capture_source(self) -> Image.Image:
        return ImageGrab.grab(bbox=self.bbox)

    def _capture_timed(self) -> tuple[Image.Image, float, float]:
        capture_started = time.perf_counter()
        image = self._capture_source()
        captured_at = time.perf_counter()
        return image, (captured_at - capture_started) * 1000.0, capture_started

    def _to_lr_cpu(self, captured: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        if captured.size != (LR_WIDTH, LR_HEIGHT):
            captured = captured.resize(
                (LR_WIDTH, LR_HEIGHT),
                resample=Image.Resampling.LANCZOS,
            )
        lr_hwc = np.array(captured, dtype=np.uint8, copy=True)
        lr_nhwc = torch.from_numpy(lr_hwc).unsqueeze(0)
        return lr_hwc, lr_nhwc

    def _run_gpu_pipeline(
        self,
        lr_nhwc_cpu: torch.Tensor,
        timings: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        source_gpu = lr_nhwc_cpu.to(self.device)
        torch.cuda.synchronize(self.device)
        timings["transfer"] = (time.perf_counter() - start) * 1000.0

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        lr_chw_pixels = source_gpu.permute(0, 3, 1, 2).float()
        if tuple(lr_chw_pixels.shape[-2:]) != (LR_HEIGHT, LR_WIDTH):
            lr_chw_pixels = F.interpolate(
                lr_chw_pixels,
                size=(LR_HEIGHT, LR_WIDTH),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        torch.cuda.synchronize(self.device)
        timings["resize_gpu"] = (time.perf_counter() - start) * 1000.0

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            baseline = catmull_rom_upscale_cuda(lr_chw_pixels, self.plan)
        torch.cuda.synchronize(self.device)
        timings["catmull_rom"] = (time.perf_counter() - start) * 1000.0

        if self.mode == "baseline":
            return baseline, baseline

        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(lr_chw_pixels / 255.0, baseline)
        torch.cuda.synchronize(self.device)
        timings["network"] = (time.perf_counter() - start) * 1000.0
        if tuple(output.shape) != (1, 3, HR_HEIGHT, HR_WIDTH):
            raise RuntimeError(f"Sortie V2 incorrecte: {tuple(output.shape)}")
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Sortie V2 contient NaN/Inf")
        return output, baseline

    def _gpu_to_pil(self, output: torch.Tensor, timings: dict[str, float]) -> Image.Image:
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        pixels = output[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        torch.cuda.synchronize(self.device)
        timings["post_quantize"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        hwc = pixels.permute(1, 2, 0).contiguous()
        torch.cuda.synchronize(self.device)
        timings["post_layout"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        image_array = hwc.cpu().numpy()
        torch.cuda.synchronize(self.device)
        timings["post_to_cpu"] = (time.perf_counter() - start) * 1000.0
        return Image.fromarray(image_array, mode="RGB")

    def _present(self, image: Image.Image, mode_label: str, timings: dict[str, float]) -> None:
        display_start = time.perf_counter()
        if image.size != self.preview_size:
            image = image.resize(self.preview_size, resample=Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        self.root.update_idletasks()
        timings["display"] = (time.perf_counter() - display_start) * 1000.0

        total_ms = sum(timings.values())
        now = time.perf_counter()
        self.frame_durations.append(now)
        if self.first_display_time is None:
            self.first_display_time = now
        self.last_display_time = now
        self.latencies.append(total_ms)
        fps_instant = 1000.0 / total_ms if total_ms > 0 else 0.0
        if len(self.frame_durations) > 1:
            fps_average = (len(self.frame_durations) - 1) / (
                self.frame_durations[-1] - self.frame_durations[0]
            )
        else:
            fps_average = fps_instant

        self.status.configure(
            text=(
                f"{mode_label} | frame {self.frame_count} | "
                f"FPS {fps_instant:.1f} / moy {fps_average:.1f} | "
                f"latence {total_ms:.1f} ms | "
                f"capture {timings['capture']:.1f} | net {timings['network']:.1f} ms"
            )
        )

    def process_next_frame(self) -> None:
        if not self.running:
            return
        if self.max_frames is not None and self.frame_count >= self.max_frames:
            self.print_summary()
            self.close()
            return

        timings = {name: 0.0 for name in self.stage_samples}
        captured, capture_ms, captured_at = self._capture_timed()
        timings["capture"] = capture_ms

        start = time.perf_counter()
        captured = captured.convert("RGB")
        if captured.size != (LR_WIDTH, LR_HEIGHT):
            captured = captured.resize(
                (LR_WIDTH, LR_HEIGHT),
                resample=Image.Resampling.LANCZOS,
            )
        timings["rgb_resize_cpu"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        lr_hwc = np.array(captured, dtype=np.uint8, copy=True)
        lr_nhwc_cpu = torch.from_numpy(lr_hwc).unsqueeze(0)
        timings["tensor_cpu"] = (time.perf_counter() - start) * 1000.0

        if self.mode == "source":
            display_image = Image.fromarray(lr_hwc, mode="RGB")
        else:
            output, baseline = self._run_gpu_pipeline(lr_nhwc_cpu, timings)
            display_image = self._gpu_to_pil(
                baseline if self.mode == "baseline" else output,
                timings,
            )

        self._present(display_image, self.mode.upper(), timings)
        timings["total"] = (time.perf_counter() - captured_at) * 1000.0
        for name, value in timings.items():
            self.stage_samples[name].append(value)
        self.frame_count += 1

        if self.frame_count % 60 == 0:
            print(
                f"Frame {self.frame_count}: capture->display={timings['total']:.2f}ms, "
                f"FPS moyen={self._average_fps():.2f}"
            )
        self.root.after(1, self.process_next_frame)

    def _average_fps(self) -> float:
        if len(self.frame_durations) < 2:
            return 0.0
        elapsed = self.frame_durations[-1] - self.frame_durations[0]
        return (len(self.frame_durations) - 1) / elapsed if elapsed > 0 else 0.0

    def print_summary(self) -> None:
        if self.frame_count == 0:
            return
        print("=== Résumé capture M3GSS V2 ===")
        print(f"Frames                 : {self.frame_count}")
        print(f"Mode final             : {self.mode}")
        display_elapsed = (
            (self.last_display_time - self.first_display_time)
            if self.first_display_time is not None
            and self.last_display_time is not None
            else 0.0
        )
        throughput_fps = (
            (self.frame_count - 1) / display_elapsed
            if self.frame_count > 1 and display_elapsed > 0
            else 0.0
        )
        print(f"Débit affiché moyen   : {throughput_fps:.3f} FPS")
        print(f"Latence capture→écran : {float(np.mean(self.latencies)):.3f} ms/frame")
        for name, samples in self.stage_samples.items():
            if samples:
                print(f"{name:22s}: {float(np.mean(samples)):.3f} ms/frame")
        if self.device.type == "cuda":
            print(f"VRAM max allouée       : {torch.cuda.max_memory_allocated(self.device) / (1024**2):.1f} MiB")
            print(f"VRAM max réservée      : {torch.cuda.max_memory_reserved(self.device) / (1024**2):.1f} MiB")

    def mainloop(self) -> None:
        self.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture un rectangle du bureau et affiche source, Catmull-Rom ou M3GSS V2"
    )
    parser.add_argument(
        "--bbox",
        type=parse_bbox,
        default=(0, 0, LR_WIDTH, LR_HEIGHT),
        help="Rectangle capturé left,top,right,bottom (défaut: 0,0,960,540)",
    )
    parser.add_argument(
        "--preview-size",
        type=parse_dimensions,
        default=(LR_WIDTH, LR_HEIGHT),
        help="Taille d'affichage WIDTHxHEIGHT (défaut: 960x540)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Arrêt automatique après N frames (test rapide)",
    )
    parser.add_argument(
        "--start-mode",
        choices=("source", "baseline", "v2"),
        default="v2",
        help="Mode initial (bascule ensuite avec M/B)",
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames doit etre positif")

    viewer = RealtimeCaptureViewer(
        args.bbox,
        args.preview_size,
        args.max_frames,
        args.start_mode,
    )
    viewer.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())