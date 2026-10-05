# M3GSS

> **M3GSS (personal project) — local/offline AI image reconstruction and upscaling, inspired by the DLSS concept.**

M3GSS is a personal, low-cost / zero-budget research project whose real objective is **not merely to maximize PSNR**: the final target is a useful **real-time image reconstruction/upscaling pipeline for games on a GTX 970 4 GB**, with practical GPU cost and no dependence on a cloud service at runtime.

The project has completed its first offline model-validation phase and is now moving into **runtime architecture and game-integration engineering**.

---

## 1. Project goal

The intended end state is conceptually:

```
Game renders lower resolution
        ↓
   M3GSS reconstruction
        ↓
Higher-resolution reconstructed frame
        ↓
      Display
```

First practical target:

- input: approximately 960×540
- reconstruction: ×2
- output: approximately 1920×1080
- target hardware: **NVIDIA GTX 970 4 GB**
- priority: **quality × latency × FPS**, not PSNR alone.

The first game-facing implementation should be treated as a **runtime prototype** before attempting deep integration into a game's graphics pipeline.

---

## 2. Hardware / development environment

### ATMOS — main gaming / benchmark PC

- Windows 11 Pro 64-bit
- Ryzen 5 5500
- 16 GB RAM
- Gigabyte B550M AORUS ELITE AX
- GTX 970 4 GB
- NVIDIA driver used during the project: 32.0.15.8266 (2026-06-09)
- Local PyTorch environment: PyTorch 2.14.1+cu126, CUDA available
- Local Python environment:
  - `D:\M3GSS_OFFLINE\m3gss_gpu_env\Scripts\python.exe`

ATMOS is the important machine for the **final GTX 970 runtime/inference and game tests**.

### T470s

Used mainly for:

- code preparation
- Git/GitHub
- VS Code / Cline
- file transfer to/from ATMOS

Some models/checkpoints and local resources exist only on ATMOS and are not necessarily present on the T470s.

### Kaggle

Additional training/experimentation environment:

- Tesla T4
- ~14.5 GB VRAM
- PyTorch 2.11.0+cu128
- CUDA 12.8
- Internet enabled

Kaggle is useful for experiments, but **it is not the final target**. The final target remains the GTX 970 on ATMOS.

---

## 3. Dataset

Training data:

- **DIV2K training:** 800 HR images
- **Flickr2K:** 2,650 HR PNG images
- **Total training images:** 3,450

Validation:

- **DIV2K validation:** 100 images

### Local dataset

- `D:\M3GSS_OFFLINE\datasets\DIV2K\train_HR`
- `D:\M3GSS_OFFLINE\datasets\DIV2K\valid_HR`
- `D:\M3GSS_OFFLINE\datasets\Flickr2K\HR\Flickr2K`
- Flickr2K archive: `D:\M3GSS_OFFLINE\datasets\Flickr2K\Flickr2K.zip`

### Kaggle dataset

Private Kaggle dataset:

- `M3GSS-Datasets`
- `raymonddjadoo/m3gss-datasets`

Kaggle Flickr2K remains compressed because extracting it exceeded the practical working-disk budget.

---

## 4. Current V1 model architecture

Model:

`M3GSS_v0_32x8`

Parameters: **158,979**

Architecture:

1. RGB input
2. `Conv2d(3, 32, 3×3, padding=1)`
3. LeakyReLU(0.1)
4. 8 residual blocks:
   - Conv 3×3 → LeakyReLU → Conv 3×3
   - local residual skip
5. trunk Conv 32→32
6. global skip from the head
7. zero-initialized final Conv 32→3
8. predicted residual added to the bicubic input

Constraints:

- **no BatchNorm**
- **no attention**
- **no PixelShuffle**
- lightweight CNN.

### Important runtime characteristic

The current model **does not perform the ×2 upsampling itself**.

The current inference path is:

```
960×540 LR
    ↓
Bicubic ×2
    ↓
1920×1080
    ↓
M3GSS V1 CNN
    ↓
1920×1080 reconstructed output
```

This detail is now considered a major architectural limitation for the final real-time target because all 8 residual blocks and the trunk operate at full 1920×1080 resolution.

---

## 5. V0 — validated baseline

V0:

- architecture: `M3GSS_v0_32x8`
- 3,450 training images
- 5,000 steps
- batch 8
- L1 loss
- GTX 970 4 GB

Official GTX 970 benchmark:

- loss: **0.016380**
- time: **5708.28 s** (~1 h 35 min)
- speed: **0.88 step/s**
- peak VRAM: **258.1 MiB**

Official 100-image DIV2K validation:

| Model | PSNR |
|---|---:|
| Bicubic | 31.2474 dB |
| M3GSS V0 | **32.7892 dB** |

Gain: **+1.5419 dB vs bicubic**

Visual findings:

- strong detail reconstruction
- ringing / oversharpening on some repetitive textures
- very fine details can look sharp but sometimes too aggressive.

---

## 6. V1 — Charbonnier + Edge/Gradient loss

V1 keeps the **exact same neural architecture**. The training objective changed.

Constants:

- `CHARBONNIER_EPS = 1e-3`
- `EDGE_WEIGHT = 0.1`
- `LOSS_VERSION = "v1_charbonnier_edge"`

Loss:

```
Charbonnier = mean(sqrt(diff² + eps²))
Edge = L1 difference between x/y image gradients
Total = Charbonnier + 0.1 × Edge
```

### Official V1 GTX 970 result

- 5,000 steps
- batch 8
- LR 1e-4
- HR patch 96×96
- ×2
- GTX 970 4 GB

Training:

- Charbonnier: **0.016299**
- Edge: **0.039409**
- Total: **0.020240**
- time: **3451.96 s** (~57 min 32 s)
- speed: **0.87 step/s**
- peak VRAM: **258.1 MiB**

Official validation:

| Model | PSNR |
|---|---:|
| Bicubic | 31.2474 dB |
| M3GSS V0 | 32.7892 dB |
| M3GSS V1 | **32.7649 dB** |

V1 gain vs bicubic: **+1.5176 dB**

V1 vs V0: **−0.0243 dB**

Visual conclusion:

- V1 reduces ringing and oversharpening
- contours are more stable
- repetitive/fine structures look more natural
- slight loss of maximum micro-detail / sharpness in some areas
- V1 remains the current spatial-quality reference model.

---

## 7. Runtime benchmark — GTX 970

This section records the **critical runtime findings** that determine the next architecture.

All measurements below were performed on ATMOS using the GTX 970 and the trained `m3gss_v1_big_best.pt` checkpoint, FP32, CUDA, with synchronized timing.

### Multi-resolution benchmark

| Network resolution | Latency | Theoretical FPS |
|---|---:|---:|
| 960×540 | **77.196 ms** | **12.95 FPS** |
| 1280×720 | **134.091 ms** | **7.46 FPS** |
| 1600×900 | **213.747 ms** | **4.68 FPS** |
| 1920×1080 | **311.321 ms** | **3.21 FPS** |

### Runtime decomposition at the actual target path

For 960×540 → 1920×1080:

| Operation | Measured latency |
|---|---:|
| Bicubic 960×540 → 1920×1080 | **2.813 ms** |
| M3GSS network at 1920×1080 | **311.703 ms** |
| Current pipeline total | **314.516 ms** |
| M3GSS network at 960×540 | **76.923 ms** |

Therefore:

- bicubic is **not** the bottleneck;
- almost all runtime is spent inside the neural network;
- moving the same network from 1920×1080 to 960×540 reduces network latency by about **4.05×**;
- current V1 at 1920×1080 is only about **3.2 FPS** on the target GTX 970;
- current V1 is therefore **not suitable for real-time 1080p game reconstruction** in its present form.

### FP16 result

A separate GTX 970 test showed:

- FP32: **310.707 ms/frame**, 3.22 FPS
- FP16: **356.815 ms/frame**, 2.80 FPS
- FP16 speedup: **0.87×**

Conclusion: **do not assume FP16 is an optimization on this GTX 970**. The tested FP16 path was slower and is not currently the optimization direction.

---

## 8. Architectural conclusion — V2 direction

The runtime measurements establish that the principal issue is **where the convolutions are performed**, not the bicubic operation.

Current architecture:

```
960×540
   ↓
Bicubic ×2                    ~2.8 ms
   ↓
1920×1080
   ↓
8 residual blocks + trunk     ~311.7 ms
   ↓
1920×1080
```

This is the wrong computational layout for the final GTX 970 target.

### Intended V2 direction

The next architecture should perform the **majority of learned computation in low-resolution feature space**, then reconstruct/upscale to 1920×1080 near the end:

```
960×540 game frame
       ↓
low-resolution feature extraction
       ↓
most residual processing at 960×540
       ↓
efficient ×2 reconstruction / upsampling head
       ↓
1920×1080 output
```

This is now the **preferred V2 design direction**.

Important:

- do not simply add more training to V1;
- do not start V1.5 automatically before runtime architecture work;
- preserve V1 as the **quality and correctness reference**;
- V2 must be benchmarked first, then trained if its runtime profile is promising;
- the objective is to move from an offline reconstruction model toward a **real-time game-capable component on GTX 970**.

The exact V2 layer count, channel count and upsampling mechanism should be chosen after designing a lightweight prototype and benchmarking it, rather than guessed in advance.

---

## 9. V1.5 — currently deprioritized

A controlled `EDGE_WEIGHT = 0.05` experiment was originally planned.

It is **not currently the next priority**.

Reason: runtime benchmarking demonstrated a much more important limitation in the existing architecture. The project should first establish a viable low-resolution runtime architecture.

V1.5 can still be performed later if the result is useful for quality comparison.

---

## 10. Current project priority

The project is now explicitly in this order:

1. **Design M3GSS V2 for low-resolution computation.**
2. Build a minimal V2 prototype.
3. Benchmark V2 latency/FPS on the GTX 970.
4. Compare V2 output quality against bicubic and V1.
5. Only then train V2 seriously if the runtime/quality tradeoff is promising.
6. Build a continuous frame-processing loop.
7. Introduce a real game/frame source.
8. Test actual playable game performance on ATMOS.

The project must **not** drift into an endless training cycle.

---

## 11. Dataset / training history

The official completed spatial benchmarks use:

- DIV2K train: 800 images
- Flickr2K HR: 2,650 images
- total: **3,450 training images**
- DIV2K validation: 100 images.

Historical chronology:

```
DIV2K-only preparation
        ↓
Flickr2K added
        ↓
combined 3,450-image training
        ↓
official V0
        ↓
official V1 loss experiment
        ↓
GTX 970 runtime investigation
```

There was also an incomplete Kaggle 5,000-step experiment after a session reset. It is not an official completed result.

---

## 12. Checkpoints

Important local ATMOS checkpoints:

- `ml/checkpoints/m3gss_v0_big_best.pt`
- `ml/checkpoints/m3gss_v0_big_latest.pt`
- `ml/checkpoints/m3gss_v0_div2k_500.pt`
- `ml/checkpoints/m3gss_v1_big_best.pt`
- `ml/checkpoints/m3gss_v1_big_latest.pt`
- `ml/checkpoints/smoke_test.pt`

The official V1 best checkpoint is:

`ml/checkpoints/m3gss_v1_big_best.pt`

It contains the trained 158,979-parameter architecture used for the runtime benchmarks.

Checkpoints are not committed to Git.

---

## 13. Current C++ prototype

The repository also contains an earlier C++17/CMake prototype.

Environment:

- C++17
- CMake
- MSVC 2022
- Windows x64

It provides:

- PNG loading/writing
- baseline non-AI upscale
- MSE / PSNR / SSIM metrics
- deterministic tests
- dataset pair generation
- visual comparison tooling.

**The current C++ application is not yet the neural M3GSS runtime.**

Build:

```powershell
cmake --build build --config Release
cmake --build build --config Debug
ctest --test-dir build -C Release --output-on-failure
```

---

## 14. Runtime scripts

The following experimental runtime scripts now exist in `ml/`:

- `inference_benchmark.py` — baseline GTX 970 inference benchmark
- `benchmark_fp16.py` — FP16 vs FP32 experiment
- `benchmark_resolutions.py` — multi-resolution latency benchmark
- `benchmark_runtime_breakdown.py` — bicubic/network runtime decomposition

These are measurement tools, not yet the final game runtime.

---

## 15. Longer-term roadmap

### V0 — Spatial baseline
**Completed.**

### V1 — Loss / robustness
**Completed.**

### V2 — Low-resolution runtime architecture
**Next priority.**

Primary objective:

> Move most neural computation from 1920×1080 to 960×540 while retaining useful reconstruction quality and providing an efficient ×2 reconstruction head.

### V3 — Real-time game integration
Future engineering phase:

- continuous frame processing
- CUDA / graphics interop as appropriate
- DirectX pipeline integration
- frame timing
- FPS impact
- practical game testing.

Temporal reconstruction using previous-frame history, motion vectors and potentially depth remains a future possibility, but should not be introduced before the spatial runtime architecture works.

---

## 16. Design principles / constraints

M3GSS should remain:

- personal-use focused
- local/offline at runtime
- minimal or zero budget
- lightweight enough for GTX 970 4 GB
- experimentally measurable
- reproducible where practical
- incremental rather than architecture-heavy
- judged by real visual quality and runtime performance.

Do not optimize only for benchmark numbers.

Do not add heavy architecture components without evidence.

Do not treat Kaggle/T4 performance as the final success criterion.

Do not lose the main objective:

> **The end goal is to run M3GSS in a real game and actually play with the reconstruction active on the GTX 970.**

---

## 17. Handoff for another AI agent

If another AI assistant (Claude/Cline/Codex/etc.) takes over this repository, it should read this README **before making project decisions**.

### Established facts

- V0 spatial model exists and is validated.
- V1 spatial model exists and is validated.
- V1 gives **32.7649 dB** vs **31.2474 dB bicubic** on the official 100-image validation set.
- V1 runtime at 1920×1080 on GTX 970 is **311.703 ms/frame (~3.21 FPS)**.
- The bicubic portion of the target pipeline is only **2.813 ms**.
- The same V1 network at 960×540 is **76.923 ms/frame (~13 FPS)**.
- FP16 was slower than FP32 in the tested GTX 970 path.
- Therefore the major bottleneck is the **full-resolution neural computation**.

### Immediate next decision

Do **not** automatically:

- launch another long training run;
- start V1.5;
- optimize bicubic;
- assume FP16 will solve the problem.

Instead:

> **Design and benchmark a V2 architecture that performs most learned computation at 960×540 and performs ×2 reconstruction near the output stage.**

The project path is now:

```
validated V1
    ↓
runtime measurements
    ↓
low-resolution V2 prototype
    ↓
GTX 970 benchmark
    ↓
quality comparison
    ↓
training
    ↓
continuous frame pipeline
    ↓
real game
    ↓
playable M3GSS test
```

This README is the project memory anchor. Future decisions should preserve this sequence unless new measurements provide a strong reason to change it.
