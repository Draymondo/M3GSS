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

## 8. V2 — low-resolution runtime architecture

The measured V1 bottleneck led to a concrete V2 design.

### Selected architecture

```
960×540 LR
      ↓
Conv 3→24
      ↓
4 residual blocks, 24 channels
      ↓
trunk Conv 24→24
      ↓
Conv 24→12
      ↓
PixelShuffle ×2
      ↓
3-channel HR residual
      +
Catmull-Rom HR baseline
      ↓
1920×1080
```

Key constraints:

- majority of learned computation stays at **960×540**;
- **no BatchNorm**;
- **no heavy attention**;
- actual learned **×2 reconstruction**;
- explicit Catmull-Rom HR baseline;
- final reconstruction layer is **zero initialized**, so an untrained V2 initially reproduces the baseline exactly;
- V1 remains the quality reference;
- V2 reuses the validated Charbonnier + Edge loss;
- target hardware remains GTX 970 4 GB;
- FP16 is not assumed to help.

### V2 prototype validation

File:

`ml/m3gss_v2/model.py`

Prototype characteristics:

- **50,148 parameters**
- input: LR `[N,3,H,W]`
- baseline: HR `[N,3,2H,2W]`
- output: HR `[N,3,2H,2W]`
- PixelShuffle ×2
- zero-initialized reconstruction head
- forward/backward self-test passed
- output initially equals the Catmull-Rom baseline exactly.

### V2 GTX 970 runtime benchmark

ATMOS / GTX 970, FP32:

- LR: 960×540
- HR: 1920×1080
- 20 warm-up iterations
- 100 measurement iterations
- mean latency: **35.625 ms/frame**
- theoretical FPS: **28.07**
- min CUDA: 34.801 ms
- max CUDA: 36.176 ms
- peak allocated VRAM: **246.5 MiB**
- V1 reference: 311.2 ms/frame / 3.21 FPS
- V2 speedup: **8.74×**
- latency reduction: **88.55%**

Important qualification:

> 28.07 FPS is a **model-only benchmark**, not final game FPS. V2 was untrained during this measurement, and its zero-initialized reconstruction head therefore behaved essentially as the Catmull-Rom baseline.

The result nevertheless validates the V2 architecture as a strong runtime direction. The architecture should not be reduced further before training/quality evaluation.

---

## 9. V2 training integration and validation

V2 was integrated into `ml/train.py` without changing the V1 default path.

Supported selector:

`--model {v1,v2}`

Important V2 training properties:

- HR patches: **96×96**
- LR patches: **48×48**
- ×2 scale
- existing `area_downscale`
- existing `catmull_rom_upscale`
- V2 receives LR + Catmull-Rom HR baseline
- target remains HR
- loss: **Charbonnier + Edge**
- `CHARBONNIER_EPS = 0.001`
- `EDGE_WEIGHT = 0.1`
- `LOSS_VERSION = "v1_charbonnier_edge"`
- V2 checkpoints:
  - `m3gss_v2_latest.pt`
  - `m3gss_v2_best.pt`
- V2 checkpoint metadata records architecture, scale, patch sizes, loss version, parameter count, dataset, step and optimizer state.
- checkpoint loader rejects wrong architecture.
- V2 save paths are separated from V0/V1 paths.
- V1 checkpoint save/round-trip behavior was preserved.

Validation before training:

- V2 training tests: **5 passed**
- targeted suite: **38 passed**
- AST/syntax validation: OK
- `train.py --help`: V1/V2 selector and smoke-test exposed
- no V0/V1 checkpoint modification.

### V2 smoke test

Command:

```powershell
D:\M3GSS_OFFLINE\m3gss_gpu_env\Scripts\python.exe .\ml\train.py --model v2 --smoke-test --steps 3 --batch-size 1
```

Result:

- **3/3 steps passed**
- V2 forward/backward: OK
- optimizer step: OK
- Charbonnier + Edge: OK
- total time: **0.57 s**
- speed: **5.28 step/s**
- checkpoint created: `m3gss_v2_smoke.pt`

### Real-data 50-step validation

Command:

```powershell
D:\M3GSS_OFFLINE\m3gss_gpu_env\Scripts\python.exe .\ml\train.py --model v2 --steps 50 --batch-size 1
```

The terminal output was only captured through step 25, but checkpoint inspection confirmed the run **did reach step 50**.

Checkpoint metadata from `m3gss_v2_latest.pt`:

- step: **50**
- loss: **0.020713699739426373**
- architecture: **M3GSS_v2**
- loss version: **v1_charbonnier_edge**
- parameters: **50,148**
- HR patch: **96**
- LR patch: **48**
- scale: **2**
- dataset: **DIV2K_train_HR + Flickr2K_HR**

This confirmed that V2 training works on the **real 3,450-image dataset** and that checkpointing reaches the requested step.

---

## 10. Official V2 5,000-step training — IN PROGRESS

The first full V2 training run has now been launched manually on ATMOS, rather than by Copilot.

Command:

```powershell
D:\M3GSS_OFFLINE\m3gss_gpu_env\Scripts\python.exe C:\M3GSS\ml\train.py --model v2 --steps 5000 --batch-size 8
```

At launch, the run confirmed:

- DIV2K: **800 images**
- Flickr2K: **2,650 images**
- total: **3,450 images**
- GPU: **NVIDIA GeForce GTX 970**
- PyTorch: **2.14.1+cu126**
- CUDA: **12.6**
- HR patch: **96×96**
- LR patch: **48×48**
- steps: **5,000**
- batch: **8**
- learning rate: **1e-4**
- loss: **v1_charbonnier_edge**
- Charbonnier eps: **0.001**
- Edge weight: **0.1**
- parameters: **50,148**
- effective start: **1**
- requested end: **5,000**

Observed early training:

| Step | Charbonnier | Edge | Total | Reported speed |
|---:|---:|---:|---:|---:|
| 1 | 0.016584 | 0.035944 | 0.020178 | 0.77 step/s |
| 25 | 0.011006 | 0.024497 | 0.013456 | 0.84 step/s |

**Status: RUNNING / IN PROGRESS.**

The machine can remain offline during this training because the dataset, code and Python environment are local on ATMOS.

Do not treat the final V2 quality or runtime as established until the 5,000-step run completes and the trained checkpoint is validated.

---

## 11. Current project priority

The project is now explicitly in this order:

1. **Complete the official V2 5,000-step training run.**
2. Validate the trained V2 checkpoint.
3. Measure V2 quality against bicubic and V1.
4. Re-run the GTX 970 runtime benchmark with the trained V2.
5. Build a continuous frame-processing loop.
6. Introduce a real game/frame source.
7. Test actual playable game performance on ATMOS.
8. Only then iterate further on architecture or temporal reconstruction.

The project must **not** drift into an endless training cycle.

---

## 12. Dataset / training history

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
        ↓
V2 low-resolution architecture
        ↓
V2 runtime benchmark
        ↓
V2 training integration
        ↓
V2 smoke + real-data validation
        ↓
official V2 5,000-step training — IN PROGRESS
```

---

## 13. Checkpoints

Important local ATMOS checkpoints:

- `ml/checkpoints/m3gss_v0_big_best.pt`
- `ml/checkpoints/m3gss_v0_big_latest.pt`
- `ml/checkpoints/m3gss_v0_div2k_500.pt`
- `ml/checkpoints/m3gss_v1_big_best.pt`
- `ml/checkpoints/m3gss_v1_big_latest.pt`
- `ml/checkpoints/m3gss_v2_smoke.pt`
- `ml/checkpoints/m3gss_v2_latest.pt`
- `ml/checkpoints/m3gss_v2_best.pt`
- `ml/checkpoints/smoke_test.pt`

Current V2 real-data validation checkpoint:

`ml/checkpoints/m3gss_v2_latest.pt`

It was verified to contain **step 50**.

Checkpoints are not committed to Git.

---

## 14. Current C++ prototype

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

## 15. Runtime scripts

The following experimental runtime scripts now exist in `ml/`:

- `inference_benchmark.py` — baseline GTX 970 inference benchmark
- `benchmark_fp16.py` — FP16 vs FP32 experiment
- `benchmark_resolutions.py` — multi-resolution latency benchmark
- `benchmark_runtime_breakdown.py` — bicubic/network runtime decomposition
- `benchmark_v2_runtime.py` — GTX 970 V2 architecture benchmark

These are measurement tools, not yet the final game runtime.

---

## 16. Longer-term roadmap

### V0 — Spatial baseline
**Completed.**

### V1 — Loss / robustness
**Completed.**

### V2 — Low-resolution runtime architecture
**Architecture validated; 5,000-step training currently in progress.**

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

## 17. Design principles / constraints

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

## 18. Handoff for another AI agent

If another AI assistant (Claude/Cline/Codex/etc.) takes over this repository, it should read this README **before making project decisions**.

### Established facts

- V0 spatial model exists and is validated.
- V1 spatial model exists and is validated.
- V1 gives **32.7649 dB** vs **31.2474 dB bicubic** on the official 100-image validation set.
- V1 runtime at 1920×1080 on GTX 970 is **311.703 ms/frame (~3.21 FPS)**.
- The bicubic portion of the target pipeline is only **2.813 ms**.
- The same V1 network at 960×540 is **76.923 ms/frame (~13 FPS)**.
- FP16 was slower than FP32 in the tested GTX 970 path.
- V2 has **50,148 parameters**.
- V2 model-only runtime benchmark is **35.625 ms/frame (~28.07 FPS)** on GTX 970 before training.
- V2 smoke test and real-data 50-step validation both passed.
- Official V2 5,000-step training is currently **in progress**.

### Immediate next decision

**Do not start another training run while the current 5,000-step V2 run is in progress.**

After it completes:

1. validate the V2 checkpoint;
2. measure quality against bicubic and V1;
3. benchmark trained V2 on GTX 970;
4. then move toward the continuous frame-processing loop and real game test.

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
training integration
    ↓
V2 5,000-step training
    ↓
quality validation
    ↓
trained runtime benchmark
    ↓
continuous frame pipeline
    ↓
real game
    ↓
playable M3GSS test
```

This README is the project memory anchor. Future decisions should preserve this sequence unless new measurements provide a strong reason to change it.
