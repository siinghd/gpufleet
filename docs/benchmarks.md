# Benchmarks

`bench/cycles_bench.py` builds the same mid-weight scene on every box (displaced ground, 60 subdivided objects mixing
diffuse/glossy/glass, a third of them animated, three area lights, sky), 1080×1920, 48 frames, Cycles 64 spp adaptive
(threshold 0.02), OIDN on the GPU, light tree off. `bench/probe.sh` runs the ladder below; `bench/vast_probe.py`
rents one box per GPU model on Vast, runs it and destroys the box.

Columns: steady-state median seconds per frame (frames 2..48), first frame (kernel load + BVH), wall time.

## Google Colab G4: RTX PRO 6000 Blackwell Server Edition (96 GB), 2026-10-19

48 vCPU AMD EPYC 9B45, 176 GB RAM, 236 GB disk, Ubuntu 24.04.4, driver 580.82.07, CUDA 13.0. Burn rate measured
**8.9 CU/h** (not the 6.7 often quoted): a Google AI Pro plan's 200 CU is ~22 G4-hours a month.

| run | s/frame | first frame | wall (48 fr) |
|---|---|---|---|
| OptiX, cold kernel cache | 0.633 | 2.82 | 32.5 |
| OptiX, warm kernel cache | **0.615** | **1.66** | 30.5 |
| OptiX, `use_persistent_data` off | 1.621 | 1.74 | 77.9 |
| CUDA | 0.893 | 1.84 | 43.8 |
| 2 processes on the GPU, each | 1.12 | 2.4 | 55.3 (96 frames: 0.587 s/frame, +9 %) |
| EEVEE (NVIDIA EGL registered), cold shader cache | 0.183 | 9.13 | 13.4 (24 fr) |
| EEVEE, warm shader cache | 0.184 | **1.29** | 5.6 (24 fr) |

- GPU utilisation: 70 % average with one process (scene sync and PNG writes between frames), 95 % with two.
- Colab's NVIDIA userspace libraries live in `/usr/lib64-nvidia` with no glvnd EGL vendor file, so EEVEE silently
  fell back to Mesa's software renderer (EGL_BAD_MATCH, minutes per frame). The `blender` profile registers the
  NVIDIA vendor; then EEVEE runs at 0.18 s/frame.
- NVENC works (h264, 48 frames in 1.07 s). Downloads from download.blender.org: 203 MB/s; Blender extracts in 13 s and
  starts in 1.4 s.
- Network through the Colab SSH tunnel (one multiplexed connection): rsync 5.7 MB/s up, 10.9 MB/s down from Helsinki.
- Allocation: G4 returned 503 (no capacity) once, then 412 (too many assignments, a leftover of the failed attempt)
  twice, then succeeded; ~3 minutes in total. SSH answered 2.4 s after assignment.

## Vast.ai

Prices paid and boot times come from the same runs; see `bench/results/` (gitignored) for raw output.
