# gpufleet

Rent GPUs across providers and run batch jobs on them: fast, packed, and without leaking a cent of billing.

gpufleet is a small daemon (`fleetd`) plus a CLI and a socket API. You describe a job (inputs, a command, outputs, a
list of units or a numeric range), and it finds the cheapest box that gets the work done soonest, keeps warm boxes for
the next job, splits the work across as many boxes as pay off, and destroys everything it rented when it is done.

Providers today: **Vast.ai** and **Google Colab** (through the official
[google-colab-cli](https://github.com/googlecolab/google-colab-cli); a Google AI plan's monthly compute units become
prepaid GPU hours). A provider is about 200 lines (`src/gpufleet/providers/`).

It was built for headless Blender rendering (OptiX/Cycles, EEVEE) and works for any batch job: ffmpeg, upscalers,
Demucs, OCR, inference.

## What it does

| | |
|---|---|
| **Scheduling** | Named units run longest-first (LPT); numeric ranges are carved into chunks sized for the GPU that leases them (~90 s of work), so fast cards take bigger chunks and the tail stays short. Straggling tail tasks are duplicated on idle boxes that are at least as fast; first result wins. |
| **Offer ranking** | Expected cost = price × (boot + work) / boot success rate + value of waiting time. Speed starts from Blender Open Data scores and is replaced by measured seconds-per-unit per GPU model and workload. A100/H100/T4 are excluded for RT-core jobs. |
| **Packing** | Several tasks per GPU ("lanes"). A box whose lanes are all busy while the GPU is under 80 % busy and has VRAM to spare gets another lane. |
| **Fast starts** | Idle boxes are *parked* (stopped, disk kept, storage-only billing) and restarted in seconds for the next job; a warm pool (`fleet warm`) keeps boxes ready; hosts that already ran our image (cached layers) rank as fast-booting; dispatch is event-driven. |
| **Never leak billing** | Box rows are written before the provider is called; every box carries a label; a reconciler compares each provider's instance list with the database every minute and destroys our orphans, terminal boxes and boxes past their deadline (never anything without our label); Vast boxes also self-destruct at their deadline; per-day spend cap; provider credit / compute-unit floors. |
| **Failures** | Stale offers are skipped; boxes that never come up are destroyed and their machine recorded (a machine with 2 strikes in 14 days, or a host with 3 bad machines, is excluded from searches); heartbeats detect silent boxes; tasks over 3× their expected time are killed and moved; output pulls retry with backoff; a box that dies mid-task costs at most one checkpoint interval (partial results are pulled while the task runs and pushed back before the retry). |
| **Restarts** | Tasks run detached on the box with their exit codes on disk. Restarting `fleetd` reattaches to every box and collects what finished meanwhile. |
| **Compiler caches** | OptiX kernels and the NVIDIA GL shader cache are saved per (profile, GPU, driver) and restored on new boxes, so only the first box of a kind compiles. |

## Measured (Blender 5.2.2, 1080×1920, Cycles 64 spp adaptive + OIDN)

See [docs/benchmarks.md](docs/benchmarks.md). Highlights on an RTX PRO 6000 Blackwell (Colab G4):

- `use_persistent_data`: **0.62 s/frame vs 1.62 s** without (2.6×).
- OptiX vs CUDA: 0.62 vs 0.89 s/frame (1.45×).
- OptiX kernel cache warm: first frame 2.8 s → 1.7 s. EEVEE with the GL shader cache warm: first frame 9.1 s → 1.3 s.
- Two Blender processes on one GPU: +9 % throughput on this card (GPU at 95 %); more on boxes whose CPU is the bottleneck, which is why lanes follow telemetry.

## Install

```bash
git clone https://github.com/siinghd/gpufleet && cd gpufleet
uv venv -p 3.13 && uv pip install -e '.[colab]'
cp deploy/fleetd.service /etc/systemd/system/   # edit User/paths, then:
systemctl enable --now fleetd
```

Credentials stay where each provider's own tooling keeps them:

- Vast: `VAST_API_KEY` or `~/.config/vastai/vast_api_key`; an SSH key registered on the account (`ssh_key` in config).
- Colab: log in once with `colab` (google-colab-cli ≥ 0.7.4; grant every scope on the consent screen), and set
  `providers.colab.enabled = true`. Colab needs an ed25519 key (`ssh_key`).

Settings: `~/.config/gpufleet/config.toml`, merged over the defaults in `src/gpufleet/config.py`:

```toml
daily_cap_usd = 10.0

[providers.vast]
enabled = true
credit_floor_usd = 0.60

[providers.colab]
enabled = true
cu_floor = 15        # stop launching here: Colab only allows SSH while the balance is positive
max_boxes = 1
```

## Jobs

```json
{
  "project": "my-film", "name": "shot-render",
  "units": ["intro", "chase", "end"],
  "unit_costs": {"intro": 300, "chase": 900, "end": 200},
  "need": {"profile": "blender", "rt_cores": true, "min_score": 5000, "max_price_h": 0.6},
  "inputs": [{"src": "~/films/my-film/", "dst": "film/", "exclude": ["renders"]}],
  "command": "cd film && blender -b --factory-startup -noaudio scenes/$GF_UNIT.blend -o //out/$GF_UNIT/f_#### -a",
  "outputs": [{"src": "film/scenes/out/{unit}/", "dst": "~/films/my-film/renders/{unit}/"}],
  "checkpoint": [{"src": "film/scenes/out/{unit}/", "dst": "~/films/my-film/.ckpt/{unit}/"}]
}
```

Or a frame range, chunked automatically: `"range": [1, 2400]` with `$GF_LO`/`$GF_HI` in the command.

```bash
fleet run job.json          # submit and stream progress; exit 0 done, 1 failed
fleet status                # boxes, lanes, utilisation, spend
fleet warm blender 1 30     # one blender box ready for 30 minutes
fleet offers --profile blender --rt
```

From code: `gpufleet.client.Client().run(spec)`, or JSON lines over the unix socket from any language.

Commands see `GF_UNIT` (named units) or `GF_LO`/`GF_HI` (ranges), `GF_TASK`, `GF_JOB`, `GF_ATTEMPT`, `GF_ROOT`.
Exit 3 means "this box is broken" (e.g. no usable GPU): the task moves to another machine and the machine is
recorded.

## Object storage (optional)

Mark a transfer `"via": "r2"` to move it through Cloudflare R2 (or any S3 API) instead of rsync over SSH: the tree is
packed once into a zstd tar named by its content hash, uploaded once, and every box downloads it in parallel straight
from the store; outputs go up from the box and come back the same way. Boxes only ever get presigned URLs for single
objects, never credentials. Configure with `~/.config/gpufleet/r2.env` (`R2_ENDPOINT`, `R2_BUCKET`,
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`; use a token scoped to one bucket). Without it, `via` falls back to rsync.

## Profiles

A profile is the software a box needs: an image for Docker providers and an idempotent bootstrap that installs or
verifies the same on providers without Docker (Colab). Built in: `base`, and `blender`: the
[docker/blender](docker/blender/Dockerfile) image (`ghcr.io/siinghd/gpufleet-blender:5.2.2`, Blender 5.2.2 + NVENC ffmpeg,
608 MB zstd), OptiX/GL cache settings, Colab's missing NVIDIA EGL vendor registered, and a 64x64 OptiX + EEVEE
self-test that rejects broken GPU hosts before they take work. Add your own under
`[profiles.<name>]` in the config (`vast_image`, `bootstrap_extra`, `env`, `caches`).

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests run the real scheduler and the real box agent against a fake provider whose boxes are local directories:
units and ranges, retries, machine faults, stale offers, a box dying mid-render and the retry resuming from its
checkpoint, a daemon crash and reattach, hung tasks, parking and restarting boxes, the warm pool, the daily cap,
the reconciler.

## License

Apache-2.0
