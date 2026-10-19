"""GPU speed model: Blender Open Data OptiX medians as the prior, our own measurements once we have them.

Scores are samples-per-minute style numbers (higher is faster). The prior only has to rank GPUs sensibly; the first
few finished tasks of a workload replace it with real seconds-per-unit for that GPU model.
"""

from __future__ import annotations

import re

REFERENCE_SCORE = 16000.0  # ~RTX 5090 / RTX PRO 6000; JobSpec.unit_cost_s is expressed against this

# Blender Open Data, OptiX, Blender 4.5 medians (large n), pulled 2026-10-19. A100/H100/T4 are CUDA-class here:
# they have no RT cores and are marked rt=False so OptiX jobs skip them.
GPUS: dict[str, dict] = {
    "RTX PRO 6000 WS": {"score": 16737, "rt": True, "nvenc": True, "vram": 96},
    "RTX PRO 6000 S": {"score": 15519, "rt": True, "nvenc": True, "vram": 96},
    "RTX PRO 6000 Max-Q": {"score": 14746, "rt": True, "nvenc": True, "vram": 96},
    "RTX 5090": {"score": 14966, "rt": True, "nvenc": True, "vram": 32},
    "RTX 5080": {"score": 9600, "rt": True, "nvenc": True, "vram": 16},
    "RTX 4090": {"score": 11061, "rt": True, "nvenc": True, "vram": 24},
    "RTX 6000Ada": {"score": 10655, "rt": True, "nvenc": True, "vram": 48},
    "RTX 4080S": {"score": 8300, "rt": True, "nvenc": True, "vram": 16},
    "RTX 4080": {"score": 8000, "rt": True, "nvenc": True, "vram": 16},
    "L40S": {"score": 7848, "rt": True, "nvenc": True, "vram": 48},
    "L40": {"score": 7400, "rt": True, "nvenc": True, "vram": 48},
    "RTX 4070 Ti": {"score": 6500, "rt": True, "nvenc": True, "vram": 12},
    "RTX 3090 Ti": {"score": 5900, "rt": True, "nvenc": True, "vram": 24},
    "RTX 3090": {"score": 5412, "rt": True, "nvenc": True, "vram": 24},
    "RTX A6000": {"score": 5300, "rt": True, "nvenc": True, "vram": 48},
    "RTX 3080": {"score": 4600, "rt": True, "nvenc": True, "vram": 10},
    "L4": {"score": 3447, "rt": True, "nvenc": True, "vram": 24},
    "H100 SXM": {"score": 6944, "rt": False, "nvenc": False, "vram": 80},
    "H100 PCIE": {"score": 6400, "rt": False, "nvenc": False, "vram": 80},
    "H200": {"score": 6174, "rt": False, "nvenc": False, "vram": 141},
    "A100 SXM4": {"score": 4194, "rt": False, "nvenc": False, "vram": 80},
    "A100 PCIE": {"score": 4143, "rt": False, "nvenc": False, "vram": 80},
    "T4": {"score": 1502, "rt": False, "nvenc": True, "vram": 16},
}

_ALIASES = [
    (r"rtx pro 6000.*(server|blackwell server)", "RTX PRO 6000 S"),
    (r"rtx pro 6000.*max-?q", "RTX PRO 6000 Max-Q"),
    (r"rtx pro 6000", "RTX PRO 6000 WS"),
    (r"rtx 6000 ?ada", "RTX 6000Ada"),
    (r"4080 ?s(uper)?\b", "RTX 4080S"),
    (r"h100.*(sxm|hbm3)", "H100 SXM"),
    (r"h100", "H100 PCIE"),
    (r"a100.*sxm", "A100 SXM4"),
    (r"a100", "A100 PCIE"),
    (r"tesla t4|\bt4\b", "T4"),
]


def canonical(name: str) -> str:
    """Map provider GPU names ("NVIDIA GeForce RTX 4090", "RTX_4090", "G4") onto the table's keys."""
    n = name.replace("_", " ").replace("NVIDIA", "").replace("GeForce", "").strip()
    low = n.lower()
    if low == "g4":
        return "RTX PRO 6000 S"
    for pat, key in _ALIASES:
        if re.search(pat, low):
            return key
    for key in sorted(GPUS, key=len, reverse=True):
        if key.lower() == low or re.search(rf"\b{re.escape(key.lower())}\b", low):
            return key
    return n


def info(name: str) -> dict:
    return GPUS.get(canonical(name), {"score": 3000, "rt": True, "nvenc": True, "vram": 0})


def prior_secs_per_unit(gpu: str, unit_cost_s: float) -> float:
    """Expected seconds per unit from the Open Data prior: unit_cost_s is measured at REFERENCE_SCORE."""
    return unit_cost_s * REFERENCE_SCORE / max(info(gpu)["score"], 1)


def secs_per_unit(store, gpu: str, workload: str, unit_cost_s: float) -> float:
    """Best estimate: learned value when it exists (blended with the prior while n is small)."""
    prior = prior_secs_per_unit(gpu, unit_cost_s)
    got = store.learned(canonical(gpu), workload) if store is not None else None
    if not got:
        # No sample for this GPU yet: scale another GPU's measurement for the same workload by the score ratio.
        if store is not None:
            rows = store.q("SELECT gpu, secs_per_unit, n FROM perf WHERE workload=? ORDER BY n DESC LIMIT 1",
                           [workload])
            if rows:
                r = rows[0]
                return r["secs_per_unit"] * info(r["gpu"])["score"] / max(info(gpu)["score"], 1)
        return prior
    learned, n = got
    w = min(n / 3, 1.0)
    return learned * w + prior * (1 - w)
