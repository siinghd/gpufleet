"""Configuration: ~/.config/gpufleet/config.toml (all keys optional), merged over these defaults.

Nothing secret lives here; provider keys stay where each provider's own tooling keeps them (Vast: VAST_API_KEY or
~/.config/vastai/vast_api_key, Colab: the google-colab-cli token in ~/.config/colab-cli).
"""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("GPUFLEET_CONFIG_DIR", "~/.config/gpufleet")).expanduser()
STATE_DIR = Path(os.environ.get("GPUFLEET_STATE_DIR", "~/.local/state/gpufleet")).expanduser()

DEFAULTS: dict = {
    "label_prefix": "gf-",
    "socket": str(STATE_DIR / "fleetd.sock"),
    "db": str(STATE_DIR / "fleet.db"),
    "daily_cap_usd": 10.0,  # paid providers, all projects together
    "max_boxes": 6,  # across providers
    "value_of_time_usd_per_h": 0.6,  # how much an hour of waiting is worth when ranking offers
    "scale_delay_s": 45,  # a queued task waits this long for a free lane before we rent another box
    "idle_s": 300,  # an idle box is kept this long for reuse
    "reuse_max_age_s": 4800,  # no new work after this age (boxes get flaky, contracts end)
    "hard_ttl_s": 7200,  # destroyed at this age no matter what (the box also self-destroys on Vast)
    "boot_timeout_s": 480,  # a host still pulling the image after 8 min is a slow host: destroy, rent another
    "max_parallel_boots": 3,
    "boot_retries": 3,
    "max_lanes_per_gpu": 4,
    "lane_autoscale": True,
    "reconcile_s": 60,
    "providers": {
        "vast": {"enabled": True, "max_boxes": 5, "credit_floor_usd": 0.60, "max_bandwidth_usd_per_tb": 10,
                 "min_reliability": 0.98, "ssh_key": "~/.ssh/id_ed25519_vast"},
        "colab": {"enabled": False, "max_boxes": 1, "cu_floor": 15, "accelerators": ["G4"], "shadow_factor": 0.35,
                  "ssh_key": "~/.ssh/id_ed25519_gpufleet"},
    },
    "profiles": {},  # extra/overridden profiles, see profiles/__init__.py
    # Object store for "via": "r2" transfers (blobs.py); active once the env file exists.
    "blobs": {"env_file": str(CONFIG_DIR / "r2.env"), "prefix": "gpufleet/"},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else CONFIG_DIR / "config.toml"
    cfg = DEFAULTS
    if p.exists():
        with open(p, "rb") as f:
            cfg = _merge(DEFAULTS, tomllib.load(f))
    cfg = copy.deepcopy(cfg)
    cfg["state_dir"] = str(Path(cfg.get("state_dir", STATE_DIR)).expanduser())
    return cfg
