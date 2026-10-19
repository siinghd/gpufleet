"""Plain data types shared by the daemon, providers and clients. Frozen where possible: state changes go through the store."""

from __future__ import annotations

import dataclasses as dc
import hashlib
import json
from typing import Any


@dc.dataclass(frozen=True)
class Need:
    """What a job needs from a box. Every field is a lower bound or a filter; providers translate it to their query."""

    gpu: bool = True
    min_score: float = 0  # Blender Open Data OptiX score (5090 ~16000); 0 = any GPU
    min_vram_gb: float = 0
    rt_cores: bool = False  # OptiX rendering: excludes A100/H100/T4-class datacenter cards without RT cores
    nvenc: bool = False
    max_price_h: float = 1.0  # USD per box-hour
    min_cpu: int = 0
    min_ram_gb: float = 0
    disk_gb: int = 40
    min_inet_down_mbps: int = 300
    cuda_min: float = 13.0  # driver >= 580: 570.x cannot compile Blender 5.2 OptiX kernels
    providers: tuple[str, ...] = ()  # empty = any configured provider
    spot: bool = False  # interruptible/bid instances allowed (only for resume-safe work)
    profile: str = "base"  # box software profile (image + bootstrap), see profiles/

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Need":
        d = dict(d or {})
        if "providers" in d:
            d["providers"] = tuple(d["providers"])
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dc.fields(cls)}})


@dc.dataclass(frozen=True)
class Offer:
    provider: str
    offer_id: str
    gpu_name: str
    num_gpus: int
    vram_gb: float
    price_h: float  # what the scheduler compares: real $/h, or a shadow price for prepaid capacity
    cpu: int = 0
    ram_gb: float = 0
    inet_down_mbps: float = 0
    machine_id: str = ""
    host_id: str = ""
    spot: bool = False
    prepaid: bool = False
    boot_s: float = 240  # expected seconds from launch until the box takes work
    raw: dict = dc.field(default_factory=dict, compare=False, hash=False)


@dc.dataclass(frozen=True)
class SshTarget:
    host: str
    port: int = 22
    user: str = "root"
    proxy_command: str | None = None


@dc.dataclass(frozen=True)
class RemoteBox:
    """A box as the provider reports it, used by the reconciler."""

    ext_id: str
    label: str
    status: str  # provider-native status string
    terminal: bool = False  # provider says it will never run again
    created_at: float = 0
    price_h: float = 0


@dc.dataclass(frozen=True)
class Sync:
    """One rsync mapping. src/dst are local or box paths depending on direction; dst on the box is relative to the
    job workspace unless absolute."""

    src: str
    dst: str
    exclude: tuple[str, ...] = ()
    delete: bool = False
    include: tuple[str, ...] = ()  # if set: only these patterns (directories are always traversed)

    @classmethod
    def parse(cls, d: dict | str) -> "Sync":
        if isinstance(d, str):
            return cls(src=d, dst=d)
        return cls(src=d["src"], dst=d.get("dst", d["src"]), exclude=tuple(d.get("exclude", ())),
                   delete=bool(d.get("delete", False)), include=tuple(d.get("include", ())))

    def filters(self) -> list[str]:
        """rsync filter arguments: excludes first, then the include whitelist."""
        out = sum((["--exclude", x] for x in self.exclude), [])
        if self.include:
            out += ["--include", "*/", *sum((["--include", x] for x in self.include), []), "--exclude", "*"]
        return out


@dc.dataclass(frozen=True)
class JobSpec:
    """A batch of work units that share inputs, setup and a command template.

    Units are either named (``units=["shot-a", "shot-b"]``, one task each) or a numeric range (``range_=(1, 300)``)
    that the scheduler carves into chunks sized to ``chunk_target_s`` on the box that leases them, so fast GPUs take
    bigger chunks. The command sees GF_UNIT (named), or GF_LO/GF_HI (range), plus GF_TASK and GF_OUT.
    """

    project: str
    name: str
    command: str
    need: Need = Need()
    units: tuple[str, ...] = ()
    range_: tuple[int, int] | None = None
    unit_cost_s: float = 60  # seconds per unit on a reference GPU (score 16000); learned per workload over time
    unit_costs: dict = dc.field(default_factory=dict)  # per named unit, for bin-packing long shots first
    workload: str = ""  # perf-learning key, e.g. "blender-cycles-toon"; defaults to project/name
    chunk_target_s: float = 90
    min_chunk: int = 1
    max_chunk: int = 10_000
    inputs: tuple[Sync, ...] = ()
    outputs: tuple[Sync, ...] = ()  # pulled after each task; src is relative to the workspace
    # Partial results of a running task (frames rendered so far), pulled every checkpoint_s while it runs. When the
    # task is retried anywhere, they are pushed back first, so a resume-aware command skips finished work: a box
    # that dies mid-shot costs at most checkpoint_s of rendering, not the whole shot.
    checkpoint: tuple[Sync, ...] = ()
    checkpoint_s: float = 120
    setup: str = ""  # runs once per box per distinct setup text (cached by hash)
    env: dict = dc.field(default_factory=dict)
    gpu_share: float = 1.0  # lanes per GPU = 1/gpu_share to start with; telemetry may raise it
    max_per_box: int = 0  # at most this many of the job's tasks on one box at once (0 = no limit); 1 for commands
                          # that share fixed paths on the box
    timeout_factor: float = 3.0  # a task running longer than this x its expected time is killed and retried
    timeout_min_s: float = 900  # ...but never before this (first runs on a new GPU have no good estimate)
    max_attempts: int = 3
    speculate: bool = True  # duplicate straggling tail tasks on idle faster boxes
    resume_safe: bool = False  # the command can be killed and rerun without corrupting outputs (enables spot)
    priority: int = 0
    budget_usd: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> "JobSpec":
        d = dict(d)
        d["need"] = Need.from_dict(d.get("need"))
        d["units"] = tuple(d.get("units", ()))
        rng = d.pop("range", None)
        if rng is not None:
            d["range_"] = tuple(rng)
        elif d.get("range_") is not None:
            d["range_"] = tuple(d["range_"])
        d["inputs"] = tuple(Sync.parse(x) for x in d.get("inputs", ()))
        d["outputs"] = tuple(Sync.parse(x) for x in d.get("outputs", ()))
        d["checkpoint"] = tuple(Sync.parse(x) for x in d.get("checkpoint", ()))
        names = {f.name for f in dc.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown job fields: {sorted(unknown)}")
        spec = cls(**d)
        if bool(spec.units) == bool(spec.range_):
            raise ValueError("a job needs exactly one of units or range")
        return spec

    def to_dict(self) -> dict:
        d = dc.asdict(self)
        d["range"] = d.pop("range_")
        return d

    @property
    def workload_key(self) -> str:
        return self.workload or f"{self.project}/{self.name}"

    def inputs_key(self) -> str:
        """Identity of the input set; boxes that already hold it skip the full sync (rsync still sends deltas)."""
        blob = json.dumps([dc.asdict(s) for s in self.inputs], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def setup_key(self) -> str:
        return hashlib.sha256(self.setup.encode()).hexdigest()[:16] if self.setup else ""
