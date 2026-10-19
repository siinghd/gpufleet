"""Google Colab adapter, through the official google-colab-cli (>= 0.7.4).

Colab runtimes are prepaid compute units (a Google AI Pro plan grants 200 CU/month; G4 = RTX PRO 6000 burns ~8.9
CU/h). Rules that shape this adapter:
- SSH and long jobs are only allowed while the CU balance is positive, so we stop launching at a floor well above 0.
- One SSH connection per runtime (a second gets HTTP 429): the transport keeps a single OpenSSH ControlMaster per
  box and multiplexes the agent channel and rsync over it.
- Without background execution (AI Pro) the VM is reclaimed soon after the connection drops, so the connection must
  stay up for the whole job; the daemon holds it open while the box exists.
- No Docker on the VM: the profile's bootstrap script installs what the image would carry on Vast.
- We only ever touch runtimes we created (session names with our label prefix): a notebook the owner opened in the
  browser is listed by the same API and must survive the reconciler.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time

from ..models import Need, Offer, RemoteBox, SshTarget
from .base import Provider, ProviderError

# Approximate burn rates (CU/hour); the live rate from ccu-info replaces them once a runtime runs.
RATES = {"G4": 8.9, "L4": 1.7, "A100": 5.4, "T4": 1.2, "H100": 18.0}
GPU_NAMES = {"G4": "RTX PRO 6000 S", "L4": "L4", "A100": "A100 SXM4", "T4": "T4", "H100": "H100 SXM"}
VRAM = {"G4": 96, "L4": 24, "A100": 40, "T4": 16, "H100": 80}
CPUS = {"G4": 48, "L4": 12, "A100": 12, "T4": 2, "H100": 24}
USD_PER_CU = 0.10


class Colab(Provider):
    name = "colab"
    prepaid = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_boxes = int(cfg.get("max_boxes", 1))
        self.floor = float(cfg.get("cu_floor", 15))
        self.accelerators = list(cfg.get("accelerators", ["G4"]))
        # Units expire 90 days after grant, so spending them is cheaper than their sticker price. The shadow
        # factor sets how much a CU "costs" when the scheduler compares Colab with paid providers.
        self.shadow = float(cfg.get("shadow_factor", 0.35))
        self.cli = cfg.get("cli") or shutil.which("colab") or os.path.expanduser("~/.local/bin/colab")
        self.ssh_key = os.path.expanduser(cfg.get("ssh_key", "~/.ssh/id_ed25519_gpufleet"))
        self._client = None
        self._bal: tuple[float, float] | None = None

    # -- low level -----------------------------------------------------------------------------------------------
    def _api(self):
        """colab_cli's own client (same auth/token file as the CLI)."""
        if self._client is None:
            try:
                from colab_cli import auth, client
            except ImportError as e:
                raise ProviderError("google-colab-cli is not installed in this environment (pip install "
                                    "'gpufleet[colab]')", retry_other=False) from e
            self._client = client.Client(client.Prod(), auth.get_credentials())
        return self._client

    async def _call(self, fn, *a):
        return await asyncio.to_thread(fn, *a)

    async def _cli(self, *args: str, timeout: float = 300) -> tuple[int, str]:
        p = await asyncio.create_subprocess_exec(self.cli, *args, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.STDOUT,
                                                 env={**os.environ, "NO_COLOR": "1", "COLUMNS": "200"})
        try:
            out, _ = await asyncio.wait_for(p.communicate(), timeout)
        except asyncio.TimeoutError:
            p.kill()
            return 124, "timeout"
        return p.returncode or 0, out.decode(errors="replace")

    # -- account -------------------------------------------------------------------------------------------------
    async def balance(self) -> float | None:
        if self._bal and time.time() - self._bal[0] < 30:
            return self._bal[1]
        info = await self._call(lambda: self._api().get_consumption_user_info())
        self._bal = (time.time(), float(info.paid_compute_units_balance))
        return self._bal[1]

    async def can_launch(self) -> tuple[bool, str]:
        bal = await self.balance()
        # Keep an hour of G4 burn above the floor, so a box we start now never runs the balance to the floor.
        need = self.floor + max(RATES.get(a, 7) for a in self.accelerators)
        if bal is None or bal < need:
            return False, f"colab balance {bal} CU is under floor {self.floor} + 1 h of burn"
        return True, ""

    # -- offers --------------------------------------------------------------------------------------------------
    async def offers(self, need: Need, exclude_machines: set[str], exclude_hosts: set[str]) -> list[Offer]:
        if not need.gpu:
            return []  # never burn GPU units on CPU-only work
        from .. import perf
        out = []
        for acc in self.accelerators:
            gpu = GPU_NAMES.get(acc, acc)
            meta = perf.info(gpu)
            if need.rt_cores and not meta["rt"]:
                continue
            if need.min_vram_gb and VRAM.get(acc, 0) < need.min_vram_gb:
                continue
            if need.min_score and meta["score"] < need.min_score:
                continue
            if need.min_cpu and CPUS.get(acc, 0) < need.min_cpu:
                continue
            out.append(Offer(provider=self.name, offer_id=acc, gpu_name=gpu, num_gpus=1, vram_gb=VRAM.get(acc, 0),
                             price_h=RATES.get(acc, 7) * USD_PER_CU * self.shadow, cpu=CPUS.get(acc, 0),
                             prepaid=True, boot_s=float(self.cfg.get("boot_s", 150)),
                             raw={"cu_per_h": RATES.get(acc)}))
        return out

    # -- lifecycle -----------------------------------------------------------------------------------------------
    async def launch(self, offer: Offer, label: str, profile: dict, ttl_s: float) -> str:
        """Assign a runtime. 503 = no capacity right now (retry), 412 = too many assignments / usage limit
        (sometimes transient right after a failed assign), 400 = not entitled to that accelerator."""
        deadline = time.time() + float(self.cfg.get("assign_timeout_s", 420))
        last = ""
        while time.time() < deadline:
            rc, out = await self._cli("new", "-s", label, "--gpu", offer.offer_id, timeout=200)
            if rc == 0:
                return label
            last = out.strip().splitlines()[-1] if out.strip() else f"rc={rc}"
            if "Bad Request" in out or " 400" in out:
                raise ProviderError(f"colab: not entitled to {offer.offer_id}: {last}", retry_other=False)
            await asyncio.sleep(20)
        raise ProviderError(f"colab: could not assign {offer.offer_id}: {last}", retry_other=False)

    async def wait_ready(self, ext_id: str, deadline: float) -> SshTarget:
        # Assignment already returned a running VM; the transport's first SSH probes reachability.
        return SshTarget(host="colab-runtime", port=22, user="root",
                         proxy_command=f"{self.cli} ssh --proxy-mode -s {ext_id} -i {self.ssh_key}")

    async def _ours(self) -> dict[str, str]:
        """endpoint -> session name, for sessions the CLI created (its sessions.json)."""
        from colab_cli.state import StateStore
        sessions = await self._call(lambda: StateStore().list())
        return {s.endpoint: name for name, s in sessions.items() if getattr(s, "endpoint", None)}

    async def destroy(self, ext_id: str) -> None:
        ours = await self._ours()
        endpoint = next((e for e, n in ours.items() if n == ext_id), None)
        await self._cli("stop", "-s", ext_id, timeout=120)
        for _ in range(5):
            live = {a.endpoint for a in await self._call(lambda: self._api().list_assignments())}
            if endpoint is None or endpoint not in live:
                return
            await self._call(lambda: self._api().unassign(endpoint))
            await asyncio.sleep(3)
        raise ProviderError(f"colab runtime {ext_id} still assigned after stop", retry_other=False)

    async def list_all(self) -> list[RemoteBox]:
        ours = await self._ours()
        out = []
        for a in await self._call(lambda: self._api().list_assignments()):
            out.append(RemoteBox(ext_id=ours.get(a.endpoint, a.endpoint), label=ours.get(a.endpoint, ""),
                                 status="running", price_h=RATES.get(a.accelerator.value, 0) * USD_PER_CU))
        return out

    def ssh_options(self) -> list[str]:
        return ["-i", self.ssh_key, "-o", "IdentitiesOnly=yes"]
