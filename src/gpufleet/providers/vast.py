"""Vast.ai adapter (REST, https://docs.vast.ai/api-reference).

Search filters encode what we learned renting ~100 boxes: verified hosts, reliability >= 0.98, a host driver new
enough for the image's CUDA (driver 570 cannot compile Blender 5.2's OptiX kernels), real download bandwidth, and caps
on per-TB bandwidth prices (some hosts charge $50-80/TB, which can cost more than the GPU). Bad machines and bad
operators from our own ledger are excluded in the query itself.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from pathlib import Path

import httpx

from .. import perf
from ..models import Need, Offer, RemoteBox, SshTarget
from .base import Provider, ProviderError

API = "https://console.vast.ai/api/v0"
API1 = "https://console.vast.ai/api/v1"
TERMINAL = {"exited", "unknown", "offline"}


def _key() -> str:
    k = os.environ.get("VAST_API_KEY")
    if k:
        return k.strip()
    p = Path(os.environ.get("VAST_API_KEY_FILE", "~/.config/vastai/vast_api_key")).expanduser()
    if p.exists():
        return p.read_text().strip()
    raise ProviderError("no Vast API key (VAST_API_KEY or ~/.config/vastai/vast_api_key)", retry_other=False)


class Vast(Provider):
    name = "vast"
    supports_park = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_boxes = int(cfg.get("max_boxes", 5))
        self.credit_floor = float(cfg.get("credit_floor_usd", 0.60))
        self.max_bw_cost_tb = float(cfg.get("max_bandwidth_usd_per_tb", 10))
        self.ssh_key = os.path.expanduser(cfg.get("ssh_key", "~/.ssh/id_ed25519_vast"))
        self.http = httpx.AsyncClient(timeout=30, headers={"Authorization": f"Bearer {_key()}"})
        self._offer_cache: tuple[float, tuple, list[Offer]] | None = None

    async def _req(self, method: str, url: str, **kw) -> dict:
        """One API call with backoff on 429/5xx. Vast rate-limits per endpoint without Retry-After."""
        delay = 1.0
        for attempt in range(6):
            try:
                r = await self.http.request(method, url, **kw)
            except httpx.TransportError as e:
                if attempt == 5:
                    raise ProviderError(f"vast {method} {url}: {e}")
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if r.status_code == 429 or r.status_code >= 500:
                if "search_quota_exceeded" in r.text:
                    raise ProviderError("vast daily search quota exceeded", retry_other=False)
                await asyncio.sleep(float(r.headers.get("Retry-After", delay)))
                delay = min(delay * 2, 30)
                continue
            if r.status_code >= 400:
                raise ProviderError(f"vast {method} {url}: {r.status_code} {r.text[:300]}")
            return r.json() if r.content else {}
        raise ProviderError(f"vast {method} {url}: still rate-limited")

    # -- offers --------------------------------------------------------------------------------------------------
    def _query(self, need: Need, exclude_machines: set[str], exclude_hosts: set[str]) -> dict:
        q: dict = {
            "verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True}, "rented": {"eq": False},
            "num_gpus": {"eq": 1},
            "reliability": {"gte": float(self.cfg.get("min_reliability", 0.98))},
            "cuda_max_good": {"gte": need.cuda_min},
            "inet_down": {"gte": need.min_inet_down_mbps},
            "inet_down_cost": {"lte": self.max_bw_cost_tb / 1000},
            "inet_up_cost": {"lte": self.max_bw_cost_tb / 1000},
            "dph_total": {"lte": need.max_price_h},
            "disk_space": {"gte": need.disk_gb},
            "direct_port_count": {"gte": 1},
            "order": [["dph_total", "asc"]],
            "type": "bid" if need.spot else "on-demand",
            "limit": 200,
            "allocated_storage": float(need.disk_gb),
        }
        if not need.gpu:
            q["num_gpus"] = {"gte": 0}
        if need.min_vram_gb:
            q["gpu_ram"] = {"gte": need.min_vram_gb * 1000}
        if need.min_cpu:
            q["cpu_cores_effective"] = {"gte": need.min_cpu}
        if need.min_ram_gb:
            q["cpu_ram"] = {"gte": need.min_ram_gb * 1000}
        names = self.cfg.get("gpu_names")
        if names:
            q["gpu_name"] = {"in": list(names)}
        if exclude_machines:
            q["machine_id"] = {"notin": sorted(int(m) for m in exclude_machines if str(m).isdigit())}
        if exclude_hosts:
            q["host_id"] = {"notin": sorted(int(h) for h in exclude_hosts if str(h).isdigit())}
        return q

    async def offers(self, need: Need, exclude_machines: set[str], exclude_hosts: set[str]) -> list[Offer]:
        key = (need, frozenset(exclude_machines), frozenset(exclude_hosts))
        if self._offer_cache and self._offer_cache[1] == key and time.time() - self._offer_cache[0] < 45:
            return self._offer_cache[2]
        body = self._query(need, exclude_machines, exclude_hosts)
        data = await self._req("POST", f"{API}/bundles/", json=body)
        out = []
        for o in data.get("offers", []):
            gpu = o.get("gpu_name", "")
            meta = perf.info(gpu)
            if need.gpu and need.rt_cores and not meta["rt"]:
                continue
            if need.nvenc and not meta["nvenc"]:
                continue
            if need.min_score and meta["score"] < need.min_score:
                continue
            # Bid 15% over the minimum: at the minimum the first competing bid preempts us.
            price = float(o.get("min_bid") or o["dph_total"]) * 1.15 if need.spot else float(o["dph_total"])
            out.append(Offer(
                provider=self.name, offer_id=str(o["id"]), gpu_name=gpu, num_gpus=int(o.get("num_gpus") or 1),
                vram_gb=float(o.get("gpu_ram") or 0) / 1000, price_h=price, cpu=int(o.get("cpu_cores_effective") or 0),
                ram_gb=float(o.get("cpu_ram") or 0) / 1000, inet_down_mbps=float(o.get("inet_down") or 0),
                machine_id=str(o.get("machine_id", "")), host_id=str(o.get("host_id", "")), spot=need.spot,
                raw={k: o.get(k) for k in ("dph_total", "min_bid", "reliability2", "geolocation", "driver_version",
                                           "cuda_max_good", "inet_up", "disk_bw", "inet_down_cost", "inet_up_cost")},
            ))
        self._offer_cache = (time.time(), key, out)
        return out

    # -- lifecycle -----------------------------------------------------------------------------------------------
    async def launch(self, offer: Offer, label: str, profile: dict, ttl_s: float) -> str:
        image = profile.get("vast_image") or profile.get("image")
        if not image:
            raise ProviderError(f"profile {profile.get('name')} has no vast_image", retry_other=False)
        env = {"NVIDIA_DRIVER_CAPABILITIES": "all", "GF_LABEL": label, **profile.get("env", {})}
        # Belt and braces with the daemon's own watchdog: the box deletes itself at its hard deadline using the
        # per-container key Vast injects (it can only stop/destroy this one instance).
        onstart = (
            # sshd refuses keys when /root/.ssh has the wrong owner or modes ("bad ownership or modes for file
            # /root/.ssh/authorized_keys" on some hosts/images): keep repairing it while Vast installs the key.
            "(for i in $(seq 60); do chown root:root /root /root/.ssh /root/.ssh/authorized_keys 2>/dev/null; "
            "chmod go-w /root; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys 2>/dev/null; sleep 2; "
            "done) >/dev/null 2>&1 & "
            "env | grep -E '^(CONTAINER_|VAST_|GF_|NVIDIA_)' >> /etc/environment; "
            f"nohup sh -c 'sleep {int(ttl_s)}; curl -fsS -X DELETE -H \"Authorization: Bearer $CONTAINER_API_KEY\" "
            f"{API}/instances/$CONTAINER_ID/' >/dev/null 2>&1 &"
        )
        body = {
            "client_id": "me", "image": image, "env": env, "disk": float(profile.get("disk_gb", 40)),
            "label": label, "onstart": onstart, "runtype": "ssh_direc ssh_proxy", "cancel_unavail": True,
            "price": offer.price_h if offer.spot else None, "image_login": profile.get("image_login"),
        }
        try:
            r = await self._req("PUT", f"{API}/asks/{offer.offer_id}/", json=body)
        except ProviderError as e:
            if self._offer_cache:  # a stale listing: never hand it out again from the cache
                t, k, offers = self._offer_cache
                self._offer_cache = (t, k, [o for o in offers if o.offer_id != offer.offer_id])
            raise ProviderError(f"create failed for offer {offer.offer_id}: {e}", retry_other=True)
        ext = r.get("new_contract")
        if not ext:
            raise ProviderError(f"create returned no contract: {r}")
        if self.cfg.get("attach_key"):
            # Only for a key that is not the account's own (Vast installs account keys itself; attaching the same
            # key again was followed by sshd refusing authorized_keys on every box, 19 Oct 2026).
            try:
                await self._req("POST", f"{API}/instances/{ext}/ssh/",
                                json={"ssh_key": Path(self.ssh_key + ".pub").read_text().strip()})
            except ProviderError:
                pass
        return str(ext)

    async def _show(self, ext_id: str) -> dict | None:
        try:
            d = await self._req("GET", f"{API}/instances/{ext_id}/", params={"owner": "me"})
        except ProviderError as e:
            if " 404 " in str(e) or "not found" in str(e).lower():
                return None
            raise
        inst = d.get("instances")
        return inst if isinstance(inst, dict) and inst else None

    async def wait_ready(self, ext_id: str, deadline: float, grace_s: float = 0) -> SshTarget:
        last = ""
        started = time.time()
        while time.time() < deadline:
            inst = await self._show(ext_id)
            if inst is None:
                raise ProviderError(f"instance {ext_id} vanished while booting", machine_fault=True)
            st = inst.get("actual_status") or ""
            last = f"{st} {inst.get('status_msg') or ''}".strip()
            if st in TERMINAL and time.time() - started > grace_s:
                raise ProviderError(f"instance {ext_id} went {st}: {inst.get('status_msg')}", machine_fault=True)
            if st == "running":
                port = None
                for m in (inst.get("ports") or {}).get("22/tcp") or []:
                    if m.get("HostPort"):
                        port = int(m["HostPort"])
                        break
                ip = (inst.get("public_ipaddr") or "").strip()
                if ip and port:
                    return SshTarget(host=ip, port=port)
                if inst.get("ssh_host") and inst.get("ssh_port"):
                    return SshTarget(host=inst["ssh_host"], port=int(inst["ssh_port"]))
            await asyncio.sleep(4)
        raise ProviderError(f"instance {ext_id} not ready by deadline (last: {last})", machine_fault=True)

    async def park(self, ext_id: str) -> None:
        await self._req("PUT", f"{API}/instances/{ext_id}/", json={"state": "stopped"})

    async def unpark(self, ext_id: str, deadline: float) -> SshTarget:
        # A stopped box restarts in seconds if its GPU is still free; if someone rented the GPU meanwhile it sits in
        # "scheduling" indefinitely, which the deadline turns into a clean failure (the caller destroys it).
        await self._req("PUT", f"{API}/instances/{ext_id}/", json={"state": "running"})
        return await self.wait_ready(ext_id, deadline, grace_s=60)

    async def destroy(self, ext_id: str) -> None:
        for _ in range(6):
            try:
                await self._req("DELETE", f"{API}/instances/{ext_id}/")
            except ProviderError as e:
                if "404" not in str(e):
                    await asyncio.sleep(3)
                    continue
            if await self._show(ext_id) is None:
                return
            await asyncio.sleep(3)
        raise ProviderError(f"instance {ext_id} still exists after 6 destroy attempts", retry_other=False)

    async def list_all(self) -> list[RemoteBox]:
        out, token = [], None
        while True:
            params = {"limit": 25}
            if token:
                params["after_token"] = token
            d = await self._req("GET", f"{API1}/instances/", params=params)
            for i in d.get("instances", []):
                st = i.get("actual_status") or i.get("cur_state") or ""
                parked = i.get("intended_status") == "stopped"
                out.append(RemoteBox(ext_id=str(i["id"]), label=i.get("label") or "", status=st,
                                     terminal=st in TERMINAL and not parked,
                                     created_at=float(i.get("start_date") or 0),
                                     price_h=float(i.get("dph_total") or 0)))
            token = d.get("next_token")
            if not token:
                return out

    async def balance(self) -> float | None:
        d = await self._req("GET", f"{API}/users/current/")
        return float(d.get("credit", 0)) + float(d.get("balance", 0) or 0)

    async def can_launch(self) -> tuple[bool, str]:
        bal = await self.balance()
        if bal is not None and bal < self.credit_floor:
            return False, f"vast credit ${bal:.2f} is under the ${self.credit_floor:.2f} floor"
        return True, ""

    def ssh_options(self) -> list[str]:
        return ["-i", self.ssh_key, "-o", "IdentitiesOnly=yes"]

    async def close(self) -> None:
        await self.http.aclose()
