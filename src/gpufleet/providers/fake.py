"""In-memory provider for tests and dry runs. Boxes are local directories (the daemon's local_boxes mode)."""

from __future__ import annotations

import asyncio
import itertools
import time

from ..models import Need, Offer, RemoteBox, SshTarget
from .base import Provider, ProviderError


class Fake(Provider):
    def __init__(self, name: str = "fake", offers: list[dict] | None = None, boot_s: float = 0.05,
                 fail_launches: int = 0, max_boxes: int = 5, prepaid: bool = False, fail_boots: int = 0):
        self.name = name
        self.prepaid = prepaid
        self.max_boxes = max_boxes
        self._offers = offers or [{"gpu_name": "RTX 4090", "price_h": 0.4}]
        self.boot_s = boot_s
        self.fail_launches = fail_launches
        self.fail_boots = fail_boots  # created, then never becomes reachable
        self.live: dict[str, RemoteBox] = {}
        self.destroyed: list[str] = []
        self.launched: list[str] = []
        self._ids = itertools.count(1000)

    async def offers(self, need: Need, exclude_machines: set[str], exclude_hosts: set[str]) -> list[Offer]:
        out = []
        for i, o in enumerate(self._offers):
            mid = str(o.get("machine_id", 10 + i))
            if mid in exclude_machines:
                continue
            out.append(Offer(provider=self.name, offer_id=str(i), gpu_name=o["gpu_name"], num_gpus=o.get("num_gpus", 1),
                             vram_gb=24, price_h=o["price_h"], machine_id=mid, prepaid=self.prepaid,
                             boot_s=self.boot_s, raw={"cu_per_h": o.get("cu_per_h", 0)}))
        return out

    async def launch(self, offer: Offer, label: str, profile: dict, ttl_s: float) -> str:
        await asyncio.sleep(0)
        if self.fail_launches > 0:
            self.fail_launches -= 1
            raise ProviderError("fake launch failure", machine_fault=True)
        ext = str(next(self._ids))
        self.live[ext] = RemoteBox(ext_id=ext, label=label, status="running", created_at=time.time(),
                                   price_h=offer.price_h)
        self.launched.append(ext)
        return ext

    async def wait_ready(self, ext_id: str, deadline: float) -> SshTarget:
        await asyncio.sleep(self.boot_s)
        if self.fail_boots > 0:
            self.fail_boots -= 1
            raise ProviderError("fake boot failure", machine_fault=True)
        if ext_id not in self.live:
            raise ProviderError("gone")
        return SshTarget(host="localhost")

    supports_park = True

    async def park(self, ext_id: str) -> None:
        b = self.live[ext_id]
        self.live[ext_id] = RemoteBox(ext_id=ext_id, label=b.label, status="stopped", created_at=b.created_at)

    async def unpark(self, ext_id: str, deadline: float) -> SshTarget:
        b = self.live.get(ext_id)
        if b is None or getattr(self, "fail_unpark", False):
            raise ProviderError("gpu taken", machine_fault=False)
        self.live[ext_id] = RemoteBox(ext_id=ext_id, label=b.label, status="running", created_at=b.created_at)
        self.unparked = getattr(self, "unparked", 0) + 1
        return SshTarget(host="localhost")

    async def destroy(self, ext_id: str) -> None:
        if self.live.pop(ext_id, None) is not None:
            self.destroyed.append(ext_id)

    async def list_all(self) -> list[RemoteBox]:
        return list(self.live.values())
