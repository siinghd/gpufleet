"""Provider adapter interface. An adapter only talks to its provider; scheduling, retries and bookkeeping live in
the daemon, so a new provider is a few hundred lines."""

from __future__ import annotations

import abc

from ..models import Need, Offer, RemoteBox, SshTarget


class ProviderError(Exception):
    """Something the provider refused or failed. `retry_other` means: try a different offer/machine."""

    def __init__(self, msg: str, *, retry_other: bool = True, machine_fault: bool = False):
        super().__init__(msg)
        self.retry_other = retry_other
        self.machine_fault = machine_fault


class Provider(abc.ABC):
    name: str = ""
    prepaid: bool = False  # capacity already paid for (Colab compute units): marginal cost is a shadow price
    max_boxes: int = 5  # concurrent boxes this provider may hold for us

    @abc.abstractmethod
    async def offers(self, need: Need, exclude_machines: set[str], exclude_hosts: set[str]) -> list[Offer]:
        """Rentable offers that satisfy `need`, cheapest-first is not required (the scheduler ranks)."""

    @abc.abstractmethod
    async def launch(self, offer: Offer, label: str, profile: dict, ttl_s: float) -> str:
        """Rent the offer. Returns the provider's id. Must tag the box with `label` so list_all() can find it, and
        arrange for the box to destroy itself after ttl_s where the provider allows that."""

    @abc.abstractmethod
    async def wait_ready(self, ext_id: str, deadline: float) -> SshTarget:
        """Block until SSH is reachable or raise ProviderError at the (absolute, time.time()) deadline."""

    @abc.abstractmethod
    async def destroy(self, ext_id: str) -> None:
        """Release the box and stop billing. Must be idempotent (destroying a gone box is success)."""

    @abc.abstractmethod
    async def list_all(self) -> list[RemoteBox]:
        """Every box the account holds at this provider, ours or not."""

    async def balance(self) -> float | None:
        """Remaining credit (USD, or provider units for prepaid providers); None if unknown."""
        return None

    async def can_launch(self) -> tuple[bool, str]:
        """Account-level guard checked before every launch (credit floors, unit floors)."""
        return True, ""

    def ssh_options(self) -> list[str]:
        """Extra ssh -o options for this provider's boxes."""
        return []

    async def close(self) -> None:
        pass
