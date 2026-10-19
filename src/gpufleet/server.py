"""fleetd's unix-socket API: one JSON object per line in, one (or, for `wait`, a stream of) JSON lines out.

Any language can use it (`socat - UNIX-CONNECT:~/.local/state/gpufleet/fleetd.sock`); the `fleet` CLI and
gpufleet.client wrap it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from pathlib import Path

from . import config
from .daemon import Fleet
from .models import JobSpec, Need
from .providers.base import Provider

log = logging.getLogger("gpufleet")


def build_providers(cfg: dict) -> dict[str, Provider]:
    out: dict[str, Provider] = {}
    pc = cfg["providers"]
    if pc.get("vast", {}).get("enabled"):
        from .providers.vast import Vast
        out["vast"] = Vast(pc["vast"])
    if pc.get("colab", {}).get("enabled"):
        from .providers.colab import Colab
        out["colab"] = Colab(pc["colab"])
    return out


class Server:
    def __init__(self, fleet: Fleet, path: str):
        self.fleet = fleet
        self.path = path

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while line := await reader.readline():
                try:
                    req = json.loads(line)
                except ValueError:
                    await self._send(writer, {"error": "bad json"})
                    continue
                try:
                    if req.get("op") == "wait":
                        await self._wait(req, writer)
                    else:
                        await self._send(writer, await self.dispatch(req))
                except Exception as e:  # report to the client, keep the daemon alive
                    log.exception("request %s", req.get("op"))
                    await self._send(writer, {"error": f"{type(e).__name__}: {e}"})
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    async def _send(self, writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj, default=str) + "\n").encode())
        await writer.drain()

    async def _wait(self, req: dict, writer: asyncio.StreamWriter) -> None:
        jid = req["job"]
        q: asyncio.Queue = asyncio.Queue()
        self.fleet.waiters.setdefault(jid, []).append(q)
        try:
            st = self.fleet.job_status(jid)
            if st.get("status") in ("done", "failed", "cancelled") or "error" in st:
                await self._send(writer, {"ev": "final", **st})
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), 30)
                except asyncio.TimeoutError:
                    await self._send(writer, {"ev": "progress", **self.fleet.job_status(jid)})
                    continue
                await self._send(writer, ev)
                if ev.get("ev") in ("job_done", "job_failed", "job_cancelled"):
                    await self._send(writer, {"ev": "final", **self.fleet.job_status(jid)})
                    return
        finally:
            self.fleet.waiters[jid].remove(q)

    async def dispatch(self, req: dict) -> dict:
        f = self.fleet
        op = req.get("op")
        if op == "submit":
            spec = JobSpec.from_dict(req["spec"])
            return {"job": f.submit(spec)}
        if op == "status":
            return f.job_status(req["job"]) if req.get("job") else f.fleet_status()
        if op == "cancel":
            f.cancel_job(req["job"])
            return {"ok": True}
        if op == "tasks":
            rows = f.store.q("SELECT id, unit, lo, hi, status, box_id, attempt, rc, started_at, finished_at, tail "
                             "FROM tasks WHERE job_id=? ORDER BY rowid", [req["job"]])
            return {"tasks": [dict(r) for r in rows]}
        if op == "warm":
            return f.set_warm(req["profile"], int(req.get("boxes", 1)), float(req.get("minutes", 30)),
                              req.get("need"))
        if op == "drain":
            await f.drain()
            return {"ok": True}
        if op == "reconcile":
            return await f.reconcile()
        if op == "offers":
            need = Need.from_dict(req.get("need"))
            out = []
            for name, prov in f.providers.items():
                bad_m, bad_h = f.store.bad_machines(name)
                ok, why = await prov.can_launch()
                offers = await prov.offers(need, bad_m, bad_h)
                for o in sorted(offers, key=lambda o: o.price_h)[: int(req.get("limit", 8))]:
                    out.append({"provider": name, "gpu": o.gpu_name, "price_h": round(o.price_h, 3),
                                "prepaid": o.prepaid, "machine": o.machine_id, "can_launch": ok, "why": why})
            return {"offers": out}
        if op == "usage":
            bal = {}
            for name, prov in f.providers.items():
                try:
                    bal[name] = await prov.balance()
                except Exception as e:
                    bal[name] = f"error: {e}"
            return {"balances": bal, **f.fleet_status()}
        if op == "events":
            rows = f.store.q("SELECT * FROM events ORDER BY at DESC LIMIT ?", [int(req.get("n", 50))])
            return {"events": [dict(r) for r in rows]}
        if op == "ping":
            return {"pong": True, "pid": os.getpid()}
        return {"error": f"unknown op {op!r}"}


async def serve(cfg: dict | None = None) -> None:
    cfg = cfg or config.load()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    fleet = Fleet(cfg, build_providers(cfg))
    path = cfg["socket"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(path):
        os.unlink(path)
    srv = Server(fleet, path)
    server = await asyncio.start_unix_server(srv.handle, path=path, limit=1 << 24)
    os.chmod(path, 0o600)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    runner = asyncio.create_task(fleet.run())
    log.info("fleetd listening on %s with providers %s", path, list(fleet.providers))
    await stop.wait()
    log.info("shutting down (boxes keep running; the next start reattaches)")
    server.close()
    await fleet.shutdown(destroy=False)
    runner.cancel()
