"""fleetd: the scheduler. One asyncio process owns every box, job and task; clients talk to it over a unix socket.

Loop, every second:
  1. dispatch  - free lanes on ready boxes take the next task they can run (named units longest-first; range jobs
                 are carved into chunks sized for the leasing GPU), after the job's inputs and setup are on the box
  2. tail      - when a job has nothing left to hand out, idle boxes duplicate straggling tasks; first result wins
  3. scale     - if work waits longer than scale_delay_s, rent the offer with the lowest expected cost
                 (price x (boot + work) + value of the waiting time), within per-provider, global and daily caps
  4. lifecycle - idle boxes are kept idle_s for reuse, retired after reuse_max_age_s, destroyed at hard_ttl_s
  5. lanes     - a box whose lanes are all busy but whose GPU is under-used (and has VRAM to spare) gets a lane more
Every reconcile_s the reconciler compares each provider's instance list with the database and destroys anything of
ours that is unknown, terminal or past its deadline. Spend is accrued every tick from the boxes' hourly prices.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import perf, profiles
from .models import JobSpec, Need, Offer, SshTarget, Sync
from .providers.base import Provider, ProviderError
from .store import Store
from .transport import LocalTransport, SshTransport, Transport, TransportError

log = logging.getLogger("gpufleet")
AGENT_SRC = Path(__file__).with_name("agent.py")
MACHINE_FAULT_RCS = {3, 75}  # job scripts exit 3 for "no usable GPU" (toon-studio LIB_REQUIRE_GPU), 75 = bad box


def _id(prefix: str) -> str:
    return prefix + secrets.token_hex(4)


@dataclass
class Lane:
    task: str | None = None
    since: float = 0.0


@dataclass
class BoxRT:
    """Live state of one box. The database row holds the durable part."""

    id: str
    provider: str
    ext_id: str = ""
    gpu: str = ""
    num_gpus: int = 1
    price_h: float = 0.0
    units_h: float = 0.0  # prepaid providers: units burned per hour
    prepaid: bool = False
    spot: bool = False
    machine_id: str = ""
    host_id: str = ""
    profile: str = "base"
    driver: str = ""
    ncpu: int = 0
    status: str = "requested"
    created: float = field(default_factory=time.time)
    ready_at: float = 0.0
    idle_since: float = field(default_factory=time.time)
    transport: Transport | None = None
    proc: asyncio.subprocess.Process | None = None
    reader: asyncio.Task | None = None
    lanes: list[Lane] = field(default_factory=list)
    prepared: dict = field(default_factory=dict)  # job id -> "ok" | asyncio.Task | Exception
    setups: set = field(default_factory=set)
    tel: list = field(default_factory=list)  # (t, util, mem, mem_total) from the agent
    full_since: float = 0.0
    last_seen: float = field(default_factory=time.time)  # last event from the agent (telemetry every 5 s)
    lost: bool = False

    def free_lanes(self) -> list[int]:
        return [i for i, l in enumerate(self.lanes) if l.task is None]

    def busy(self) -> int:
        return sum(1 for l in self.lanes if l.task)


class Fleet:
    def __init__(self, cfg: dict, providers: dict[str, Provider], store: Store | None = None,
                 local_boxes: bool = False):
        self.cfg = cfg
        self.providers = providers
        self.store = store or Store(cfg["db"])
        self.state_dir = Path(cfg["state_dir"])
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.boxes: dict[str, BoxRT] = {}
        self.jobs: dict[str, JobSpec] = {}
        self.waiters: dict[str, list[asyncio.Queue]] = {}
        self.local_boxes = local_boxes  # tests: boxes are local directories
        self.launching: set[str] = set()
        self._last_reconcile = 0.0
        self._last_accrue = time.time()
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()  # set on submit / task exit / box ready: dispatch now, not at the next tick
        self._pulls = asyncio.Semaphore(int(cfg.get("max_parallel_pulls", 6)))
        self._ckpt_at: dict[str, float] = {}  # task id -> last checkpoint pull
        self._ckpt_busy: set[str] = set()
        self.warm: dict[str, dict] = {}  # profile -> {"boxes": n, "until": t, "need": Need}

    # =========================================================================================== jobs and tasks
    def submit(self, spec: JobSpec) -> str:
        jid = _id("j")
        now = time.time()
        self.store.x("INSERT INTO jobs (id, project, name, spec, status, created_at, cursor) VALUES (?,?,?,?,?,?,?)",
                     [jid, spec.project, spec.name, json.dumps(spec.to_dict()), "queued", now,
                      spec.range_[0] if spec.range_ else None])
        if spec.units:
            # Longest first: with greedy assignment that is the classic LPT schedule, the tail is a short unit.
            order = sorted(spec.units, key=lambda u: -float(spec.unit_costs.get(u, spec.unit_cost_s)))
            for u in order:
                self.store.x("INSERT INTO tasks (id, job_id, unit, status, est_s) VALUES (?,?,?,?,?)",
                             [_id("t"), jid, u, "queued", float(spec.unit_costs.get(u, spec.unit_cost_s))])
        self.jobs[jid] = spec
        self._wake.set()
        self.store.event("job_submitted", jid, project=spec.project, name=spec.name)
        self._notify(jid, ev="submitted")
        return jid

    def _spec(self, jid: str) -> JobSpec:
        if jid not in self.jobs:
            r = self.store.one("SELECT spec FROM jobs WHERE id=?", [jid])
            self.jobs[jid] = JobSpec.from_dict(json.loads(r["spec"]))
        return self.jobs[jid]

    def active_jobs(self) -> list[str]:
        rows = self.store.q("SELECT id FROM jobs WHERE status IN ('queued','running') ORDER BY created_at")
        jids = [r["id"] for r in rows]
        return sorted(jids, key=lambda j: -self._spec(j).priority)

    def _range_left(self, jid: str) -> int:
        spec = self._spec(jid)
        if not spec.range_:
            return 0
        cur = self.store.one("SELECT cursor FROM jobs WHERE id=?", [jid])["cursor"]
        return max(0, spec.range_[1] - cur + 1)

    def pending_secs(self, jid: str) -> float:
        """Queued work of a job in reference-GPU seconds."""
        spec = self._spec(jid)
        q = self.store.one("SELECT COALESCE(SUM(est_s),0) AS s, COUNT(*) AS n FROM tasks WHERE job_id=? AND "
                           "status='queued'", [jid])
        rng = self._range_left(jid) * spec.unit_cost_s
        return float(q["s"]) + rng

    def has_queued(self, jid: str) -> bool:
        if self._range_left(jid) > 0:
            return True
        return self.store.one("SELECT 1 FROM tasks WHERE job_id=? AND status='queued' LIMIT 1", [jid]) is not None

    def oldest_wait(self, jid: str) -> float:
        r = self.store.one("SELECT created_at FROM jobs WHERE id=?", [jid])
        last = self.store.one("SELECT MAX(finished_at) AS f FROM tasks WHERE job_id=?", [jid])
        start = max(r["created_at"], last["f"] or 0)
        return time.time() - start

    def _secs_per_unit(self, spec: JobSpec, gpu: str) -> float:
        return perf.secs_per_unit(self.store, gpu, spec.workload_key, spec.unit_cost_s)

    def _lease(self, jid: str, box: BoxRT) -> dict | None:
        """Take the next unit of work of `jid` for `box`; returns the task row as a dict or None."""
        spec = self._spec(jid)
        excl_ok = lambda r: box.id not in json.loads(r["excluded"]) and box.machine_id not in json.loads(r["excluded"])
        for r in self.store.q("SELECT * FROM tasks WHERE job_id=? AND status='queued' ORDER BY rowid", [jid]):
            if excl_ok(r):
                return dict(r)
        if spec.range_:
            cur = self.store.one("SELECT cursor FROM jobs WHERE id=?", [jid])["cursor"]
            hi_all = spec.range_[1]
            if cur > hi_all:
                return None
            spu = self._secs_per_unit(spec, box.gpu)
            n = max(spec.min_chunk, min(spec.max_chunk, int(spec.chunk_target_s / max(spu, 1e-3))))
            # Near the end, split what is left across every lane that could take it, so no lane gets a long tail.
            lanes = sum(len(b.lanes) for b in self.boxes.values() if b.status == "ready") or 1
            left = hi_all - cur + 1
            n = max(spec.min_chunk, min(n, math.ceil(left / lanes)))
            lo, hi = cur, min(hi_all, cur + n - 1)
            self.store.x("UPDATE jobs SET cursor=? WHERE id=?", [hi + 1, jid])
            tid = _id("t")
            self.store.x("INSERT INTO tasks (id, job_id, lo, hi, status, est_s) VALUES (?,?,?,?,?,?)",
                         [tid, jid, lo, hi, "queued", (hi - lo + 1) * spec.unit_cost_s])
            return dict(self.store.one("SELECT * FROM tasks WHERE id=?", [tid]))
        return None

    def _fits(self, spec: JobSpec, box: BoxRT) -> bool:
        need = spec.need
        if need.providers and box.provider not in need.providers:
            return False
        if box.spot and not (need.spot and spec.resume_safe):
            return False
        if box.profile != need.profile:
            return False
        if need.gpu:
            meta = perf.info(box.gpu)
            if need.rt_cores and not meta["rt"]:
                return False
            if need.min_score and meta["score"] < need.min_score:
                return False
        return True

    @staticmethod
    def _subst(text: str, task: dict, spec: JobSpec) -> str:
        unit = task.get("unit") or ""
        lo = task.get("lo") if task.get("lo") is not None else ""
        hi = task.get("hi") if task.get("hi") is not None else ""
        return (text.replace("{unit}", str(unit)).replace("{lo}", str(lo)).replace("{hi}", str(hi))
                .replace("{task}", task["id"]).replace("{project}", spec.project))

    def _workspace(self, spec: JobSpec) -> str:
        return f"{spec.project}"

    # ================================================================================================ dispatch
    async def _prepare(self, box: BoxRT, jid: str) -> None:
        """Inputs + setup for a job on a box. Inputs land in the project workspace, so a later job of the same project
        on a warm box only sends what changed."""
        spec = self._spec(jid)
        ws = self._workspace(spec)
        t0 = time.time()
        sem = asyncio.Semaphore(4)

        async def push(s: Sync):
            async with sem:
                dst = s.dst if s.dst.startswith("/") else f"{ws}/{s.dst}"
                await box.transport.push(Sync(src=s.src, dst=dst, exclude=s.exclude, delete=s.delete))

        await asyncio.gather(*(push(s) for s in spec.inputs))
        if spec.setup and spec.setup_key() not in box.setups:
            rc, out = await box.transport.exec(
                f"cd {shlex.quote(box.transport.abspath(ws))} && {spec.setup}", timeout=1800)
            if rc:
                raise TransportError(f"setup failed rc={rc}: {out[-500:]}", rc)
            box.setups.add(spec.setup_key())
        self.store.event("prepared", box.id, job=jid, secs=round(time.time() - t0, 1))

    def _start(self, box: BoxRT, lane: int, jid: str, task: dict, dup_of: str | None = None) -> None:
        spec = self._spec(jid)
        prof = profiles.get(spec.need.profile, self.cfg.get("profiles"))
        env = {**prof.get("env", {}), **spec.env, "GF_JOB": jid, "GF_PROJECT": spec.project,
               "GF_ATTEMPT": str(task["attempt"] + 1)}
        if task.get("unit"):
            env["GF_UNIT"] = task["unit"]
        if task.get("lo") is not None:
            env["GF_LO"], env["GF_HI"] = str(task["lo"]), str(task["hi"])
        gpu = lane % box.num_gpus if box.num_gpus > 1 else None
        msg = {"op": "run", "task": task["id"], "cmd": self._subst(spec.command, task, spec), "env": env,
               "cwd": self._workspace(spec), "gpu": gpu}
        box.lanes[lane] = Lane(task=task["id"], since=time.time())
        self.store.update("tasks", task["id"], status="running", box_id=box.id, started_at=time.time(),
                          attempt=task["attempt"] + (0 if dup_of else 1))
        self.store.x("UPDATE jobs SET status='running' WHERE id=? AND status='queued'", [jid])
        if spec.checkpoint:
            asyncio.create_task(self._restore_then_run(box, spec, task, msg))
        else:
            self._send(box, msg)
        self._notify(jid, ev="started", task=task["id"], unit=task.get("unit"), lo=task.get("lo"),
                     hi=task.get("hi"), box=box.id, gpu=box.gpu)

    async def _restore_then_run(self, box: BoxRT, spec: JobSpec, task: dict, msg: dict) -> None:
        """Push the task's saved partial results (from an earlier attempt, on any box) back before it starts."""
        for c in spec.checkpoint:
            local = os.path.expanduser(self._subst(c.dst, task, spec))
            if os.path.exists(local) and (not os.path.isdir(local) or any(os.scandir(local))):
                src = local.rstrip("/") + ("/" if os.path.isdir(local) else "")
                dst = f"{self._workspace(spec)}/{self._subst(c.src, task, spec)}"
                try:
                    await box.transport.push(Sync(src=src, dst=dst), timeout=1800)
                    self.store.event("checkpoint_restored", task["id"], box=box.id, path=c.src)
                except TransportError as e:
                    log.warning("checkpoint restore for %s: %s", task["id"], e)
        self._send(box, msg)

    async def _pull_checkpoint(self, box: BoxRT, spec: JobSpec, task: dict, timeout: float = 600) -> bool:
        if not spec.checkpoint or not box.transport:
            return True
        ok = True
        for c in spec.checkpoint:
            src = f"{self._workspace(spec)}/{self._subst(c.src, task, spec)}"
            try:
                await box.transport.pull(Sync(src=src, dst=self._subst(c.dst, task, spec), exclude=c.exclude),
                                         timeout=timeout)
            except TransportError as e:
                ok = False
                log.info("checkpoint pull %s: %s", task["id"], e)
        self._ckpt_at[task["id"]] = time.time()
        return ok

    async def _pull_outputs(self, box: BoxRT, spec: JobSpec, task: dict) -> None:
        """Outputs of a finished task, with retries: a link that drops for a minute must not cost the render."""
        delays = [5, 15, 40]
        for attempt in range(len(delays) + 1):
            try:
                async with self._pulls:
                    await asyncio.gather(*(box.transport.pull(Sync(
                        src=f"{self._workspace(spec)}/{self._subst(s.src, task, spec)}",
                        dst=self._subst(s.dst, task, spec), exclude=s.exclude)) for s in spec.outputs))
                return
            except TransportError:
                if attempt == len(delays):
                    raise
                await asyncio.sleep(delays[attempt])
                if isinstance(box.transport, SshTransport) and not box.transport.alive:
                    await box.transport.open(timeout=60)

    def checkpoints(self) -> None:
        now = time.time()
        for r in self.store.q("SELECT * FROM tasks WHERE status='running'"):
            tid = r["id"]
            if tid in self._ckpt_busy:
                continue
            spec = self._spec(r["job_id"])
            box = self.boxes.get(r["box_id"] or "")
            if not spec.checkpoint or not box or box.status not in ("ready", "draining"):
                continue
            if now - max(self._ckpt_at.get(tid, 0), r["started_at"] or now) < spec.checkpoint_s:
                continue
            self._ckpt_busy.add(tid)

            async def go(b=box, sp=spec, t=dict(r)):
                try:
                    await self._pull_checkpoint(b, sp, t)
                finally:
                    self._ckpt_busy.discard(t["id"])

            asyncio.create_task(go())

    def watchdogs(self) -> None:
        """Heartbeats and task timeouts."""
        now = time.time()
        for b in self.boxes.values():
            if b.status in ("ready", "draining") and b.proc and b.proc.returncode is None \
                    and now - b.last_seen > float(self.cfg.get("heartbeat_s", 45)):
                # The agent sends telemetry every 5 s. Silence means the link (or box) hung: kill the channel; the
                # reader's exit path reconnects or, failing that, requeues and destroys.
                log.warning("box %s silent for %.0fs", b.id, now - b.last_seen)
                self.store.event("heartbeat_lost", b.id, silent_s=round(now - b.last_seen))
                b.last_seen = now
                b.proc.kill()
        for r in self.store.q("SELECT * FROM tasks WHERE status='running' AND started_at IS NOT NULL"):
            spec = self._spec(r["job_id"])
            box = self.boxes.get(r["box_id"] or "")
            if not box:
                continue
            units = (r["hi"] - r["lo"] + 1) if r["lo"] is not None else 1
            weight = (float(spec.unit_costs.get(r["unit"], spec.unit_cost_s)) / spec.unit_cost_s) if r["unit"] else 1
            expect = self._secs_per_unit(spec, box.gpu) * units * weight
            limit = max(spec.timeout_min_s, spec.timeout_factor * expect)
            if now - r["started_at"] > limit:
                log.warning("task %s over its %.0fs limit on %s", r["id"], limit, box.id)
                self.store.event("task_timeout", r["id"], box=box.id, limit=round(limit))
                self._send(box, {"op": "cancel", "task": r["id"]})
                for l in box.lanes:
                    if l.task == r["id"]:
                        l.task = None
                excl = json.loads(r["excluded"]) + [box.id]
                if r["dup_of"] or r["attempt"] >= spec.max_attempts:
                    self.store.update("tasks", r["id"], status="failed", rc=124, finished_at=now,
                                      tail=f"timed out after {limit:.0f}s")
                else:
                    self.store.update("tasks", r["id"], status="queued", box_id=None, rc=124,
                                      excluded=json.dumps(excl))
                self._notify(r["job_id"], ev="timeout", task=r["id"], limit=round(limit), box=box.id)
                self._finish_jobs()

    def _send(self, box: BoxRT, msg: dict) -> None:
        if box.proc and box.proc.stdin and not box.proc.stdin.is_closing():
            box.proc.stdin.write((json.dumps(msg) + "\n").encode())

    def dispatch(self) -> None:
        jobs = self.active_jobs()
        for box in list(self.boxes.values()):
            if box.status != "ready" or not box.free_lanes():
                continue
            for jid in jobs:
                spec = self._spec(jid)
                if not self._fits(spec, box) or not self.has_queued(jid):
                    continue
                st = box.prepared.get(jid)
                if st is None:
                    box.prepared[jid] = asyncio.create_task(self._prepare(box, jid))
                    break
                if isinstance(st, asyncio.Task):
                    if not st.done():
                        break  # this box is busy getting the job's inputs; keep its lanes for it
                    exc = st.exception()
                    box.prepared[jid] = "ok" if exc is None else exc
                    if exc is not None:
                        log.warning("prepare %s on %s failed: %s", jid, box.id, exc)
                        self.store.event("prepare_failed", box.id, job=jid, error=str(exc)[-300:])
                        self._box_fault(box, f"prepare failed: {exc}")
                        break
                    st = "ok"
                if isinstance(st, Exception):
                    continue
                for lane in box.free_lanes():
                    task = self._lease(jid, box)
                    if not task:
                        break
                    self._start(box, lane, jid, task)
                if not box.free_lanes():
                    break

    def speculate(self) -> None:
        """Tail: duplicate the slowest running task of a job on an idle box that is at least as fast."""
        for jid in self.active_jobs():
            spec = self._spec(jid)
            if not spec.speculate or self.has_queued(jid):
                continue
            running = self.store.q("SELECT * FROM tasks WHERE job_id=? AND status='running' AND dup_of IS NULL",
                                   [jid])
            for r in running:
                if self.store.one("SELECT 1 FROM tasks WHERE dup_of=? AND status IN ('running','queued')",
                                  [r["id"]]):
                    continue
                src = self.boxes.get(r["box_id"])
                if not src:
                    continue
                units = (r["hi"] - r["lo"] + 1) if r["lo"] is not None else 1
                expect = self._secs_per_unit(spec, src.gpu) * units
                if r["unit"] and spec.unit_costs.get(r["unit"]):
                    expect *= float(spec.unit_costs[r["unit"]]) / spec.unit_cost_s
                elapsed = time.time() - (r["started_at"] or time.time())
                if elapsed < max(1.5 * expect, float(self.cfg.get("speculate_min_s", 60))):
                    continue
                for box in self.boxes.values():
                    if (box.id == src.id or box.status != "ready" or box.busy() or not self._fits(spec, box)
                            or box.prepared.get(jid) != "ok"):
                        continue
                    if perf.info(box.gpu)["score"] < perf.info(src.gpu)["score"] * 0.9:
                        continue
                    tid = _id("t")
                    self.store.x("INSERT INTO tasks (id, job_id, unit, lo, hi, status, est_s, dup_of, attempt) "
                                 "VALUES (?,?,?,?,?,?,?,?,?)",
                                 [tid, jid, r["unit"], r["lo"], r["hi"], "queued", r["est_s"], r["id"], r["attempt"]])
                    task = dict(self.store.one("SELECT * FROM tasks WHERE id=?", [tid]))
                    self.store.event("speculate", tid, of=r["id"], box=box.id)
                    self._start(box, box.free_lanes()[0], jid, task, dup_of=r["id"])
                    break

    # =============================================================================================== agent events
    async def _read_agent(self, box: BoxRT) -> None:
        assert box.proc and box.proc.stdout
        try:
            while True:
                line = await box.proc.stdout.readline()
                if not line:
                    break
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                try:
                    await self._on_event(box, ev)
                except Exception:
                    log.exception("handling agent event %s", ev.get("ev"))
        finally:
            if box.status in ("ready", "draining") and not self._stop.is_set():
                log.warning("agent channel to %s closed", box.id)
                asyncio.create_task(self._reconnect(box))

    async def _on_event(self, box: BoxRT, ev: dict) -> None:
        kind = ev.get("ev")
        box.last_seen = time.time()
        if kind == "tel":
            gpus = ev.get("gpus") or []
            if gpus:
                util = sum(g["util"] for g in gpus) / len(gpus)
                mem = sum(g["mem"] for g in gpus)
                tot = sum(g["mem_total"] for g in gpus)
                box.tel.append((time.time(), util, mem, tot))
                box.tel = box.tel[-120:]
            box.ncpu = ev.get("ncpu") or box.ncpu
        elif kind == "exit":
            await self._on_exit(box, ev)
        elif kind == "error":
            log.warning("agent on %s: %s", box.id, ev)

    async def _on_exit(self, box: BoxRT, ev: dict) -> None:
        tid = ev["task"]
        self._wake.set()
        r = self.store.one("SELECT * FROM tasks WHERE id=?", [tid])
        for l in box.lanes:
            if l.task == tid:
                l.task = None
        box.idle_since = time.time() if not box.busy() else box.idle_since
        if r is None or r["status"] in ("done", "cancelled"):
            self._send(box, {"op": "ack", "task": tid})
            return
        jid = r["job_id"]
        spec = self._spec(jid)
        rc = int(ev.get("rc", 1))
        secs = float(ev.get("secs") or 0)
        task = dict(r)
        if rc == 0:
            self.store.update("tasks", tid, status="pulling", rc=0, tail=ev.get("tail", "")[-4000:])
            try:
                await self._pull_outputs(box, spec, task)
            except TransportError as e:
                # The task finished but its outputs did not come back: treat the box as faulty. It drains, and
                # destroying it requeues the task elsewhere (a rerun is cheaper than trusting a flaky link).
                log.warning("pull for %s failed: %s", tid, e)
                self.store.update("tasks", tid, status="running")
                self._box_fault(box, f"output pull failed: {e}")
                return
            now = time.time()
            winner = self.store.one("SELECT status FROM tasks WHERE id=?", [tid])["status"] == "pulling"
            self.store.update("tasks", tid, status="done", finished_at=now)
            self._send(box, {"op": "ack", "task": tid})
            units = (r["hi"] - r["lo"] + 1) if r["lo"] is not None else 1
            weight = (float(spec.unit_costs.get(r["unit"], spec.unit_cost_s)) / spec.unit_cost_s) if r["unit"] else 1
            if secs > 0 and winner:
                self.store.learn(perf.canonical(box.gpu), spec.workload_key, secs / max(units * weight, 1e-6))
            cost = box.price_h * secs / 3600 / max(len(box.lanes), 1)
            self.store.x("UPDATE jobs SET cost=cost+? WHERE id=?", [0 if box.prepaid else cost, jid])
            self.store.mark_machine(box.provider, box.machine_id, box.host_id, "ok")
            # A speculative twin of this task (or the original, if this was the twin) is now redundant.
            twin_ids = [x["id"] for x in self.store.q(
                "SELECT id FROM tasks WHERE (dup_of=? OR id=?) AND id!=? AND status IN ('running','queued')",
                [r["dup_of"] or tid, r["dup_of"] or "", tid])]
            for other in twin_ids:
                if other == r["dup_of"]:
                    # The copy won: the original's unit is done; stop it where it still runs.
                    orow = self.store.one("SELECT box_id FROM tasks WHERE id=?", [other])
                    self.store.update("tasks", other, status="done", finished_at=now, rc=0, tail="won by " + tid)
                    ob = self.boxes.get(orow["box_id"] or "") if orow else None
                    if ob:
                        self._send(ob, {"op": "cancel", "task": other})
                else:
                    self._cancel_task(other)
            self._notify(jid, ev="done", task=tid, unit=r["unit"], lo=r["lo"], hi=r["hi"], secs=secs, box=box.id)
        else:
            lost = bool(ev.get("lost"))
            if rc not in (-15, 143) and spec.checkpoint:
                await self._pull_checkpoint(box, spec, task, timeout=300)  # keep what it finished before failing
            self._send(box, {"op": "ack", "task": tid})
            excluded = json.loads(r["excluded"])
            machine_fault = rc in MACHINE_FAULT_RCS or lost
            if machine_fault:
                excluded.append(box.machine_id or box.id)
                self.store.mark_machine(box.provider, box.machine_id, box.host_id, "task_crash", f"rc={rc}")
            # A plain non-zero exit is the job's own failure: retrying on the same warm box is the fastest retry,
            # and excluding it could deadlock a one-box fleet.
            attempts = r["attempt"]
            if r["dup_of"]:
                self.store.update("tasks", tid, status="failed", rc=rc, finished_at=time.time())
            elif attempts < spec.max_attempts:
                self.store.update("tasks", tid, status="queued", rc=rc, box_id=None, tail=ev.get("tail", "")[-4000:],
                                  excluded=json.dumps(excluded))
            else:
                self.store.update("tasks", tid, status="failed", rc=rc, finished_at=time.time(),
                                  tail=ev.get("tail", "")[-4000:])
            self._notify(jid, ev="failed", task=tid, unit=r["unit"], lo=r["lo"], hi=r["hi"], rc=rc,
                         attempt=attempts, tail=ev.get("tail", "")[-1500:], box=box.id)
            if machine_fault:
                self._box_fault(box, f"task {tid} rc={rc}{' (lost)' if lost else ''}")
        self._finish_jobs()

    def _cancel_task(self, tid: str) -> None:
        r = self.store.one("SELECT * FROM tasks WHERE id=?", [tid])
        if not r:
            return
        self.store.update("tasks", tid, status="cancelled", finished_at=time.time())
        box = self.boxes.get(r["box_id"] or "")
        if box:
            self._send(box, {"op": "cancel", "task": tid})

    def _finish_jobs(self) -> None:
        for jid in self.active_jobs():
            if self.has_queued(jid):
                continue
            live = self.store.one("SELECT COUNT(*) AS n FROM tasks WHERE job_id=? AND status IN "
                                  "('running','pulling','queued')", [jid])["n"]
            if live:
                continue
            failed = self.store.q("SELECT id FROM tasks WHERE job_id=? AND status='failed' AND dup_of IS NULL AND "
                                  "NOT EXISTS (SELECT 1 FROM tasks t2 WHERE t2.dup_of=tasks.id AND t2.status='done')",
                                  [jid])
            status = "failed" if failed else "done"
            self.store.x("UPDATE jobs SET status=?, finished_at=? WHERE id=?", [status, time.time(), jid])
            cost = self.store.one("SELECT cost FROM jobs WHERE id=?", [jid])["cost"]
            self.store.event("job_" + status, jid, cost=cost)
            self._notify(jid, ev="job_" + status, cost=round(cost, 4), failed=[r["id"] for r in failed])

    def cancel_job(self, jid: str) -> None:
        self.store.x("UPDATE jobs SET status='cancelled', finished_at=? WHERE id=?", [time.time(), jid])
        for r in self.store.q("SELECT id FROM tasks WHERE job_id=? AND status IN ('queued','running')", [jid]):
            self._cancel_task(r["id"])
        self._notify(jid, ev="job_cancelled")

    # ================================================================================================= boxes
    def _box_fault(self, box: BoxRT, why: str) -> None:
        """A box misbehaved: stop giving it work and destroy it once its lanes are free (or now, if it is gone)."""
        self.store.event("box_fault", box.id, why=why)
        self.store.mark_machine(box.provider, box.machine_id, box.host_id, "box_fault", why[:200])
        if box.status == "ready":
            box.status = "draining"
            self.store.update("boxes", box.id, status="draining", note=why[:300])

    def _requeue_box_tasks(self, box: BoxRT, why: str) -> None:
        for r in self.store.q("SELECT * FROM tasks WHERE box_id=? AND status IN ('running','pulling')", [box.id]):
            if r["dup_of"]:
                self.store.update("tasks", r["id"], status="failed", finished_at=time.time())
                continue
            excl = json.loads(r["excluded"]) + [box.id]
            self.store.update("tasks", r["id"], status="queued", box_id=None, excluded=json.dumps(excl))
            self._notify(r["job_id"], ev="requeued", task=r["id"], why=why)
        for l in box.lanes:
            l.task = None

    async def _launch(self, offer: Offer, need: Need) -> None:
        """Rent, boot, bootstrap and connect one box; on failure try the next offer (other machine) a few times."""
        prov = self.providers[offer.provider]
        prof = profiles.get(need.profile, self.cfg.get("profiles"))
        tried: set[str] = set()
        o: Offer | None = offer
        boots = stale = 0
        # Two budgets: offers the provider rejects at create time (stale listings, common on Vast) are cheap to
        # skip; boxes that were created and then failed to boot cost real time and money.
        while boots < int(self.cfg["boot_retries"]) and stale < 8:
            if o is None:
                return
            bid = _id("b")
            label = f"{self.cfg['label_prefix']}{bid}"
            now = time.time()
            box = BoxRT(id=bid, provider=o.provider, gpu=o.gpu_name, num_gpus=o.num_gpus, price_h=o.price_h,
                        prepaid=o.prepaid, spot=o.spot, machine_id=o.machine_id, host_id=o.host_id,
                        profile=need.profile, units_h=float(o.raw.get("cu_per_h") or 0))
            # Durable record first: if we die mid-launch the reconciler finds the label and destroys the box.
            self.store.x("INSERT INTO boxes (id, provider, label, status, offer, profile, gpu_name, num_gpus, price_h, "
                         "prepaid, spot, machine_id, host_id, created_at, hard_deadline) "
                         "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         [bid, o.provider, label, "requested", json.dumps({**o.__dict__, "raw": o.raw}), need.profile,
                          o.gpu_name, o.num_gpus, o.price_h, int(o.prepaid), int(o.spot), o.machine_id, o.host_id,
                          now, now + float(self.cfg["hard_ttl_s"])])
            self.boxes[bid] = box
            t0 = time.time()
            try:
                box.ext_id = await prov.launch(o, label, prof, float(self.cfg["hard_ttl_s"]))
                box.status = "booting"
                self.store.update("boxes", bid, ext_id=box.ext_id, status="booting")
                self.store.event("box_launch", bid, provider=o.provider, gpu=o.gpu_name, price_h=o.price_h,
                                 ext_id=box.ext_id)
                target = await prov.wait_ready(box.ext_id, t0 + float(self.cfg["boot_timeout_s"]))
                self.store.update("boxes", bid, ssh=json.dumps(target.__dict__))
                await self._connect(box, target, prov, fresh=True)
                box.status = "ready"
                box.ready_at = box.idle_since = time.time()
                self.store.update("boxes", bid, status="ready", ready_at=box.ready_at, gpu_name=box.gpu)
                self.store.x("INSERT INTO boot_stats VALUES (?,?,?,1)", [o.provider, time.time(), time.time() - t0])
                self.store.event("box_ready", bid, secs=round(time.time() - t0, 1), gpu=box.gpu, driver=box.driver)
                self._wake.set()
                return
            except (ProviderError, TransportError, OSError) as e:
                log.warning("launch %s (%s %s) failed: %s", bid, o.provider, o.gpu_name, e)
                self.store.event("box_boot_failed", bid, error=str(e)[-400:])
                if box.ext_id:
                    boots += 1
                    self.store.x("INSERT INTO boot_stats VALUES (?,?,?,0)", [o.provider, time.time(), time.time() - t0])
                    self.store.mark_machine(o.provider, o.machine_id, o.host_id, "boot_fail", str(e)[:200])
                else:
                    stale += 1
                await self._destroy(box, "boot failed")
                if isinstance(e, ProviderError) and not e.retry_other:
                    return
                tried.add(o.machine_id or o.offer_id)
                o = await self._next_offer(need, o.provider, tried)
        self.store.event("launch_gave_up", need.profile)

    async def _next_offer(self, need: Need, provider: str, tried: set[str]) -> Offer | None:
        bad_m, bad_h = self.store.bad_machines(provider)
        try:
            offers = await self.providers[provider].offers(need, bad_m | tried, bad_h)
        except ProviderError:
            return None
        offers = [o for o in offers if (o.machine_id or o.offer_id) not in tried]
        return min(offers, key=lambda o: o.price_h) if offers else None

    async def _connect(self, box: BoxRT, target: SshTarget, prov: Provider, fresh: bool,
                       restore: bool = False) -> None:
        """Open the transport, bootstrap (fresh boxes), restore caches, start the agent channel."""
        if self.local_boxes:
            box.transport = box.transport or LocalTransport(box.id, self.state_dir / "local-boxes" / box.id)
        else:
            box.transport = SshTransport(box.id, target, self.state_dir, prov.ssh_options())
        await box.transport.open(timeout=240)
        prof = profiles.get(box.profile, self.cfg.get("profiles"))
        state = "/var/lib/gpufleet" if not self.local_boxes else f"{box.transport.root}/../state"
        agent_path = "/opt/gpufleet/agent.py" if not self.local_boxes else f"{box.transport.root}/../agent.py"
        if fresh:
            if not self.local_boxes:
                rc, out = await box.transport.exec(
                    "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1", 60)
                if rc == 0 and "," in out:
                    name, drv = [x.strip() for x in out.strip().split(",", 1)]
                    box.gpu, box.driver = name.replace("NVIDIA ", "") or box.gpu, drv
                await self._restore_caches(box, prof)  # before bootstrap: its GPU self-test then hits warm caches
            rc, out = await box.transport.exec("bash -s", timeout=900, stdin=prof["bootstrap"].encode()) \
                if not self.local_boxes else (0, "GF_OK local")
            for line in out.splitlines():
                if line.startswith("GF_GPU "):
                    parts = [x.strip() for x in line[7:].split(",")]
                    box.gpu = parts[0].replace("NVIDIA ", "") or box.gpu
                    box.driver = parts[1] if len(parts) > 1 else ""
                elif line.startswith("GF_CPU "):
                    box.ncpu = int(line[7:] or 0)
            if rc or "GF_FAIL" in out:
                raise TransportError(f"bootstrap failed rc={rc}: {out[-600:]}", rc or 1)
            await box.transport.exec(f"mkdir -p {os.path.dirname(agent_path)}", 60)
            await box.transport.push(Sync(src=str(AGENT_SRC), dst=agent_path))
            st = next((l for l in out.splitlines() if l.startswith("GF_SELFTEST_OK")), "")
            if st:
                self.store.event("selftest", box.id, result=st[15:])
        n = max(1, round(box.num_gpus / max(self._min_share(box), 0.01)))
        if not box.lanes:
            box.lanes = [Lane() for _ in range(n)]
        box.proc = await box.transport.spawn(
            f"python3 {agent_path} serve --root {box.transport.root} --state {state}")
        hello = await asyncio.wait_for(box.proc.stdout.readline(), 60)
        h = json.loads(hello) if hello else {}
        if h.get("ev") != "hello":
            raise TransportError(f"agent did not start: {hello[:200]!r}", 1)
        known = set(h.get("known") or [])
        if not fresh:
            # Tasks the database believes run here but the agent has never heard of (lost before they started,
            # e.g. the daemon died between lease and send): requeue them.
            for r in self.store.q("SELECT id, job_id FROM tasks WHERE box_id=? AND status IN ('running','pulling')",
                                  [box.id]):
                if r["id"] not in known:
                    self.store.update("tasks", r["id"], status="queued", box_id=None)
                    self._notify(r["job_id"], ev="requeued", task=r["id"], why="unknown to agent after reconnect")
            busy = [t for t in h.get("running") or [] if t in known]
            for i, t in enumerate(busy):
                if i < len(box.lanes):
                    box.lanes[i] = Lane(task=t, since=time.time())
        box.last_seen = time.time()
        box.reader = asyncio.create_task(self._read_agent(box))

    def _min_share(self, box: BoxRT) -> float:
        """Initial lanes per GPU come from the smallest gpu_share among jobs that could use this box."""
        shares = [self._spec(j).gpu_share for j in self.active_jobs() if self._spec(j).need.profile == box.profile]
        return min(shares) if shares else 1.0

    async def _reconnect(self, box: BoxRT) -> None:
        """The agent channel dropped. Running tasks keep going on the box; reconnect and let the agent re-report.
        If the box is unreachable, requeue its tasks and destroy it."""
        if box.status not in ("ready", "draining") or box.lost:
            return
        prov = self.providers[box.provider]
        row = self.store.one("SELECT ssh FROM boxes WHERE id=?", [box.id])
        for attempt in range(3):
            try:
                await asyncio.sleep(3 * (attempt + 1))
                if isinstance(box.transport, SshTransport):
                    await box.transport.close()
                target = SshTarget(**json.loads(row["ssh"])) if row and row["ssh"] else SshTarget(host="")
                await self._connect(box, target, prov, fresh=False)
                self.store.event("box_reconnected", box.id)
                return
            except Exception as e:
                log.warning("reconnect %s attempt %d: %s", box.id, attempt + 1, e)
        box.lost = True
        self._requeue_box_tasks(box, "box unreachable")
        await self._destroy(box, "unreachable")

    async def _save_caches(self, box: BoxRT) -> None:
        prof = profiles.get(box.profile, self.cfg.get("profiles"))
        if not prof.get("caches") or not box.driver or not box.transport or self.local_boxes:
            return
        key = self._cache_key(box)
        for c in prof["caches"]:
            dst = self.state_dir / "caches" / key / c.strip("/").replace("/", "_")
            try:
                dst.mkdir(parents=True, exist_ok=True)
                await box.transport.pull(Sync(src=c.rstrip("/") + "/", dst=str(dst) + "/"), timeout=300)
            except TransportError as e:
                log.info("cache save %s: %s", c, e)

    async def _restore_caches(self, box: BoxRT, prof: dict) -> None:
        if not prof.get("caches") or not box.driver:
            return
        key = self._cache_key(box)
        for c in prof["caches"]:
            src = self.state_dir / "caches" / key / c.strip("/").replace("/", "_")
            await box.transport.exec(f"mkdir -p {shlex.quote(c)}", 30)
            if src.exists() and any(src.iterdir()):
                try:
                    await box.transport.push(Sync(src=str(src) + "/", dst=c.rstrip("/") + "/"), timeout=300)
                    self.store.event("cache_restored", box.id, cache=c, key=key)
                except TransportError as e:
                    log.info("cache restore %s: %s", c, e)

    def _cache_key(self, box: BoxRT) -> str:
        return f"{box.profile}--{perf.canonical(box.gpu).replace(' ', '_')}--{box.driver}"

    async def _destroy(self, box: BoxRT, why: str) -> None:
        if box.status in ("destroying", "destroyed"):
            return
        prev = box.status
        box.status = "destroying"
        self.store.update("boxes", box.id, status="destroying", note=why[:300])
        if prev in ("ready", "draining") and not box.lost:
            for r in self.store.q("SELECT * FROM tasks WHERE box_id=? AND status='running'", [box.id]):
                spec = self._spec(r["job_id"])
                if spec.checkpoint:
                    try:
                        await asyncio.wait_for(self._pull_checkpoint(box, spec, dict(r), timeout=240), 260)
                    except Exception:
                        pass
            try:
                await asyncio.wait_for(self._save_caches(box), 120)
            except Exception:
                pass
        self._requeue_box_tasks(box, f"box destroyed: {why}")
        if box.reader:
            box.reader.cancel()
        if box.proc and box.proc.returncode is None:
            box.proc.kill()
        if box.transport:
            try:
                await box.transport.close()
            except Exception:
                pass
        if box.ext_id:
            try:
                await self.providers[box.provider].destroy(box.ext_id)
            except ProviderError as e:
                # Leave it 'destroying': the reconciler retries every cycle until the provider confirms.
                log.error("DESTROY FAILED for %s (%s): %s", box.id, box.ext_id, e)
                self.store.event("destroy_failed", box.id, error=str(e)[-300:])
                return
        box.status = "destroyed"
        self.store.update("boxes", box.id, status="destroyed", destroyed_at=time.time())
        self.store.event("box_destroyed", box.id, why=why)
        self.boxes.pop(box.id, None)

    # ================================================================================================= scaling
    async def _budget_ok(self, offer: Offer) -> tuple[bool, str]:
        if offer.prepaid:
            return True, ""
        spent = sum(self.store.spend_today(p) for p, prov in self.providers.items() if not prov.prepaid)
        burn = sum(b.price_h for b in self.boxes.values() if not b.prepaid and b.status != "destroyed")
        cap = float(self.cfg["daily_cap_usd"])
        if spent + (burn + offer.price_h) * 0.25 > cap:  # leave room for 15 more minutes of the whole fleet
            return False, f"daily cap ${cap:.2f}: spent ${spent:.2f}, fleet burning ${burn:.2f}/h"
        return True, ""

    def _can_take(self, jid: str, box: BoxRT) -> bool:
        """Could this box run some queued work of the job (fits, and not excluded for every queued task)?"""
        if self._range_left(jid) > 0:
            return True
        for r in self.store.q("SELECT excluded FROM tasks WHERE job_id=? AND status='queued'", [jid]):
            ex = json.loads(r["excluded"])
            if box.id not in ex and (not box.machine_id or box.machine_id not in ex):
                return True
        return False

    def _capacity_for(self, jid: str, spec: JobSpec) -> tuple[int, int]:
        """(free lanes now, lanes booting) that could serve this job's queued work."""
        free = booting = 0
        for b in self.boxes.values():
            if b.status == "ready" and self._fits(spec, b) and self._can_take(jid, b):
                free += len(b.free_lanes())
            elif b.status in ("requested", "booting") and b.profile == spec.need.profile:
                booting += max(1, round(b.num_gpus / max(spec.gpu_share, 0.01)))
        return free, booting

    async def scale(self) -> None:
        alive = [b for b in self.boxes.values() if b.status not in ("destroying", "destroyed", "parked")]
        if (len(alive) >= int(self.cfg["max_boxes"])
                or len(self.launching) >= int(self.cfg.get("max_parallel_boots", 3))):
            return
        # Warm pool: keep N ready boxes of a profile for a while (seconds-latency for the next job).
        for prof, w in list(self.warm.items()):
            want = self._warm_need(prof)
            have = sum(1 for b in alive if b.profile == prof and b.status in ("requested", "booting", "ready",
                                                                             "unparking"))
            if have < want:
                if await self._try_unpark(prof, w["need"]):
                    return
                best = await self._best_offer(JobSpec(project="warm", name=prof, command="", units=("x",),
                                                      need=w["need"]), 600)
                if best and (await self._budget_ok(best))[0]:
                    await self._spawn_launch(best, w["need"])
                    return
        for jid in self.active_jobs():
            spec = self._spec(jid)
            if not self.has_queued(jid):
                continue
            free, booting = self._capacity_for(jid, spec)
            if free:
                continue
            serving = [b for b in alive if b.profile == spec.need.profile and b.status in ("ready", "draining")
                       and self._can_take(jid, b)]
            if serving and self.oldest_wait(jid) < float(self.cfg["scale_delay_s"]):
                continue
            pending = self.pending_secs(jid)
            if booting and pending < 600 * booting:
                continue  # the boxes already on their way are enough for what is queued
            if await self._try_unpark(spec.need.profile, spec.need, spec):
                self._notify(jid, ev="scaling", provider="parked", gpu="")
                return
            if self.store.one("SELECT cost FROM jobs WHERE id=?", [jid])["cost"] >= spec.budget_usd:
                self._notify(jid, ev="budget_reached", budget=spec.budget_usd)
                continue
            best = await self._best_offer(spec, pending)
            if not best:
                continue
            ok, why = await self._budget_ok(best)
            if not ok:
                self._notify(jid, ev="blocked", why=why)
                self.store.event("scale_blocked", jid, why=why)
                continue
            await self._spawn_launch(best, spec.need)
            self._notify(jid, ev="scaling", provider=best.provider, gpu=best.gpu_name, price_h=round(best.price_h, 3))
            return  # one launch per tick; the next tick sees the new box as booting capacity

    async def _spawn_launch(self, offer: Offer, need: Need) -> None:
        key = f"{offer.provider}:{offer.offer_id}"
        self.launching.add(key)

        async def go():
            try:
                await self._launch(offer, need)
            finally:
                self.launching.discard(key)

        asyncio.create_task(go())

    async def _try_unpark(self, profile: str, need: Need, spec: JobSpec | None = None) -> bool:
        for b in self.boxes.values():
            if b.status != "parked" or b.profile != profile:
                continue
            if need.providers and b.provider not in need.providers:
                continue
            if spec is not None and not self._fits(spec, b):
                continue
            key = f"unpark:{b.id}"
            self.launching.add(key)

            async def go(box=b, k=key):
                try:
                    await self._unpark(box)
                finally:
                    self.launching.discard(k)

            asyncio.create_task(go())
            return True
        return False

    def set_warm(self, profile: str, boxes: int, minutes: float, need: dict | None = None) -> dict:
        n = Need.from_dict({**(need or {}), "profile": profile})
        self.warm[profile] = {"boxes": boxes, "until": time.time() + minutes * 60, "need": n}
        self._wake.set()
        return {"profile": profile, "boxes": boxes, "until": self.warm[profile]["until"]}

    async def _best_offer(self, spec: JobSpec, pending_ref_s: float) -> Offer | None:
        """Lowest expected cost of getting the queued work done on one new box of this offer:
            price x (boot + work_on_it) / boot_success_rate + value_of_time x (boot + work_on_it)
        where work_on_it is capped at an hour (the box will not do everything alone if the queue is long)."""
        vot = float(self.cfg["value_of_time_usd_per_h"])
        cands: list[tuple[float, Offer]] = []
        for name, prov in self.providers.items():
            if spec.need.providers and name not in spec.need.providers:
                continue
            mine = [b for b in self.boxes.values() if b.provider == name and b.status != "destroyed"]
            if len(mine) >= prov.max_boxes:
                continue
            try:
                ok, why = await prov.can_launch()
                if not ok:
                    self.store.event("provider_blocked", name, why=why)
                    continue
                bad_m, bad_h = self.store.bad_machines(name)
                offers = await prov.offers(spec.need, bad_m, bad_h)
            except ProviderError as e:
                log.warning("offers from %s: %s", name, e)
                continue
            boot_med, rate = self.store.boot_estimate(name, 240)
            good = {r["machine_id"] for r in self.store.q(
                "SELECT DISTINCT machine_id FROM machines WHERE provider=? AND kind='ok' AND at>?",
                [name, time.time() - 14 * 86400])}
            for o in offers:
                if o.price_h > spec.need.max_price_h:
                    continue
                if o.machine_id in good:
                    o = Offer(**{**o.__dict__, "boot_s": min(boot_med, 75)})  # image layers likely cached there
                spu = self._secs_per_unit(spec, o.gpu_name) / spec.unit_cost_s  # box-seconds per reference-second
                work = min(pending_ref_s * spu, 3600)
                boot = o.boot_s if (o.prepaid or o.machine_id in good) else boot_med
                hours = (boot + work) / 3600
                cost = o.price_h * hours / max(rate, 0.3) + vot * hours
                cands.append((cost, o))
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        return cands[0][1]

    def lanes(self) -> None:
        """Add a lane when every lane is busy, the GPU is under-used and there is VRAM to spare."""
        if not self.cfg.get("lane_autoscale", True):
            return
        now = time.time()
        for b in self.boxes.values():
            if b.status != "ready" or not b.lanes:
                continue
            if b.busy() < len(b.lanes):
                b.full_since = 0
                continue
            b.full_since = b.full_since or now
            if now - b.full_since < 60 or len(b.lanes) >= b.num_gpus * int(self.cfg["max_lanes_per_gpu"]):
                continue
            recent = [t for t in b.tel if t[0] > now - 45]
            if len(recent) < 6:
                continue
            util = sum(t[1] for t in recent) / len(recent)
            mem = max(t[2] for t in recent)
            tot = recent[-1][3]
            per_lane = mem / len(b.lanes)
            if util < 80 and tot - mem > 1.4 * per_lane + 1024:
                b.lanes.append(Lane())
                b.full_since = 0
                self.store.event("lane_added", b.id, lanes=len(b.lanes), util=round(util), mem=mem, total=tot)

    def _warm_need(self, profile: str) -> int:
        w = self.warm.get(profile)
        if not w or time.time() > w["until"]:
            self.warm.pop(profile, None)
            return 0
        return int(w["boxes"])

    async def _park(self, b: BoxRT) -> bool:
        """Idle box -> stopped box that keeps its disk (image, inputs, caches) and restarts in seconds."""
        prov = self.providers[b.provider]
        parked = sum(1 for x in self.boxes.values() if x.status == "parked")
        if (not prov.supports_park or b.spot or parked >= int(self.cfg.get("park_max", 2))
                or time.time() - b.created > float(self.cfg["reuse_max_age_s"]) * 0.75):
            return False
        try:
            await asyncio.wait_for(self._save_caches(b), 120)
        except Exception:
            pass
        if b.proc and b.proc.returncode is None:
            b.proc.kill()
        if b.reader:
            b.reader.cancel()
        if b.transport:
            await b.transport.close()
        try:
            await prov.park(b.ext_id)
        except ProviderError as e:
            log.info("park %s failed: %s", b.id, e)
            return False
        b.status = "parked"
        b.lanes = []
        self.store.update("boxes", b.id, status="parked", note="parked idle",
                          hard_deadline=time.time() + float(self.cfg.get("park_ttl_s", 86400)))
        self.store.event("box_parked", b.id)
        return True

    async def _unpark(self, b: BoxRT) -> None:
        prov = self.providers[b.provider]
        b.status = "unparking"
        self.store.update("boxes", b.id, status="unparking")
        t0 = time.time()
        try:
            target = await prov.unpark(b.ext_id, t0 + float(self.cfg.get("unpark_timeout_s", 120)))
            self.store.update("boxes", b.id, ssh=json.dumps(target.__dict__))
            b.transport = None if not self.local_boxes else b.transport
            await self._connect(b, target, prov, fresh=False, restore=True)
            b.status = "ready"
            b.created = time.time()  # age counts from the restart: the contract and host are what they were
            b.idle_since = time.time()
            self.store.update("boxes", b.id, status="ready", hard_deadline=time.time() + float(self.cfg["hard_ttl_s"]))
            self.store.event("box_unparked", b.id, secs=round(time.time() - t0, 1))
            self._wake.set()
        except Exception as e:
            log.info("unpark %s failed: %s", b.id, e)
            self.store.event("unpark_failed", b.id, error=str(e)[-300:])
            b.lost = True
            await self._destroy(b, "unpark failed")

    async def lifecycle(self) -> None:
        now = time.time()
        queued_any = any(self.has_queued(j) for j in self.active_jobs())
        for b in list(self.boxes.values()):
            row = self.store.one("SELECT hard_deadline FROM boxes WHERE id=?", [b.id])
            if row and now > row["hard_deadline"]:
                b.lost = b.lost or b.status == "parked"
                await self._destroy(b, "hard deadline" if b.status != "parked" else "parked too long")
                continue
            if b.status == "ready" and now - b.created > float(self.cfg["reuse_max_age_s"]):
                b.status = "draining"
                self.store.update("boxes", b.id, status="draining", note="age")
            if b.status == "draining" and not b.busy():
                await self._destroy(b, "drained")
                continue
            if b.status == "ready" and not b.busy():
                usable = queued_any and any(self._fits(self._spec(j), b) and self.has_queued(j)
                                            for j in self.active_jobs())
                ready_same = sum(1 for x in self.boxes.values() if x.profile == b.profile and x.status == "ready")
                if usable or ready_same <= self._warm_need(b.profile):
                    continue
                if now - b.idle_since > float(self.cfg["idle_s"]):
                    if not await self._park(b):
                        await self._destroy(b, "idle")

    def accrue(self) -> None:
        now = time.time()
        dt = now - self._last_accrue
        self._last_accrue = now
        for b in self.boxes.values():
            if b.status == "parked":
                self.store.add_spend(b.provider, 0.012 * dt / 3600)
                continue
            if b.status in ("booting", "ready", "draining", "destroying", "unparking") and b.ext_id:
                if b.prepaid:
                    self.store.add_spend(b.provider, 0, b.units_h * dt / 3600)
                else:
                    self.store.add_spend(b.provider, b.price_h * dt / 3600)

    # ============================================================================================== reconciler
    async def reconcile(self) -> dict:
        """Provider truth vs our database. Destroys our orphans, terminal boxes and boxes past their deadline;
        never touches instances without our label prefix."""
        report = {"orphans": [], "gone": [], "foreign": 0, "retried": []}
        prefix = self.cfg["label_prefix"]
        for name, prov in self.providers.items():
            try:
                remote = await prov.list_all()
            except Exception as e:
                log.warning("reconcile %s: %s", name, e)
                continue
            known = {b["ext_id"]: b for b in self.store.alive_boxes(name) if b["ext_id"]}
            seen = set()
            for rb in remote:
                if not rb.label.startswith(prefix):
                    report["foreign"] += 1
                    continue
                seen.add(rb.ext_id)
                row = known.get(rb.ext_id)
                if row is None or rb.terminal or time.time() > (row["hard_deadline"] if row else 0):
                    if row is not None and row["id"] in self.boxes:
                        await self._destroy(self.boxes[row["id"]], "reconcile: terminal or past deadline")
                        continue
                    try:
                        await prov.destroy(rb.ext_id)
                        report["orphans"].append(rb.ext_id)
                        self.store.event("orphan_destroyed", rb.ext_id, provider=name, label=rb.label)
                        if row is not None:
                            self.store.update("boxes", row["id"], status="destroyed", destroyed_at=time.time())
                    except ProviderError as e:
                        log.error("orphan %s destroy failed: %s", rb.ext_id, e)
            for ext, row in known.items():
                if ext in seen:
                    continue
                if row["status"] == "destroying":
                    report["retried"].append(ext)  # provider confirms it is gone: finish the bookkeeping
                if time.time() - row["created_at"] < 120 and row["status"] == "requested":
                    continue
                box = self.boxes.get(row["id"])
                if box:
                    box.lost = True
                    self._requeue_box_tasks(box, "provider no longer lists the box")
                    box.status = "destroyed"
                    self.boxes.pop(box.id, None)
                self.store.update("boxes", row["id"], status="destroyed", destroyed_at=time.time(),
                                  note="vanished at provider")
                report["gone"].append(ext)
        self._last_reconcile = time.time()
        return report

    async def recover(self) -> None:
        """Daemon start: reattach to boxes that were ready, destroy anything that was mid-launch."""
        for row in self.store.alive_boxes():
            prov = self.providers.get(row["provider"])
            if prov is None:
                continue
            if row["status"] == "parked":
                box = BoxRT(id=row["id"], provider=row["provider"], ext_id=row["ext_id"] or "",
                            gpu=row["gpu_name"] or "", num_gpus=row["num_gpus"] or 1, price_h=row["price_h"] or 0,
                            machine_id=row["machine_id"] or "", host_id=row["host_id"] or "", status="parked",
                            created=row["created_at"], profile=row["profile"] or "base")
                self.boxes[box.id] = box
                continue
            if row["status"] in ("ready", "draining") and row["ssh"]:
                box = BoxRT(id=row["id"], provider=row["provider"], ext_id=row["ext_id"] or "",
                            gpu=row["gpu_name"] or "", num_gpus=row["num_gpus"] or 1, price_h=row["price_h"] or 0,
                            prepaid=bool(row["prepaid"]), spot=bool(row["spot"]), machine_id=row["machine_id"] or "",
                            host_id=row["host_id"] or "", status=row["status"], created=row["created_at"])
                box.profile = row["profile"] or box.profile
                self.boxes[box.id] = box
                try:
                    await self._connect(box, SshTarget(**json.loads(row["ssh"])), prov, fresh=False)
                    self.store.event("box_recovered", box.id)
                    # Tasks the database thinks are running there and the agent does not know: requeue.
                    continue
                except Exception as e:
                    log.warning("recover %s failed: %s", row["id"], e)
                    await self._destroy(box, "recover failed")
            else:
                box = BoxRT(id=row["id"], provider=row["provider"], ext_id=row["ext_id"] or "",
                            status=row["status"])
                self.boxes[box.id] = box
                await self._destroy(box, "daemon restarted mid-launch")
        for r in self.store.q("SELECT id FROM tasks WHERE status IN ('running','pulling') AND (box_id IS NULL OR "
                              "box_id NOT IN (SELECT id FROM boxes WHERE status IN ('ready','draining')))"):
            self.store.update("tasks", r["id"], status="queued", box_id=None)
        await self.reconcile()

    # ================================================================================================ main loop
    def _notify(self, jid: str, **ev) -> None:
        ev["job"] = jid
        ev["t"] = round(time.time(), 2)
        for q in self.waiters.get(jid, []) + self.waiters.get("*", []):
            q.put_nowait(ev)

    async def tick(self) -> None:
        self.accrue()
        self.watchdogs()
        self.dispatch()
        self.speculate()
        self.checkpoints()
        self.lanes()
        await self.scale()
        await self.lifecycle()
        if time.time() - self._last_reconcile > float(self.cfg["reconcile_s"]):
            await self.reconcile()

    async def run(self) -> None:
        await self.recover()
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception:
                log.exception("tick failed")
            try:
                await asyncio.wait_for(self._wake.wait(), 1.0)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

    async def shutdown(self, destroy: bool = False) -> None:
        self._stop.set()
        for b in list(self.boxes.values()):
            if destroy:
                await self._destroy(b, "shutdown")
            elif b.transport and isinstance(b.transport, SshTransport):
                await b.transport.close()  # boxes keep running; the next daemon start reattaches
        for p in self.providers.values():
            await p.close()

    async def drain(self) -> None:
        for b in list(self.boxes.values()):
            if b.busy():
                b.status = "draining"
                self.store.update("boxes", b.id, status="draining", note="drain requested")
            else:
                await self._destroy(b, "drain requested")

    # ================================================================================================ queries
    def job_status(self, jid: str) -> dict:
        j = self.store.one("SELECT * FROM jobs WHERE id=?", [jid])
        if not j:
            return {"error": "no such job"}
        counts = {r["status"]: r["n"] for r in self.store.q(
            "SELECT status, COUNT(*) AS n FROM tasks WHERE job_id=? AND dup_of IS NULL GROUP BY status", [jid])}
        return {"id": jid, "project": j["project"], "name": j["name"], "status": j["status"],
                "cost": round(j["cost"], 4), "tasks": counts, "range_left": self._range_left(jid),
                "created": j["created_at"], "finished": j["finished_at"]}

    def fleet_status(self) -> dict:
        boxes = []
        for b in self.boxes.values():
            util = round(sum(t[1] for t in b.tel[-6:]) / max(len(b.tel[-6:]), 1)) if b.tel else None
            boxes.append({"id": b.id, "provider": b.provider, "gpu": b.gpu, "status": b.status,
                          "price_h": round(b.price_h, 3), "lanes": len(b.lanes), "busy": b.busy(),
                          "age_min": round((time.time() - b.created) / 60, 1), "util": util})
        jobs = [self.job_status(j) for j in self.active_jobs()]
        spend = {r["provider"]: {"usd": round(r["usd"], 3), "units": round(r["units"], 2)} for r in self.store.q(
            "SELECT provider, usd, units FROM spend WHERE day=?", [self.store.today()])}
        return {"boxes": boxes, "jobs": jobs, "spend_today": spend, "daily_cap_usd": self.cfg["daily_cap_usd"]}
