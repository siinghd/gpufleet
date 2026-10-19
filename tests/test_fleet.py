"""End-to-end scheduler tests on the fake provider: boxes are local directories, the real agent runs the tasks."""

import asyncio
import json
import time
from pathlib import Path

import pytest

from gpufleet import config
from gpufleet.daemon import Fleet
from gpufleet.models import JobSpec
from gpufleet.providers.fake import Fake
from gpufleet.store import Store


def make_cfg(tmp: Path, **over) -> dict:
    cfg = config._merge(config.DEFAULTS, {
        "db": str(tmp / "fleet.db"), "scale_delay_s": 0, "idle_s": 1, "reconcile_s": 3600, "boot_retries": 3,
        "lane_autoscale": False, "park_max": 0, **over})
    cfg["state_dir"] = str(tmp / "state")
    return cfg


async def run_until(fleet: Fleet, jid: str, timeout: float = 60) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        await fleet.tick()
        st = fleet.job_status(jid)
        if st["status"] in ("done", "failed", "cancelled"):
            return st
        await asyncio.sleep(0.1)
    raise AssertionError(f"job did not finish: {fleet.job_status(jid)}")


async def settle(fleet: Fleet, secs: float = 3):
    end = time.time() + secs
    while time.time() < end:
        await fleet.tick()
        await asyncio.sleep(0.1)


def spec(tmp: Path, **kw) -> JobSpec:
    base = {"project": "t", "name": "j", "command": "mkdir -p out && echo ${GF_UNIT:-$GF_LO-$GF_HI} > out/$GF_TASK.txt",
            "outputs": [{"src": "out/{task}.txt", "dst": str(tmp / "results") + "/{task}.txt"}],
            "unit_cost_s": 1, "need": {"rt_cores": True}}
    base.update(kw)
    return JobSpec.from_dict(base)


@pytest.fixture
def tmp(tmp_path):
    return tmp_path


async def test_named_units_run_and_boxes_are_released(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 4090", "price_h": 0.4}])
    fleet = Fleet(make_cfg(tmp, max_boxes=2), {"fake": prov}, local_boxes=True)
    jid = fleet.submit(spec(tmp, units=["a", "b", "c", "d"], unit_costs={"a": 5, "b": 1, "c": 3, "d": 2}))
    st = await run_until(fleet, jid)
    assert st["status"] == "done", st
    got = sorted(p.read_text().strip() for p in (tmp / "results").glob("*.txt"))
    assert got == ["a", "b", "c", "d"]
    await settle(fleet, 3)  # idle_s=1: boxes go away
    assert not prov.live, prov.live
    assert all(r["status"] == "destroyed" for r in fleet.store.q("SELECT status FROM boxes"))


async def test_range_job_covers_every_unit_exactly_once(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 5090", "price_h": 0.6}, {"gpu_name": "RTX 3090", "price_h": 0.2}])
    fleet = Fleet(make_cfg(tmp, max_boxes=3), {"fake": prov}, local_boxes=True)
    jid = fleet.submit(spec(tmp, range=[1, 97], unit_cost_s=0.5, chunk_target_s=4,
                            command="mkdir -p out && seq $GF_LO $GF_HI > out/$GF_TASK.txt"))
    st = await run_until(fleet, jid)
    assert st["status"] == "done", st
    nums = sorted(int(x) for p in (tmp / "results").glob("*.txt") for x in p.read_text().split())
    assert nums == list(range(1, 98))
    chunks = fleet.store.q("SELECT lo, hi FROM tasks WHERE job_id=? ORDER BY rowid", [jid])
    # The first chunk is sized from the prior (4 s / 0.5 s per unit = 8); later ones grow once real timings
    # (milliseconds here) are learned.
    assert len(chunks) > 1 and chunks[0]["hi"] - chunks[0]["lo"] + 1 <= 8
    await fleet.shutdown(destroy=True)
    assert not prov.live


async def test_failed_task_is_retried(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 4090", "price_h": 0.4}])
    fleet = Fleet(make_cfg(tmp, max_boxes=2), {"fake": prov}, local_boxes=True)
    # Fails on the first attempt only.
    cmd = 'mkdir -p out && [ "$GF_ATTEMPT" -ge 2 ] && echo $GF_UNIT > out/$GF_TASK.txt'
    jid = fleet.submit(spec(tmp, units=["x"], command=cmd))
    st = await run_until(fleet, jid)
    assert st["status"] == "done", st
    t = fleet.store.one("SELECT * FROM tasks WHERE job_id=?", [jid])
    assert t["attempt"] == 2
    await fleet.shutdown(destroy=True)


async def test_machine_fault_moves_task_to_another_machine(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 4090", "price_h": 0.4, "machine_id": 1},
                        {"gpu_name": "RTX 4090", "price_h": 0.5, "machine_id": 2}])
    fleet = Fleet(make_cfg(tmp, max_boxes=2), {"fake": prov}, local_boxes=True)
    # rc 3 = "no usable GPU": the machine is blamed and the retry must land on the other machine.
    cmd = 'mkdir -p out && [ "$GF_ATTEMPT" -ge 2 ] || exit 3; echo $GF_UNIT > out/$GF_TASK.txt'
    jid = fleet.submit(spec(tmp, units=["x"], command=cmd))
    st = await run_until(fleet, jid)
    assert st["status"] == "done", st
    t = fleet.store.one("SELECT * FROM tasks WHERE job_id=?", [jid])
    assert json.loads(t["excluded"]) == ["1"]  # the cheaper machine took it first and was blamed
    await fleet.shutdown(destroy=True)


async def test_job_fails_after_max_attempts(tmp):
    prov = Fake()
    fleet = Fleet(make_cfg(tmp, max_boxes=1), {"fake": prov}, local_boxes=True)
    jid = fleet.submit(spec(tmp, units=["x"], command="exit 7", max_attempts=2))
    st = await run_until(fleet, jid)
    assert st["status"] == "failed"
    await fleet.shutdown(destroy=True)
    assert not prov.live


async def test_boot_failure_tries_another_machine(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 4090", "price_h": 0.4, "machine_id": 1},
                        {"gpu_name": "RTX 4090", "price_h": 0.5, "machine_id": 2}], fail_launches=1, fail_boots=1)
    fleet = Fleet(make_cfg(tmp), {"fake": prov}, local_boxes=True)
    jid = fleet.submit(spec(tmp, units=["a"]))
    st = await run_until(fleet, jid)
    assert st["status"] == "done"
    # The stale offer (rejected at create) is skipped without blame; the box that was created and never came up
    # is destroyed and its machine recorded.
    assert fleet.store.one("SELECT COUNT(*) AS n FROM machines WHERE kind='boot_fail'")["n"] == 1
    assert len(prov.destroyed) >= 1 and prov.destroyed[0] == prov.launched[0]
    await fleet.shutdown(destroy=True)


async def test_reconciler_kills_our_orphans_and_spares_foreign_boxes(tmp):
    from gpufleet.models import RemoteBox
    prov = Fake()
    prov.live["1"] = RemoteBox(ext_id="1", label="gf-bdeadbeef", status="running")
    prov.live["2"] = RemoteBox(ext_id="2", label="liftloom-keep", status="running")
    fleet = Fleet(make_cfg(tmp), {"fake": prov}, local_boxes=True)
    rep = await fleet.reconcile()
    assert rep["orphans"] == ["1"] and rep["foreign"] == 1
    assert list(prov.live) == ["2"]


async def test_daily_cap_blocks_paid_launch_but_not_prepaid(tmp):
    paid = Fake("paid", offers=[{"gpu_name": "RTX 4090", "price_h": 0.4}])
    fleet = Fleet(make_cfg(tmp, daily_cap_usd=0.05), {"paid": paid}, local_boxes=True)
    fleet.store.add_spend("paid", 0.05)
    jid = fleet.submit(spec(tmp, units=["a"]))
    await settle(fleet, 1.5)
    assert not paid.launched
    assert fleet.store.one("SELECT 1 FROM events WHERE kind='scale_blocked'")
    pre = Fake("pre", offers=[{"gpu_name": "RTX PRO 6000 S", "price_h": 0.3, "cu_per_h": 8.9}], prepaid=True)
    fleet.providers["pre"] = pre
    st = await run_until(fleet, jid)
    assert st["status"] == "done" and pre.launched
    await fleet.shutdown(destroy=True)


async def test_prefers_cheaper_expected_cost(tmp):
    # A fast expensive card vs a slow cheap one for a long queue: the scheduler should weigh speed, not just price.
    prov = Fake(offers=[{"gpu_name": "RTX 5090", "price_h": 0.6}, {"gpu_name": "L4", "price_h": 0.3}])
    fleet = Fleet(make_cfg(tmp), {"fake": prov}, local_boxes=True)
    jid = fleet.submit(spec(tmp, units=["a"], unit_cost_s=1800))
    best = await fleet._best_offer(fleet._spec(jid), 1800)
    assert best.gpu_name == "RTX 5090"
    await fleet.shutdown(destroy=True)


async def test_speculative_copy_finishes_a_straggler(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 4090", "price_h": 0.4}])
    fleet = Fleet(make_cfg(tmp, max_boxes=2, idle_s=30, speculate_min_s=2), {"fake": prov}, local_boxes=True)
    # The first attempt of unit "slow" hangs; its speculative twin (GF_ATTEMPT stays 1 for the twin, so key the
    # hang on the box workspace instead) finishes at once.
    cmd = ('mkdir -p out; if [ "$GF_UNIT" = slow ] && [ ! -e /tmp/gf-spec-$GF_JOB ]; then touch /tmp/gf-spec-$GF_JOB; '
           'sleep 300; fi; echo $GF_UNIT > out/$GF_TASK.txt')
    jid = fleet.submit(spec(tmp, units=["slow", "fast"], unit_cost_s=0.01, command=cmd))
    st = await run_until(fleet, jid, timeout=150)
    assert st["status"] == "done", st
    assert fleet.store.one("SELECT 1 FROM events WHERE kind='speculate'")
    await fleet.shutdown(destroy=True)
    Path(f"/tmp/gf-spec-{jid}").unlink(missing_ok=True)


def _kill_box(fleet: Fleet, prov: Fake, box_id: str) -> None:
    """Simulate a box that stops answering: its processes die and SSH no longer connects."""
    import os
    import signal
    b = fleet.boxes[box_id]
    b.transport.alive = False
    state = Path(b.transport.root).parent / "state"
    for meta in state.glob("*/meta.json"):
        try:
            os.killpg(json.loads(meta.read_text())["pid"], signal.SIGKILL)
        except OSError:
            pass
    b.proc.kill()


async def test_box_dies_mid_task_and_retry_resumes_from_checkpoint(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 4090", "price_h": 0.4, "machine_id": 1},
                        {"gpu_name": "RTX 4090", "price_h": 0.5, "machine_id": 2}])
    fleet = Fleet(make_cfg(tmp, max_boxes=2), {"fake": prov}, local_boxes=True)
    renders = tmp / "renders.log"  # every frame actually rendered, on any box
    # A resume-aware "render": 20 frames, skips frames already present, 0.15 s each.
    cmd = (f'mkdir -p frames out; for f in $(seq 1 20); do [ -e frames/$f.png ] && continue; sleep 0.15; '
           f'echo $f > frames/$f.png; echo $f >> {renders}; done; ls frames | wc -l > out/$GF_TASK.txt')
    jid = fleet.submit(spec(tmp, units=["shot"], command=cmd, checkpoint_s=0.3,
                            checkpoint=[{"src": "frames/", "dst": str(tmp / "ckpt") + "/"}]))
    end = time.time() + 30
    killed = None
    while time.time() < end:
        await fleet.tick()
        done_frames = len(list((tmp / "ckpt").glob("*.png"))) if (tmp / "ckpt").exists() else 0
        if killed is None and done_frames >= 6:
            killed = fleet.store.one("SELECT box_id FROM tasks WHERE job_id=?", [jid])["box_id"]
            _kill_box(fleet, prov, killed)
        if fleet.job_status(jid)["status"] == "done":
            break
        await asyncio.sleep(0.1)
    st = await run_until(fleet, jid, timeout=60)
    assert killed and st["status"] == "done", st
    assert (tmp / "results").glob("*.txt") and next((tmp / "results").glob("*.txt")).read_text().strip() == "20"
    rendered = renders.read_text().split()
    # Frames saved by the checkpoint were not rendered again (a few after the last checkpoint may be).
    assert len(rendered) < 20 + 6, rendered
    assert fleet.store.one("SELECT 1 FROM events WHERE kind='checkpoint_restored'")
    await fleet.shutdown(destroy=True)
    assert not prov.live


async def test_daemon_restart_reattaches_and_collects_finished_work(tmp):
    prov = Fake()
    cfg = make_cfg(tmp, idle_s=60)
    a = Fleet(cfg, {"fake": prov}, local_boxes=True)
    jid = a.submit(spec(tmp, units=["u"], command="sleep 2; mkdir -p out; echo u > out/$GF_TASK.txt"))
    end = time.time() + 20
    while time.time() < end and not a.store.one("SELECT 1 FROM tasks WHERE status='running'"):
        await a.tick()
        await asyncio.sleep(0.1)
    # Crash: the daemon disappears without destroying anything; the task keeps running on the box.
    for b in a.boxes.values():
        b.reader.cancel()
        b.proc.kill()
    a.store.db.close()
    b = Fleet(cfg, {"fake": prov}, local_boxes=True)
    await b.recover()
    st = await run_until(b, jid, timeout=30)
    assert st["status"] == "done", st
    assert len(prov.launched) == 1  # reattached, not re-rented
    assert b.store.one("SELECT 1 FROM events WHERE kind='box_recovered'")
    await b.shutdown(destroy=True)


async def test_hung_task_times_out_and_moves(tmp):
    prov = Fake(offers=[{"gpu_name": "RTX 4090", "price_h": 0.4, "machine_id": 1},
                        {"gpu_name": "RTX 4090", "price_h": 0.5, "machine_id": 2}])
    fleet = Fleet(make_cfg(tmp, max_boxes=2), {"fake": prov}, local_boxes=True)
    cmd = 'mkdir -p out; [ "$GF_ATTEMPT" = 1 ] && sleep 600; echo ok > out/$GF_TASK.txt'
    jid = fleet.submit(spec(tmp, units=["u"], command=cmd, timeout_min_s=2, speculate=False))
    st = await run_until(fleet, jid, timeout=40)
    assert st["status"] == "done", st
    assert fleet.store.one("SELECT 1 FROM events WHERE kind='task_timeout'")
    await fleet.shutdown(destroy=True)


async def test_idle_box_is_parked_and_restarted_for_the_next_job(tmp):
    prov = Fake()
    fleet = Fleet(make_cfg(tmp, park_max=2), {"fake": prov}, local_boxes=True)
    j1 = fleet.submit(spec(tmp, units=["a"]))
    assert (await run_until(fleet, j1))["status"] == "done"
    await settle(fleet, 2.5)  # idle_s=1 -> parked, not destroyed
    assert [b.status for b in fleet.boxes.values()] == ["parked"]
    t0 = time.time()
    j2 = fleet.submit(spec(tmp, units=["b"]))
    assert (await run_until(fleet, j2))["status"] == "done"
    assert len(prov.launched) == 1 and prov.unparked == 1  # reused the parked box, no new rental
    assert time.time() - t0 < 10
    await fleet.shutdown(destroy=True)
    assert not prov.live


async def test_failed_unpark_falls_back_to_a_new_box(tmp):
    prov = Fake()
    fleet = Fleet(make_cfg(tmp, park_max=2), {"fake": prov}, local_boxes=True)
    j1 = fleet.submit(spec(tmp, units=["a"]))
    await run_until(fleet, j1)
    await settle(fleet, 2.5)
    prov.fail_unpark = True
    j2 = fleet.submit(spec(tmp, units=["b"]))
    assert (await run_until(fleet, j2))["status"] == "done"
    assert len(prov.launched) == 2 and fleet.store.one("SELECT 1 FROM events WHERE kind='unpark_failed'")
    await fleet.shutdown(destroy=True)
    assert not prov.live


async def test_warm_pool_keeps_a_ready_box(tmp):
    prov = Fake()
    fleet = Fleet(make_cfg(tmp), {"fake": prov}, local_boxes=True)
    fleet.set_warm("base", 1, 1)
    await settle(fleet, 2.5)  # idle_s=1, but the warm target keeps it
    assert [b.status for b in fleet.boxes.values()] == ["ready"]
    t0 = time.time()
    j = fleet.submit(spec(tmp, units=["a"]))
    assert (await run_until(fleet, j))["status"] == "done"
    assert time.time() - t0 < 3 and len(prov.launched) == 1
    await fleet.shutdown(destroy=True)


async def test_stuck_destroying_box_is_retried_by_reconciler(tmp):
    from gpufleet.models import RemoteBox
    prov = Fake()
    fleet = Fleet(make_cfg(tmp), {"fake": prov}, local_boxes=True)
    now = time.time()
    prov.live["77"] = RemoteBox(ext_id="77", label="gf-bstuck", status="running")
    fleet.store.x("INSERT INTO boxes (id, provider, ext_id, label, status, created_at, hard_deadline) "
                  "VALUES ('bstuck','fake','77','gf-bstuck','destroying',?,?)", [now, now + 3600])
    await fleet.reconcile()
    assert "77" not in prov.live
    assert fleet.store.one("SELECT status FROM boxes WHERE id='bstuck'")["status"] == "destroyed"
