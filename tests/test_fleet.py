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
        "lane_autoscale": False, **over})
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
