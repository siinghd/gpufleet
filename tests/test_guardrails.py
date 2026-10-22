"""Spending guardrails on the fake provider: prepaid unit caps and floors acting on running boxes, per-project daily
budgets, alerts, and how status shows them. (Oct 2026: an overnight render ran the Colab month from 200 CU to 0.)"""

import asyncio
import json
import time

import pytest

from gpufleet import daemon
from gpufleet.daemon import Fleet
from gpufleet.providers.fake import Fake
from gpufleet.store import Store
from tests.test_fleet import make_cfg, run_until, settle, spec

G4 = {"gpu_name": "RTX PRO 6000 S", "price_h": 0.3, "cu_per_h": 8.9}
SLOW = "sleep 1.5; mkdir -p out && echo $GF_UNIT > out/$GF_TASK.txt"


def prepaid(**kw) -> Fake:
    return Fake("pre", offers=[G4], prepaid=True, max_boxes=1, **kw)  # one box, like Colab


async def tick_until(fleet: Fleet, cond, timeout: float = 30) -> None:
    end = time.time() + timeout
    while time.time() < end:
        await fleet.tick()
        if cond():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("condition not reached")


def alerts(fleet: Fleet) -> list[str]:
    return [r["ref"] for r in fleet.store.q("SELECT ref FROM events WHERE kind='alert' ORDER BY at")]


def running(fleet: Fleet):
    return lambda: fleet.store.one("SELECT 1 FROM tasks WHERE status='running'")


async def test_unit_cap_blocks_prepaid_launch(tmp_path):
    pre = prepaid()
    fleet = Fleet(make_cfg(tmp_path, providers={"pre": {"daily_cap_units": 5}}), {"pre": pre}, local_boxes=True)
    fleet.store.add_spend("pre", 0, 4)  # 4 units + 15 min of a G4 (2.2) would pass the cap
    jid = fleet.submit(spec(tmp_path, units=["a"]))
    await settle(fleet, 1.5)
    assert not pre.launched
    st = fleet.job_status(jid)
    assert st["status"] == "queued" and "pre daily cap 5 units" in st["held"], st
    assert alerts(fleet) == ["cap:pre:units"]
    # A paid provider still takes the work: the USD cap governs it.
    paid = Fake("paid")
    fleet.providers["paid"] = paid
    st = await run_until(fleet, jid)
    assert st["status"] == "done" and st["held"] is None and paid.launched and not pre.launched
    await fleet.shutdown(destroy=True)


async def test_unit_cap_drains_a_running_prepaid_box(tmp_path):
    pre = prepaid()
    fleet = Fleet(make_cfg(tmp_path, idle_s=60, providers={"pre": {"daily_cap_units": 5}}), {"pre": pre},
                  local_boxes=True)
    jid = fleet.submit(spec(tmp_path, units=["a", "b", "c"], command=SLOW, speculate=False))
    await tick_until(fleet, running(fleet))
    fleet.store.add_spend("pre", 0, 5)  # today's units reach the cap while the first unit renders
    await tick_until(fleet, lambda: not pre.live)
    assert fleet.job_status(jid)["tasks"] == {"done": 1, "queued": 2}  # it finished its task, started no other
    assert fleet.store.one("SELECT 1 FROM events WHERE kind='guard_drain'")
    assert fleet.store.one("SELECT 1 FROM events WHERE kind='box_destroyed' AND data LIKE '%drained%'")
    assert alerts(fleet) == ["cap:pre:units"]
    await settle(fleet, 1)
    assert len(pre.launched) == 1  # no new box while the cap holds
    assert "daily cap 5 units reached" in fleet.job_status(jid)["held"]
    await fleet.shutdown(destroy=True)


async def test_floor_drains_a_busy_prepaid_box(tmp_path):
    pre = prepaid(balance_units=100)
    fleet = Fleet(make_cfg(tmp_path, idle_s=60, providers={"pre": {"cu_floor": 15}}), {"pre": pre},
                  local_boxes=True)
    jid = fleet.submit(spec(tmp_path, units=["a", "b"], command=SLOW, speculate=False))
    await tick_until(fleet, running(fleet))
    pre.balance_units = 12
    await fleet.reconcile()  # the reconciler reads the balance
    await tick_until(fleet, lambda: not pre.live)
    assert fleet.job_status(jid)["tasks"] == {"done": 1, "queued": 1}
    ev = fleet.store.one("SELECT data FROM events WHERE kind='provider_stopped' AND ref='pre'")
    assert ev and "under the floor 15" in ev["data"]
    await settle(fleet, 1)
    assert len(pre.launched) == 1 and "under the floor" in fleet.job_status(jid)["held"]
    # Back over the floor (a new month's grant): work resumes.
    pre.balance_units = 200
    await fleet.reconcile()
    assert (await run_until(fleet, jid))["status"] == "done" and len(pre.launched) == 2
    await fleet.shutdown(destroy=True)


async def test_floor_destroys_an_idle_box_at_once(tmp_path):
    pre = prepaid(balance_units=100)
    fleet = Fleet(make_cfg(tmp_path, providers={"pre": {"cu_floor": 15}}), {"pre": pre}, local_boxes=True)
    fleet.set_warm("base", 1, 10)  # an idle box the warm pool would otherwise keep
    await tick_until(fleet, lambda: [b.status for b in fleet.boxes.values()] == ["ready"])
    pre.balance_units = 10
    await fleet.reconcile()
    await fleet.tick()
    assert not pre.live and not fleet.boxes
    await settle(fleet, 1)
    assert len(pre.launched) == 1  # the warm pool rents nothing under the floor


async def test_daily_usd_cap_alerts_once_and_shows_why(tmp_path):
    paid = Fake("paid")
    fleet = Fleet(make_cfg(tmp_path, daily_cap_usd=0.05), {"paid": paid}, local_boxes=True)
    fleet.store.add_spend("paid", 0.05)
    jid = fleet.submit(spec(tmp_path, units=["a"]))
    await settle(fleet, 1.5)
    assert not paid.launched and alerts(fleet) == ["cap:usd"]
    assert fleet.job_status(jid)["held"].startswith("daily cap $0.05")


async def test_project_budget_holds_its_jobs_and_spend_is_tracked_per_project(tmp_path):
    prov = Fake()
    fleet = Fleet(make_cfg(tmp_path, budgets={"film": {"daily_usd": 0.5}}), {"fake": prov}, local_boxes=True)
    fleet.store.add_project_spend("film", "fake", 0.5)
    held = fleet.submit(spec(tmp_path, project="film", units=["a"]))
    other = fleet.submit(spec(tmp_path, project="other", units=["b"]))
    assert (await run_until(fleet, other))["status"] == "done"
    st = fleet.job_status(held)
    assert st["status"] == "queued" and "project film daily budget $0.50 reached" in st["held"], st
    assert fleet.store.one("SELECT COUNT(*) AS n FROM tasks WHERE job_id=? AND status='queued'", [held])["n"] == 1
    assert alerts(fleet) == ["budget:film:usd"]
    # The box rented for the other project is charged to it, not to the held one.
    assert fleet.store.project_spend_today("other")[0] > 0
    assert fleet.store.project_spend_today("film")[0] == pytest.approx(0.5)
    p = fleet.fleet_status()["projects"]
    assert p["film"]["held"] and p["film"]["daily_usd"] == 0.5 and p["other"]["daily_usd"] is None
    fleet.cfg["budgets"]["film"]["daily_usd"] = 5  # a bigger budget releases the job
    assert (await run_until(fleet, held))["status"] == "done"
    await fleet.shutdown(destroy=True)


async def test_project_unit_budget_refuses_a_prepaid_launch(tmp_path):
    pre = prepaid()
    fleet = Fleet(make_cfg(tmp_path, budgets={"film": {"daily_units": 3}}), {"pre": pre}, local_boxes=True)
    fleet.store.add_project_spend("film", "pre", 0, 1.5)  # 1.5 + 15 min of a G4 (2.2) would pass 3
    jid = fleet.submit(spec(tmp_path, project="film", units=["a"]))
    await settle(fleet, 1.5)
    assert not pre.launched and "project film budget 3 units/day" in fleet.job_status(jid)["held"]


async def test_balance_alerts_once_per_threshold_and_run_alert_cmd(tmp_path):
    got = tmp_path / "alerts.txt"
    hook = tmp_path / "hook.sh"
    hook.write_text(f'#!/bin/sh\necho "$1|$2" >> {got}\n')
    hook.chmod(0o755)
    pre = prepaid(balance_units=90)
    cfg = make_cfg(tmp_path, alert_cmd=f"{hook} desk", providers={"pre": {"monthly_units": 200}})
    fleet = Fleet(cfg, {"pre": pre}, local_boxes=True)
    await fleet.reconcile()
    await settle(fleet, 1)
    await settle(fleet, 0.5)  # still under 50 %: no second alert
    pre.balance_units = 30
    await fleet.reconcile()
    await settle(fleet, 1)
    assert alerts(fleet) == ["balance:pre:50", "balance:pre:20"]
    lines = got.read_text().splitlines()
    assert len(lines) == 2 and all(x.startswith("desk|gpufleet: pre balance") for x in lines), lines  # message last
    # A restart the same day does not raise them again.
    again = Fleet(cfg, {"pre": pre}, local_boxes=True)
    await again.reconcile()
    await settle(again, 1)
    assert len(alerts(again)) == 2 and len(got.read_text().splitlines()) == 2


async def test_alert_cmd_failures_and_hangs_are_ignored(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(daemon, "ALERT_CMD_TIMEOUT_S", 0.5)
    fleet = Fleet(make_cfg(tmp_path), {"fake": Fake()}, local_boxes=True)
    t0 = time.time()
    await fleet._run_alert_cmd("sh -c 'sleep 30'", "x")  # the message lands in $0 and is ignored
    assert time.time() - t0 < 5
    await fleet._run_alert_cmd(str(tmp_path / "missing"), "x")
    await fleet._run_alert_cmd("false", "x")
    assert "timed out" in caplog.text and "alert_cmd failed" in caplog.text and "exited 1" in caplog.text


def test_old_database_gets_the_project_spend_table(tmp_path):
    db = tmp_path / "fleet.db"
    s = Store(db)
    s.add_spend("vast", 1.25)
    s.x("DROP TABLE spend_project")  # a database from before per-project budgets
    s.db.close()
    s = Store(db)
    assert s.spend_today("vast") == 1.25 and s.project_spend_today("p") == (0.0, 0.0)
    s.add_project_spend("p", "vast", 0.5)
    s.add_project_spend("p", "colab", 0, 2)
    assert s.project_spend_today("p") == (0.5, 2.0)


async def test_status_and_cli_show_caps_budgets_and_holds(tmp_path, monkeypatch, capsys):
    pre = prepaid(balance_units=120)
    cfg = make_cfg(tmp_path, budgets={"film": {"daily_units": 3}},
                   providers={"pre": {"daily_cap_units": 40, "cu_floor": 15, "monthly_units": 200}})
    fleet = Fleet(cfg, {"pre": pre}, local_boxes=True)
    fleet.store.add_spend("pre", 0, 3.5)
    fleet.store.add_project_spend("film", "pre", 0, 3.5)
    jid = fleet.submit(spec(tmp_path, project="film", units=["a"]))
    await fleet.reconcile()
    await settle(fleet, 0.5)
    out = fleet.fleet_status()
    assert out["units_today"]["pre"] == {"units": 3.5, "cap": 40, "balance": 120.0, "floor": 15,
                                         "monthly_units": 200, "stopped": None}
    assert out["projects"]["film"]["held"] and out["jobs"][0]["held"].startswith("project film daily budget 3 units")

    from gpufleet import cli, client

    class FakeClient:
        def __init__(self, *a):
            pass

        def call(self, op, **kw):
            return {"balances": {"pre": 120.0}, **out} if op == "usage" else out

    monkeypatch.setattr(client, "Client", FakeClient)
    assert cli.main(["status"]) == 0
    text = capsys.readouterr().out
    assert "units today pre 3.5/40  balance 120.0 (floor 15, month 200)" in text, text
    assert "project film  $0.00  3.5/3 units  HELD: project film daily budget 3 units reached" in text, text
    assert f"{jid}  film/j  queued" in text and "held: project film" in text, text
    assert cli.main(["usage"]) == 0
    usage = json.loads(capsys.readouterr().out)
    assert usage["units_today"]["pre"]["cap"] == 40 and usage["held"][jid].startswith("project film")
    assert usage["projects"]["film"]["daily_units"] == 3
