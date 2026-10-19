"""fleet: command line for gpufleet.

  fleet daemon                         run the scheduler (or: systemctl --user start fleetd)
  fleet submit job.json [--wait]       submit a job (JSON spec, see README); --wait streams progress
  fleet run job.json                   submit and wait; exit 0 done, 1 failed, 2 cancelled
  fleet status [JOB]                   fleet overview, or one job
  fleet tasks JOB                      per-task state, attempts, log tails
  fleet cancel JOB
  fleet warm PROFILE [BOXES] [MINUTES] keep boxes ready ahead of work (seconds-latency starts)
  fleet offers [--profile P] [--max-price X] [--gpu]   what renting would cost right now
  fleet usage                          spend today and provider balances
  fleet events [N]                     recent scheduler events
  fleet drain                          destroy idle boxes now, the rest when their tasks end
  fleet reconcile                      compare providers with the database now

Client commands import nothing heavy, so they return in milliseconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _fmt_ev(ev: dict) -> str:
    kind = ev.get("ev")
    what = ev.get("unit") or (f"{ev.get('lo')}-{ev.get('hi')}" if ev.get("lo") is not None else "")
    if kind == "started":
        return f"start  {what:<16} on {ev.get('box')} ({ev.get('gpu')})"
    if kind == "done":
        return f"done   {what:<16} {ev.get('secs', 0):.1f}s on {ev.get('box')}"
    if kind == "failed":
        tail = (ev.get("tail") or "").strip().splitlines()[-3:]
        return f"FAIL   {what:<16} rc={ev.get('rc')} attempt {ev.get('attempt')}\n         " + "\n         ".join(tail)
    if kind == "scaling":
        return f"rent   {ev.get('provider')} {ev.get('gpu')} ${ev.get('price_h')}/h"
    if kind == "progress":
        return f"...    {json.dumps(ev.get('tasks'))} cost ${ev.get('cost')}"
    if kind == "final":
        return f"{ev.get('status').upper()}  cost ${ev.get('cost')}  tasks {json.dumps(ev.get('tasks'))}"
    return f"{kind:<6} " + json.dumps({k: v for k, v in ev.items() if k not in ('ev', 'job', 't')})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fleet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("daemon")
    s = sub.add_parser("submit")
    s.add_argument("spec")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--json", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("spec")
    r.add_argument("--json", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("job", nargs="?")
    t = sub.add_parser("tasks")
    t.add_argument("job")
    c = sub.add_parser("cancel")
    c.add_argument("job")
    w = sub.add_parser("warm")
    w.add_argument("profile")
    w.add_argument("boxes", nargs="?", type=int, default=1)
    w.add_argument("minutes", nargs="?", type=float, default=30)
    o = sub.add_parser("offers")
    o.add_argument("--profile", default="base")
    o.add_argument("--max-price", type=float, default=1.0)
    o.add_argument("--rt", action="store_true", help="RT cores required (OptiX)")
    sub.add_parser("usage")
    e = sub.add_parser("events")
    e.add_argument("n", nargs="?", type=int, default=40)
    sub.add_parser("drain")
    sub.add_parser("reconcile")
    a = p.parse_args(argv)

    if a.cmd == "daemon":
        import asyncio
        from .server import serve
        asyncio.run(serve())
        return 0

    from .client import Client
    cl = Client()
    try:
        if a.cmd in ("submit", "run"):
            spec = json.load(sys.stdin if a.spec == "-" else open(a.spec))
            job = cl.submit(spec)
            if a.cmd == "submit" and not a.wait:
                print(job)
                return 0
            print(f"job {job}", file=sys.stderr)
            final = {}
            for ev in cl.wait(job):
                final = ev
                print(json.dumps(ev) if a.json else _fmt_ev(ev), flush=True)
            return {"done": 0, "failed": 1}.get(final.get("status"), 2)
        if a.cmd == "status":
            out = cl.call("status", job=a.job) if a.job else cl.call("status")
            if a.job:
                print(json.dumps(out, indent=1))
                return 0
            print(f"spend today {json.dumps(out['spend_today'])} (cap ${out['daily_cap_usd']})")
            for b in out["boxes"]:
                print(f"  {b['id']}  {b['provider']:<6} {b['gpu']:<26} {b['status']:<9} {b['busy']}/{b['lanes']} lanes "
                      f"util {b['util']}%  ${b['price_h']}/h  {b['age_min']} min")
            for j in out["jobs"]:
                print(f"  {j['id']}  {j['project']}/{j['name']}  {j['status']}  {json.dumps(j['tasks'])}  ${j['cost']}")
            return 0
        if a.cmd == "tasks":
            for t in cl.call("tasks", job=a.job)["tasks"]:
                what = t["unit"] or f"{t['lo']}-{t['hi']}"
                dur = f"{(t['finished_at'] or time.time()) - t['started_at']:.0f}s" if t["started_at"] else ""
                print(f"{t['id']}  {what:<16} {t['status']:<9} attempt {t['attempt']} rc={t['rc']} {dur} {t['box_id'] or ''}")
                if t["status"] == "failed" and t["tail"]:
                    print("    " + "\n    ".join(t["tail"].strip().splitlines()[-8:]))
            return 0
        if a.cmd == "cancel":
            cl.call("cancel", job=a.job)
            return 0
        if a.cmd == "warm":
            print(json.dumps(cl.call("warm", profile=a.profile, boxes=a.boxes, minutes=a.minutes)))
            return 0
        if a.cmd == "offers":
            need = {"profile": a.profile, "max_price_h": a.max_price, "rt_cores": a.rt}
            for x in cl.call("offers", need=need)["offers"]:
                print(f"{x['provider']:<6} {x['gpu']:<26} ${x['price_h']}/h {'prepaid ' if x['prepaid'] else ''}"
                      f"{'' if x['can_launch'] else 'BLOCKED: ' + x['why']}")
            return 0
        if a.cmd == "usage":
            out = cl.call("usage")
            print(json.dumps({"balances": out["balances"], "spend_today": out["spend_today"]}, indent=1))
            return 0
        if a.cmd == "events":
            for ev in reversed(cl.call("events", n=a.n)["events"]):
                print(time.strftime("%H:%M:%S", time.localtime(ev["at"])), ev["kind"], ev["ref"] or "", ev["data"] or "")
            return 0
        if a.cmd in ("drain", "reconcile"):
            print(json.dumps(cl.call(a.cmd)))
            return 0
    except (ConnectionError, RuntimeError) as e:
        print(f"fleet: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
