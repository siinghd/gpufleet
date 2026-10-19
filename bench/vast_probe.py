"""Rent one box per GPU model on Vast, run bench/probe.sh on each, destroy them (verified). Exercises the Vast adapter
and SshTransport end to end.

    .venv/bin/python bench/vast_probe.py "RTX 3090" "RTX 4090" "RTX 5090"
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from gpufleet.models import Need, Sync
from gpufleet.providers.vast import Vast
from gpufleet.transport import SshTransport

IMAGE = "ghcr.io/siinghd/numora-render-box:5.2.2-20261008"
STATE = Path.home() / ".local/state/gpufleet"
HERE = Path(__file__).resolve().parent


async def one(gpu: str, out: Path) -> dict:
    v = Vast({"gpu_names": [gpu], "max_bandwidth_usd_per_tb": 10})
    res: dict = {"gpu": gpu}
    ext = None
    try:
        offers = await v.offers(Need(max_price_h=0.9, rt_cores=True), set(), set())
        if not offers:
            return {**res, "error": "no offers"}
        t0 = time.time()
        tried = []
        for o in sorted(offers, key=lambda x: x.price_h)[:8]:
            try:
                ext = await v.launch(o, f"gf-probe-{gpu.split()[-1].lower()}", {"vast_image": IMAGE, "disk_gb": 40},
                                     3600)
                break
            except Exception as e:
                tried.append(f"{o.offer_id}: {str(e)[-80:]}")
        res["stale_offers"] = len(tried)
        if not ext:
            return {**res, "error": "all offers failed", "tried": tried}
        res.update(offer=o.offer_id, price_h=o.price_h, machine=o.machine_id, raw=o.raw)
        res["ext_id"] = ext
        tgt = await v.wait_ready(ext, time.time() + 900)
        res["running_s"] = round(time.time() - t0, 1)
        tr = SshTransport(f"probe{ext}", tgt, STATE, v.ssh_options())
        await tr.open(timeout=300)
        res["ssh_s"] = round(time.time() - t0, 1)
        t1 = time.time()
        await tr.push(Sync(src=str(HERE), dst="/work/bench"))
        res["push_s"] = round(time.time() - t1, 1)
        rc, text = await tr.exec("PROBE_DIR=/work/probe bash /work/bench/probe.sh 5.2.2 48", timeout=2400)
        (out / f"probe-{gpu.replace(' ', '_')}.txt").write_text(text)
        res["rc"] = rc
        res["bench"] = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("BENCH=")]
        res["kv"] = dict(l.split("=", 1) for l in text.splitlines() if "=" in l and not l.startswith("BENCH"))
        res["total_s"] = round(time.time() - t0, 1)
        await tr.close()
        # Park/unpark: how fast does a stopped box come back?
        await v.park(ext)
        await asyncio.sleep(20)
        t2 = time.time()
        try:
            tgt2 = await v.unpark(ext, time.time() + 180)
            tr2 = SshTransport(f"probe2{ext}", tgt2, STATE, v.ssh_options())
            await tr2.open(timeout=180)
            rc2, _ = await tr2.exec("test -x /work/probe/blender/blender && nvidia-smi -L", 60)
            res["unpark_s"] = round(time.time() - t2, 1)
            res["unpark_files_kept"] = rc2 == 0
            await tr2.close()
        except Exception as e:
            res["unpark_error"] = repr(e)[-200:]
    except Exception as e:
        res["error"] = repr(e)
    finally:
        if ext:
            try:
                await v.destroy(ext)
                res["destroyed"] = True
            except Exception as e:
                res["destroyed"] = f"FAILED: {e!r}"
        await v.close()
    return res


async def main():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    results = await asyncio.gather(*(one(g, out) for g in sys.argv[1:]))
    for r in results:
        print(json.dumps(r))
    (out / f"vast-{time.strftime('%Y%m%d-%H%M')}.json").write_text(json.dumps(results, indent=1))


asyncio.run(main())
