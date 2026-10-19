"""The socket API and the CLI against a daemon on the fake provider."""

import asyncio
import json
import threading

from gpufleet.client import Client
from gpufleet.daemon import Fleet
from gpufleet.providers.fake import Fake
from gpufleet.server import Server
from tests.test_fleet import make_cfg


async def test_submit_wait_over_socket(tmp_path):
    cfg = make_cfg(tmp_path)
    sock = str(tmp_path / "s.sock")
    fleet = Fleet(cfg, {"fake": Fake()}, local_boxes=True)
    srv = Server(fleet, sock)
    server = await asyncio.start_unix_server(srv.handle, path=sock)
    runner = asyncio.create_task(fleet.run())
    spec = {"project": "t", "name": "s", "units": ["a", "b"], "unit_cost_s": 1,
            "command": "mkdir -p out && echo $GF_UNIT > out/$GF_TASK.txt",
            "outputs": [{"src": "out/{task}.txt", "dst": str(tmp_path / "res") + "/{unit}.txt"}]}
    result = {}

    def client():
        cl = Client(sock)
        result["final"] = cl.run(spec)
        result["status"] = cl.call("status")

    th = threading.Thread(target=client)
    th.start()
    while th.is_alive():
        await asyncio.sleep(0.1)
    assert result["final"]["status"] == "done"
    assert sorted(p.read_text().strip() for p in (tmp_path / "res").glob("*.txt")) == ["a", "b"]
    assert "boxes" in result["status"]
    fleet._stop.set()
    await runner
    await fleet.shutdown(destroy=True)
    server.close()
