"""Python client for fleetd. Other languages: write JSON lines to the socket (see server.py)."""

from __future__ import annotations

import json
import socket
from typing import Iterator

from . import config


class Client:
    def __init__(self, path: str | None = None):
        self.path = path or config.load()["socket"]

    def _conn(self) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(self.path)
        except OSError as e:
            raise ConnectionError(f"fleetd is not running ({self.path}): start it with `fleet daemon` "
                                  "or `systemctl --user start fleetd`") from e
        return s

    def call(self, op: str, **kw) -> dict:
        with self._conn() as s:
            s.sendall((json.dumps({"op": op, **kw}) + "\n").encode())
            f = s.makefile("r")
            line = f.readline()
            out = json.loads(line) if line else {"error": "no reply"}
        if "error" in out and len(out) == 1:
            raise RuntimeError(out["error"])
        return out

    def submit(self, spec: dict) -> str:
        return self.call("submit", spec=spec)["job"]

    def wait(self, job: str) -> Iterator[dict]:
        """Yields events until the job ends; the last one has ev='final' and the job status."""
        with self._conn() as s:
            s.sendall((json.dumps({"op": "wait", "job": job}) + "\n").encode())
            for line in s.makefile("r"):
                ev = json.loads(line)
                yield ev
                if ev.get("ev") == "final":
                    return

    def run(self, spec: dict) -> dict:
        """Submit and block until done; returns the final status."""
        job = self.submit(spec)
        final = {}
        for ev in self.wait(job):
            final = ev
        return final
