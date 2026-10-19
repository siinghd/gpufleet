#!/usr/bin/env python3
"""gpufleet box agent. Copied to the box and run over the daemon's SSH connection:

    python3 agent.py serve --root /work --state /var/lib/gpufleet

Speaks JSON lines: commands on stdin, events on stdout. Standard library only, Python >= 3.8, because it runs on
whatever the box image ships.

Tasks outlive the agent: each one runs in its own session with its exit code written to a file, so if the SSH
channel drops (or the daemon restarts) the work keeps going and a new `serve` reports what finished meanwhile. The
daemon acknowledges every exit; unacknowledged exits are re-reported on reconnect.

Commands:  {"op":"run","task":ID,"cmd":"...","env":{...},"cwd":"rel/or/abs","gpu":0|null}
           {"op":"cancel","task":ID}   {"op":"ack","task":ID}   {"op":"ping"}
Events:    {"ev":"hello",...} {"ev":"running","task":ID} {"ev":"exit","task":ID,"rc":N,"secs":S,"tail":"..."}
           {"ev":"tel","gpus":[{"i":0,"util":..,"mem":..,"mem_total":..,"power":..}],"load":..} {"ev":"pong"}
"""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

OUT_LOCK = threading.Lock()


def emit(**ev):
    line = json.dumps(ev, separators=(",", ":"))
    with OUT_LOCK:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class Telemetry(threading.Thread):
    """Samples every GPU once a second via a single long-lived nvidia-smi and keeps a 10-second window."""

    FIELDS = "index,utilization.gpu,memory.used,memory.total,power.draw"

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = {}  # gpu index -> list of (t, util, mem, mem_total, power)
        self.lock = threading.Lock()

    def run(self):
        while True:
            try:
                p = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits",
                                      "-lms", "1000"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except FileNotFoundError:
                return  # CPU box
            for line in p.stdout:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) < 5:
                    continue
                try:
                    i, util, mem, tot = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    power = float(parts[4]) if parts[4] not in ("[N/A]", "N/A") else 0.0
                except ValueError:
                    continue
                now = time.time()
                with self.lock:
                    s = self.samples.setdefault(i, [])
                    s.append((now, util, mem, tot, power))
                    while s and s[0][0] < now - 10:
                        s.pop(0)
            p.wait()
            time.sleep(5)

    def snapshot(self):
        with self.lock:
            out = []
            for i, s in sorted(self.samples.items()):
                if not s:
                    continue
                n = len(s)
                out.append({"i": i, "util": round(sum(x[1] for x in s) / n, 1), "mem": max(x[2] for x in s),
                            "mem_total": s[-1][3], "power": round(sum(x[4] for x in s) / n, 1)})
            return out


class Agent:
    def __init__(self, root, state):
        self.root = root
        self.state = state
        os.makedirs(root, exist_ok=True)
        os.makedirs(state, exist_ok=True)
        self.tel = Telemetry()
        self.reported = set()
        self.procs = {}

    def tdir(self, task):
        return os.path.join(self.state, task)

    def meta(self, task):
        try:
            with open(os.path.join(self.tdir(task), "meta.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def run(self, msg):
        task = msg["task"]
        d = self.tdir(task)
        if os.path.exists(d):
            emit(ev="running", task=task, dup=True)  # already started (re-sent after a reconnect)
            return
        os.makedirs(d)
        cwd = msg.get("cwd") or "."
        cwd = cwd if cwd.startswith("/") else os.path.join(self.root, cwd)
        os.makedirs(cwd, exist_ok=True)
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (msg.get("env") or {}).items()})
        env["GF_TASK"] = task
        env["GF_ROOT"] = self.root
        if msg.get("gpu") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(msg["gpu"])
        log = os.path.join(d, "log")
        rc = os.path.join(d, "rc")
        # The wrapper writes the exit code atomically; a task killed with its whole session never writes one,
        # which reads as "lost" rather than as a success.
        wrapped = "(%s) >%s 2>&1; echo $? >%s.tmp && mv %s.tmp %s" % (
            msg["cmd"], shlex.quote(log), shlex.quote(rc), shlex.quote(rc), shlex.quote(rc))
        p = subprocess.Popen(["bash", "-c", wrapped], cwd=cwd, env=env, start_new_session=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"task": task, "pid": p.pid, "started": time.time()}, f)
        self.procs[task] = p
        emit(ev="running", task=task)

    def cancel(self, task):
        m = self.meta(task)
        if not m:
            return
        try:
            os.killpg(m["pid"], signal.SIGTERM)
        except OSError:
            return

        def later():
            time.sleep(10)
            try:
                os.killpg(m["pid"], signal.SIGKILL)
            except OSError:
                pass

        threading.Thread(target=later, daemon=True).start()

    def ack(self, task):
        d = self.tdir(task)
        for name in ("log", "rc", "meta.json"):
            try:
                os.remove(os.path.join(d, name))
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass
        self.procs.pop(task, None)
        self.reported.discard(task)

    def tail(self, task, n=40, limit=6000):
        try:
            with open(os.path.join(self.tdir(task), "log"), "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - limit))
                return b"\n".join(f.read().splitlines()[-n:]).decode(errors="replace")
        except OSError:
            return ""

    def scan(self):
        """Report finished tasks once each; detect tasks whose process died without writing an exit code."""
        try:
            tasks = os.listdir(self.state)
        except OSError:
            return
        for task in tasks:
            if task in self.reported:
                continue
            d = self.tdir(task)
            rcf = os.path.join(d, "rc")
            m = self.meta(task)
            if os.path.exists(rcf):
                try:
                    with open(rcf) as f:
                        rc = int(f.read().strip() or 1)
                except (OSError, ValueError):
                    rc = 1
                p = self.procs.get(task)
                if p is not None:
                    p.poll()  # reap the wrapper
                secs = time.time() - m["started"] if m else 0
                self.reported.add(task)
                emit(ev="exit", task=task, rc=rc, secs=round(secs, 2), tail=self.tail(task))
            elif m and not self._alive(m["pid"], task):
                self.reported.add(task)
                emit(ev="exit", task=task, rc=-9, secs=round(time.time() - m["started"], 2),
                     tail=self.tail(task) + "\n[gpufleet] task process vanished without an exit code", lost=True)

    def _alive(self, pid, task):
        p = self.procs.get(task)
        if p is not None:
            return p.poll() is None or os.path.exists(os.path.join(self.tdir(task), "rc"))
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def loop(self):
        self.tel.start()
        known = os.listdir(self.state)
        emit(ev="hello", pid=os.getpid(), root=self.root, py=sys.version.split()[0], known=known,
             running=[t for t in known if not os.path.exists(os.path.join(self.tdir(t), "rc"))])

        def reader():
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                op = msg.get("op")
                try:
                    if op == "run":
                        self.run(msg)
                    elif op == "cancel":
                        self.cancel(msg["task"])
                    elif op == "ack":
                        self.ack(msg["task"])
                    elif op == "ping":
                        emit(ev="pong", t=msg.get("t"))
                except Exception as e:  # report and keep serving; one bad message must not kill the channel
                    emit(ev="error", op=op, task=msg.get("task"), error=repr(e))
            os._exit(0)  # daemon went away; running tasks continue in their own sessions

        threading.Thread(target=reader, daemon=True).start()
        last_tel = 0.0
        while True:
            self.scan()
            now = time.time()
            if now - last_tel >= 5:
                last_tel = now
                try:
                    load = os.getloadavg()[0]
                except OSError:
                    load = 0
                emit(ev="tel", gpus=self.tel.snapshot(), load=round(load, 2), ncpu=os.cpu_count())
            time.sleep(0.5)


def main():
    args = sys.argv[1:]
    if not args or args[0] != "serve":
        print(__doc__)
        sys.exit(2)
    opts = dict(zip(args[1::2], args[2::2]))
    Agent(opts.get("--root", "/work"), opts.get("--state", "/var/lib/gpufleet")).loop()


if __name__ == "__main__":
    main()
