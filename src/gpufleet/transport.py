"""How the daemon reaches a box: one multiplexed SSH connection (OpenSSH ControlMaster) per box carries the agent
channel, one-off commands and every rsync. One connection is a hard rule on Colab (a second gets HTTP 429) and is
simply faster everywhere else (no handshake per rsync).

LocalTransport runs a "box" in a local directory, so the whole scheduler can be tested without renting anything.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from pathlib import Path

from .models import SshTarget, Sync


class TransportError(Exception):
    def __init__(self, msg: str, rc: int = 1):
        super().__init__(msg)
        self.rc = rc


async def _run(argv: list[str], timeout: float, stdin: bytes | None = None) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(p.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        p.kill()
        await p.wait()
        return 124, f"timeout after {timeout:.0f}s: {' '.join(argv[:6])}"
    return p.returncode or 0, out.decode(errors="replace")


class Transport:
    root = "/work"  # workspace root on the box

    async def open(self, timeout: float = 120) -> None: ...
    async def close(self) -> None: ...
    async def exec(self, cmd: str, timeout: float = 600, stdin: bytes | None = None) -> tuple[int, str]: ...
    async def spawn(self, cmd: str) -> asyncio.subprocess.Process: ...
    async def push(self, s: Sync, timeout: float = 3600) -> None: ...
    async def pull(self, s: Sync, timeout: float = 3600) -> None: ...

    def abspath(self, p: str) -> str:
        return p if p.startswith("/") else f"{self.root}/{p}"


class SshTransport(Transport):
    def __init__(self, box_id: str, target: SshTarget, state_dir: Path, extra_opts: list[str] = ()):
        self.box_id = box_id
        self.t = target
        d = state_dir / "ssh"
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        # ControlPath must stay short (108-byte socket path limit), hence the short name.
        self.sock = str(d / f"cm-{box_id[-12:]}")
        self.opts = [
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=20",
            "-o", "ControlMaster=auto", "-o", f"ControlPath={self.sock}", "-o", "ControlPersist=600",
            "-o", "Compression=no", "-o", "IPQoS=throughput",
            # Never wait on a password/passphrase prompt: there is no terminal, a refused key must fail fast.
            "-o", "BatchMode=yes", "-o", "PasswordAuthentication=no", "-o", "KbdInteractiveAuthentication=no",
            "-p", str(target.port), *extra_opts,
        ]
        if target.proxy_command:
            self.opts += ["-o", f"ProxyCommand={target.proxy_command}"]
        self.dest = f"{target.user}@{target.host}"
        self.master: asyncio.subprocess.Process | None = None
        # sshd allows MaxSessions (10 by default) sessions per connection, and everything here shares one: the agent
        # channel, a task's output pulls, the next job's input pushes, setup. Past the limit a session is refused
        # ("Session open refused by peer"), which read as a box fault and drained a healthy box. One session is the
        # agent's; the rest queue here.
        self._sessions = asyncio.Semaphore(int(os.environ.get("GF_SSH_SESSIONS", 6)))

    def _ssh(self) -> list[str]:
        return ["ssh", *self.opts]

    async def open(self, timeout: float = 120) -> None:
        """Start the master connection as our own child process and wait until a command runs through it.

        Providers report "running" before sshd accepts connections (Vast: often 5-60 s of "connection refused"),
        so a master that exits early is restarted until the deadline."""
        if self.master and self.master.returncode is None:
            return
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        last = master_err = ""
        denied_since = None
        while loop.time() < end:
            if "Permission denied" in master_err:
                # Some hosts never install the account key (seen repeatedly on Vast). Two minutes of refusals on an
                # otherwise reachable sshd is a host fault, not a slow boot.
                denied_since = denied_since or loop.time()
                if loop.time() - denied_since > float(os.environ.get("GF_SSH_DENIED_S", 120)):
                    break
            if self.master is None or self.master.returncode is not None:
                if self.master is not None and self.master.stderr:
                    master_err = (await self.master.stderr.read()).decode(errors="replace").strip()[-300:] \
                        or master_err
                # ssh keeps the FIRST value of an option, so the master's own settings go before self.opts. With
                # ControlPersist the master would fork into the background and look dead to us.
                self.master = await asyncio.create_subprocess_exec(
                    "ssh", "-o", "ControlMaster=yes", "-o", "ControlPersist=no", *self.opts, "-N", self.dest,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
            await asyncio.sleep(1.0)
            if self.master.returncode is not None:
                await asyncio.sleep(3)
                continue
            # Only ever ask the master's control socket: a plain probe connection would be a second SSH
            # connection, which Colab refuses (one per runtime) and which can knock the master over.
            rc, out = await _run(["ssh", *self.opts, "-O", "check", self.dest], 10)
            if rc == 0:
                rc, out = await _run(["ssh", *self.opts, self.dest, "true"], 30)
                if rc == 0:
                    return
            last = out.strip()[-300:] or last
        if self.master and self.master.returncode is None:
            self.master.kill()
        raise TransportError(f"ssh not reachable within {timeout:.0f}s: {master_err or last}", rc=255)

    async def close(self) -> None:
        if self.master and self.master.returncode is None:
            await _run(["ssh", *self.opts, "-O", "exit", self.dest], 10)
            try:
                await asyncio.wait_for(self.master.wait(), 10)
            except asyncio.TimeoutError:
                self.master.kill()
        self.master = None

    @property
    def alive(self) -> bool:
        return self.master is not None and self.master.returncode is None

    async def exec(self, cmd: str, timeout: float = 600, stdin: bytes | None = None) -> tuple[int, str]:
        async with self._sessions:
            return await _run([*self._ssh(), self.dest, cmd], timeout, stdin)

    async def spawn(self, cmd: str) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(*self._ssh(), self.dest, cmd, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL, limit=1 << 22)

    def _rsync(self, s: Sync) -> list[str]:
        e = " ".join(shlex.quote(x) for x in self._ssh())
        argv = ["rsync", "-a", "--partial", "--inplace", "--no-owner", "--no-group", "-e", e, *s.filters()]
        if s.include:
            argv.append("--prune-empty-dirs")
        if s.delete:
            argv.append("--delete")
        return argv

    async def push(self, s: Sync, timeout: float = 3600) -> None:
        src = os.path.expanduser(s.src)
        dst = self.abspath(s.dst)
        trail = "/" if os.path.isdir(src) else ""
        parent = dst.rstrip("/") if trail else os.path.dirname(dst)
        rc, out = await self.exec(f"mkdir -p {shlex.quote(parent)}", 60)
        if rc:
            raise TransportError(f"mkdir {parent}: {out[-300:]}", rc)
        async with self._sessions:
            rc, out = await _run([*self._rsync(s), src.rstrip("/") + trail, f"{self.dest}:{dst.rstrip('/') + trail}"],
                                 timeout)
        if rc:
            raise TransportError(f"push {s.src} -> {dst} failed rc={rc}: {out[-400:]}", rc)

    async def pull(self, s: Sync, timeout: float = 3600) -> None:
        src = self.abspath(s.src)
        dst = os.path.expanduser(s.dst)
        os.makedirs(dst if src.endswith("/") else os.path.dirname(dst) or ".", exist_ok=True)
        async with self._sessions:
            rc, out = await _run([*self._rsync(s), f"{self.dest}:{src}", dst], timeout)
        if rc == 23 and ("No such file" in out or "vanished" in out):
            return  # nothing to pull for this task (e.g. no logs written)
        if rc:
            raise TransportError(f"pull {src} -> {dst} failed rc={rc}: {out[-400:]}", rc)


class LocalTransport(Transport):
    """A box that is just a directory on this machine. Used by tests and the fake provider."""

    def __init__(self, box_id: str, base: Path | None = None):
        self.base = Path(base or tempfile.mkdtemp(prefix=f"gf-{box_id}-"))
        self.root = str(self.base / "work")
        Path(self.root).mkdir(parents=True, exist_ok=True)
        self.alive = True

    async def open(self, timeout: float = 120) -> None:
        if not self.alive:
            raise TransportError("box is gone", rc=255)

    async def close(self) -> None:
        pass

    async def exec(self, cmd: str, timeout: float = 600, stdin: bytes | None = None) -> tuple[int, str]:
        if not self.alive:
            return 255, "box is gone"
        return await _run(["bash", "-c", cmd], timeout, stdin)

    async def spawn(self, cmd: str) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec("bash", "-c", cmd, stdin=asyncio.subprocess.PIPE,
                                                    stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.DEVNULL, limit=1 << 22)

    async def push(self, s: Sync, timeout: float = 3600) -> None:
        src = os.path.expanduser(s.src)
        dst = self.abspath(s.dst)
        trail = "/" if os.path.isdir(src) else ""
        os.makedirs(dst if trail else os.path.dirname(dst), exist_ok=True)
        argv = ["rsync", "-a", *s.filters(), *(["--delete"] if s.delete else []),
                src.rstrip("/") + trail, dst.rstrip("/") + trail]
        rc, out = await _run(argv, timeout)
        if rc:
            raise TransportError(f"push failed: {out}", rc)

    async def pull(self, s: Sync, timeout: float = 3600) -> None:
        src = self.abspath(s.src)
        dst = os.path.expanduser(s.dst)
        os.makedirs(dst if src.endswith("/") else os.path.dirname(dst) or ".", exist_ok=True)
        rc, out = await _run(["rsync", "-a", *s.filters(), src, dst], timeout)
        if rc == 23:
            return
        if rc:
            raise TransportError(f"pull failed: {out}", rc)
