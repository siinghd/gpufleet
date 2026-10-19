"""SQLite state: the single source of truth for boxes, jobs, tasks, spend and learned performance.

Only the daemon writes. Every box row is inserted *before* the provider is asked to create anything (status
'requested'), so a crash between "rent" and "record" cannot leave an untracked box; the reconciler also matches
provider instances by label prefix as a second net.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS boxes (
  id TEXT PRIMARY KEY,            -- our id, also the label suffix
  provider TEXT NOT NULL,
  ext_id TEXT,                    -- provider instance id / session name
  label TEXT NOT NULL,
  status TEXT NOT NULL,           -- requested, booting, ready, draining, parked, unparking, destroying, destroyed
  offer TEXT,                     -- JSON of the offer it was rented from
  profile TEXT,
  gpu_name TEXT, num_gpus INTEGER, price_h REAL, prepaid INTEGER DEFAULT 0, spot INTEGER DEFAULT 0,
  machine_id TEXT, host_id TEXT,
  ssh TEXT,                       -- JSON SshTarget once known
  created_at REAL NOT NULL, ready_at REAL, last_busy REAL, destroyed_at REAL,
  hard_deadline REAL NOT NULL,    -- the reconciler destroys the box after this, whatever it is doing
  note TEXT
);
CREATE INDEX IF NOT EXISTS boxes_status ON boxes(status);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, project TEXT NOT NULL, name TEXT NOT NULL, spec TEXT NOT NULL,
  status TEXT NOT NULL,           -- queued, running, done, failed, cancelled
  created_at REAL NOT NULL, finished_at REAL, cost REAL DEFAULT 0,
  cursor INTEGER,                 -- next unleased unit for range jobs
  error TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
  unit TEXT,                      -- named unit, or NULL for range chunks
  lo INTEGER, hi INTEGER,         -- inclusive range for chunks
  status TEXT NOT NULL,           -- queued, running, pulling, done, failed, lost, cancelled
  box_id TEXT, attempt INTEGER DEFAULT 0, est_s REAL,
  started_at REAL, finished_at REAL, rc INTEGER, tail TEXT,
  dup_of TEXT,                    -- speculative copy of another task
  excluded TEXT DEFAULT '[]'      -- box ids / machine ids this task must not run on again
);
CREATE INDEX IF NOT EXISTS tasks_job ON tasks(job_id, status);

CREATE TABLE IF NOT EXISTS spend (
  day TEXT NOT NULL, provider TEXT NOT NULL, usd REAL NOT NULL DEFAULT 0, units REAL NOT NULL DEFAULT 0,
  PRIMARY KEY (day, provider)
);

CREATE TABLE IF NOT EXISTS perf (
  gpu TEXT NOT NULL, workload TEXT NOT NULL,
  secs_per_unit REAL NOT NULL, n INTEGER NOT NULL,
  PRIMARY KEY (gpu, workload)
);

CREATE TABLE IF NOT EXISTS machines (
  provider TEXT NOT NULL, machine_id TEXT NOT NULL, host_id TEXT,
  at REAL NOT NULL, kind TEXT NOT NULL,   -- boot_fail, gpu_missing, slow_net, task_crash, ok
  note TEXT
);
CREATE INDEX IF NOT EXISTS machines_id ON machines(provider, machine_id);

CREATE TABLE IF NOT EXISTS boot_stats (
  provider TEXT NOT NULL, at REAL NOT NULL, secs REAL NOT NULL, ok INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  at REAL NOT NULL, kind TEXT NOT NULL, ref TEXT, data TEXT
);
"""

ALIVE = ("requested", "booting", "ready", "draining", "destroying", "parked", "unparking")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    # -- generic -------------------------------------------------------------------------------------------------
    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, tuple(args)).fetchall()

    def one(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Row | None:
        return self.db.execute(sql, tuple(args)).fetchone()

    def x(self, sql: str, args: Iterable[Any] = ()) -> int:
        return self.db.execute(sql, tuple(args)).rowcount

    def update(self, table: str, key: str, **cols: Any) -> None:
        if not cols:
            return
        sets = ", ".join(f"{c}=?" for c in cols)
        self.x(f"UPDATE {table} SET {sets} WHERE id=?", [*cols.values(), key])

    def event(self, kind: str, ref: str = "", **data: Any) -> None:
        self.x("INSERT INTO events VALUES (?,?,?,?)", [time.time(), kind, ref, json.dumps(data) if data else None])

    # -- boxes ---------------------------------------------------------------------------------------------------
    def alive_boxes(self, provider: str | None = None) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(ALIVE))
        if provider:
            return self.q(f"SELECT * FROM boxes WHERE status IN ({marks}) AND provider=?", [*ALIVE, provider])
        return self.q(f"SELECT * FROM boxes WHERE status IN ({marks})", ALIVE)

    # -- spend ---------------------------------------------------------------------------------------------------
    @staticmethod
    def today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def add_spend(self, provider: str, usd: float, units: float = 0) -> None:
        self.x("INSERT INTO spend VALUES (?,?,?,?) ON CONFLICT(day, provider) DO UPDATE SET "
               "usd=usd+excluded.usd, units=units+excluded.units", [self.today(), provider, usd, units])

    def spend_today(self, provider: str | None = None) -> float:
        if provider:
            r = self.one("SELECT usd FROM spend WHERE day=? AND provider=?", [self.today(), provider])
        else:
            r = self.one("SELECT SUM(usd) AS usd FROM spend WHERE day=?", [self.today()])
        return float(r["usd"] or 0) if r else 0.0

    # -- perf ----------------------------------------------------------------------------------------------------
    def learn(self, gpu: str, workload: str, secs_per_unit: float, alpha: float = 0.3) -> None:
        """Exponential moving average of seconds per work unit for this GPU model and workload."""
        r = self.one("SELECT secs_per_unit, n FROM perf WHERE gpu=? AND workload=?", [gpu, workload])
        if r is None:
            self.x("INSERT INTO perf VALUES (?,?,?,1)", [gpu, workload, secs_per_unit])
        else:
            a = max(alpha, 1 / (r["n"] + 1))  # plain mean for the first few samples, then EMA
            self.x("UPDATE perf SET secs_per_unit=?, n=n+1 WHERE gpu=? AND workload=?",
                   [r["secs_per_unit"] * (1 - a) + secs_per_unit * a, gpu, workload])

    def learned(self, gpu: str, workload: str) -> tuple[float, int] | None:
        r = self.one("SELECT secs_per_unit, n FROM perf WHERE gpu=? AND workload=?", [gpu, workload])
        return (r["secs_per_unit"], r["n"]) if r else None

    # -- machine ledger ------------------------------------------------------------------------------------------
    def mark_machine(self, provider: str, machine_id: str, host_id: str | None, kind: str, note: str = "") -> None:
        if machine_id:
            self.x("INSERT INTO machines VALUES (?,?,?,?,?,?)", [provider, machine_id, host_id, time.time(), kind, note])

    def bad_machines(self, provider: str, days: float = 14, strikes: int = 2) -> tuple[set[str], set[str]]:
        """Machines with `strikes` failures in the window, and hosts with three bad machines (a bad operator)."""
        since = time.time() - days * 86400
        rows = self.q("SELECT machine_id, host_id, COUNT(*) AS n FROM machines WHERE provider=? AND at>? "
                      "AND kind!='ok' GROUP BY machine_id", [provider, since])
        bad = {r["machine_id"] for r in rows if r["n"] >= strikes}
        per_host: dict[str, int] = {}
        for r in rows:
            if r["machine_id"] in bad and r["host_id"]:
                per_host[r["host_id"]] = per_host.get(r["host_id"], 0) + 1
        return bad, {h for h, n in per_host.items() if n >= 3}

    # -- boot stats ----------------------------------------------------------------------------------------------
    def boot_estimate(self, provider: str, default: float) -> tuple[float, float]:
        """(median boot seconds of successful boots, success rate) over the last 30 boots, with a prior."""
        rows = self.q("SELECT secs, ok FROM boot_stats WHERE provider=? ORDER BY at DESC LIMIT 30", [provider])
        oks = sorted(r["secs"] for r in rows if r["ok"])
        med = oks[len(oks) // 2] if oks else default
        rate = (sum(r["ok"] for r in rows) + 9) / (len(rows) + 10)  # prior: 9 of 10 boots work
        return med, rate
