"""Job traces in the shape a Slurm `sacct` export has, and the capacity-loss analysis that runs on them.

`export` writes one simulated site-week as a trace; `analyse` reads any trace with these columns (a real
`sacct --parsable2 --format=JobID,Partition,AllocTRES,Submit,Start,End,State` export maps onto them) and answers:
where did the GPU-hours go, and how many were lost while work was waiting?
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .. import config
from ..sim.designs import run_design
from ..world.world import build_site

BASE = datetime(2026, 11, 2, tzinfo=UTC)
COLUMNS = ["JobID", "Kind", "Partition", "AllocGPUs", "Submit", "Start", "End", "State"]


def _ts(h: float) -> str:
    return (BASE + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S")


def _h(s: str) -> float:
    return (datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC) - BASE).total_seconds() / 3600.0


def export(seed: int, out: Path, variant: str = "dev", design: str = "platform") -> int:
    site = build_site(seed, variant)
    d = config.designs()["designs"][design]
    run = run_design(site, design, d, config.platform(), keep_placements=True)
    rows: list[dict[str, Any]] = []
    H = len(run.placements)
    # serving: contiguous spans of a constant GPU count per (service, GPU type)
    spans: dict[tuple[str, str], tuple[int, int]] = {}  # key -> (start hour, gpus)
    n = 0
    for h in range(H + 1):
        now: dict[tuple[str, str], int] = {}
        if h < H:
            for r in run.placements[h].replicas:
                k = (r.shape.model, r.shape.gpu)
                now[k] = now.get(k, 0) + r.shape.tp
        for k in set(spans) | set(now):
            cur = spans.get(k)
            if cur and (k not in now or now[k] != cur[1]):
                n += 1
                rows.append(
                    {
                        "JobID": f"svc-{k[0]}-{n}",
                        "Kind": "serve",
                        "Partition": k[1],
                        "AllocGPUs": cur[1],
                        "Submit": _ts(cur[0]),
                        "Start": _ts(cur[0]),
                        "End": _ts(h),
                        "State": "COMPLETED" if h < H else "RUNNING",
                    }
                )
                spans.pop(k)
            if k in now and k not in spans:
                spans[k] = (h, now[k])
    # fine-tuning: one row per running segment
    jobs = {j.id: j for j in site.training_jobs()}
    seg: dict[int, tuple[int, str, int]] = {}
    queued_since = {j: info["arrival"] for j, info in run.jobs.items()}
    for h in range(H + 1):
        running = {a.job: a for a in run.placements[h].training} if h < H else {}
        preempted = set(run.placements[h].preempted) if h < H else set()
        for jid in list(seg):
            if jid in running:
                continue
            start, gpu, gpus = seg.pop(jid)
            done = run.jobs[jid]["done_at"]
            if done is not None and done <= h:
                state, end = "COMPLETED", done
            elif jid in preempted:
                state, end = "PREEMPTED", float(h)
                queued_since[jid] = h
            elif h >= H:
                state, end = "RUNNING", float(H)
            else:
                state, end = "OUT_OF_MEMORY", float(h)
                queued_since[jid] = h
            rows.append({"JobID": f"ft-{jid}.{start}", "Kind": "train", "Partition": gpu, "AllocGPUs": gpus, "Submit": _ts(queued_since[jid]), "Start": _ts(start), "End": _ts(end), "State": state})
        for jid, a in running.items():
            if jid not in seg:
                seg[jid] = (h, a.gpu, a.gpus)
    for jid, info in run.jobs.items():  # never started
        if not any(r["JobID"].startswith(f"ft-{jid}.") for r in rows) and info["done_at"] is None:
            rows.append({"JobID": f"ft-{jid}.0", "Kind": "train", "Partition": "", "AllocGPUs": jobs[jid].gpus, "Submit": _ts(info["arrival"]), "Start": "", "End": "", "State": "PENDING"})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        f.write("# fleet: " + ",".join(f"{g}={c}" for g, c in site.fleet.items()) + f"; hours: {H}; simulated site {variant}/{seed}, design {design}\n")
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["Start"] or "~", r["JobID"])))
    return len(rows)


def analyse(path: Path, checkpoint_hours: float = 0.25) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").splitlines()
    head = text[0]
    if not head.startswith("# fleet:"):
        raise ValueError("first line must be '# fleet: TYPE=COUNT,...; hours: N'")
    fleet = {k: int(v) for k, v in (x.split("=") for x in head[len("# fleet:") :].split(";")[0].strip().split(","))}
    hours = int(head.split("hours:")[1].split(";")[0])
    rows = list(csv.DictReader(text[1:]))
    total = sum(fleet.values()) * hours
    serve = train = lost = 0.0
    used = [0.0] * hours
    pending = [0.0] * hours
    for r in rows:
        g = int(r["AllocGPUs"])
        sub = _h(r["Submit"])
        if r["Start"]:
            s, e = _h(r["Start"]), _h(r["End"]) if r["End"] else float(hours)
            for h in range(int(s), min(hours, int(-(-e // 1)))):
                frac = min(e, h + 1) - max(s, h)
                used[h] += g * frac
            if r["Kind"] == "serve":
                serve += g * (e - s)
            else:
                train += g * (e - s)
                if r["State"] == "PREEMPTED":
                    lost += g * min(checkpoint_hours, e - s)
            waits = (sub, s)
        else:
            waits = (sub, float(hours))
        if r["Kind"] == "train":
            for h in range(int(waits[0]), min(hours, int(waits[1]))):
                pending[h] += g
    fl = sum(fleet.values())
    stranded = sum(min(max(0.0, fl - used[h]), pending[h]) for h in range(hours))
    idle = sum(max(0.0, fl - used[h]) for h in range(hours)) - stranded
    return {
        "gpu_hours": total,
        "serving_allocated": serve / total,
        "training_kept": (train - lost) / total,
        "training_lost_to_preemption": lost / total,
        "stranded_jobs_waiting": stranded / total,
        "idle_nothing_queued": idle / total,
        "jobs": sum(1 for r in rows if r["Kind"] == "train"),
        "preemptions": sum(1 for r in rows if r["State"] == "PREEMPTED"),
    }
