"""Measures how long the platform takes to train and to make one hourly placement, on this machine.
Measured, not simulated: the numbers are wall-clock times of the real code."""

from __future__ import annotations

import json
import platform as _platform
import statistics
import time

import torch

from .. import config
from ..platform.build import build_platform
from ..world.world import build_site


def main() -> int:
    seed = config.seeds()["dev"][0]
    site = build_site(seed, "dev")
    cfg = config.platform()
    t0 = time.perf_counter()
    from ..platform import perfmodel

    perfmodel._FITTED.clear()
    sched, train = build_platform(site.services, site.fleet, site.profiles(), cfg, seed=seed)
    train_s = time.perf_counter() - t0
    times = []
    for h in range(48):
        demand = {m: float(site.demand[h, j]) for j, m in enumerate(site.models)}
        t = time.perf_counter()
        sched.plan(h, demand, demand)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    out = {
        "measured": True,
        "machine": {"cpu": _platform.processor() or _platform.machine(), "threads": torch.get_num_threads(), "torch": torch.__version__.split("+")[0]},
        "site": {"gpus": sum(site.fleet.values()), "gpu_types": len(site.fleet), "services": len(site.services), "profiles": len(site.profiles())},
        "train_s": round(train_s, 2),
        "train_points": train["points"],
        "plan_ms_median": round(statistics.median(times), 2),
        "plan_ms_p95": round(times[int(0.95 * (len(times) - 1))], 2),
        "plans": len(times),
    }
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / "bench.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out))
    return 0
