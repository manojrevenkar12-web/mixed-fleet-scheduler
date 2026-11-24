"""Measure decode-like kernels on the machine at hand and turn them into the scheduler's profile records.

There is no GPU where this repository is built, so this measures the CPU it runs on, with the same method a
site would use on a GPU node: time a decode step's dominant operation (weights x activations for `batch`
sequences) across batch sizes with torch.profiler, then fit how close the machine gets to its memory-bound
and compute-bound limits. The result is labelled measured, with the machine it came from.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from .. import config

OUT = config.REPORTS / "host_profile.json"


def _cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def time_step(hidden: int, batch: int, layers: int = 4, repeats: int = 5) -> dict[str, float]:
    """A decode step's matmuls for `layers` layers: [batch, hidden] x [hidden, 4*hidden] and back, BF16 weights
    upcast like a CPU kernel would. Returns median wall time and the profiler's matmul share."""
    torch.manual_seed(0)
    w1 = [torch.randn(hidden, 4 * hidden) for _ in range(layers)]
    w2 = [torch.randn(4 * hidden, hidden) for _ in range(layers)]
    x = torch.randn(batch, hidden)

    def step() -> torch.Tensor:
        h = x
        for a, b in zip(w1, w2, strict=True):
            h = torch.nn.functional.silu(h @ a) @ b
        return h

    step()
    walls = []
    for _ in range(repeats):
        t = time.perf_counter()
        step()
        walls.append(time.perf_counter() - t)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        step()
    total = sum(e.self_cpu_time_total for e in prof.key_averages())
    mm = sum(e.self_cpu_time_total for e in prof.key_averages() if e.key in ("aten::mm", "aten::addmm", "aten::matmul"))
    wall = statistics.median(walls)
    bytes_moved = layers * 2 * hidden * 4 * hidden * 4  # fp32 weights read once
    flops = layers * 2 * 2 * batch * hidden * 4 * hidden
    return {"hidden": hidden, "batch": batch, "wall_ms": wall * 1000, "gb_s": bytes_moved / wall / 1e9, "gflop_s": flops / wall / 1e9, "matmul_share": mm / total if total else 0.0}


def main() -> int:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    rows = [time_step(h, b) for h in (1024, 2048) for b in (1, 4, 16, 64, 256)]
    peak_bw = max(r["gb_s"] for r in rows if r["batch"] == 1)
    peak_flops = max(r["gflop_s"] for r in rows)
    # where the measured step stops being memory-bound: the ridge the scheduler's roofline needs
    ridge = next((r["batch"] for r in rows if r["hidden"] == 2048 and r["gflop_s"] > 0.6 * peak_flops), None)
    out: dict[str, Any] = {
        "measured": True,
        "machine": {"cpu": _cpu_name(), "threads": torch.get_num_threads(), "torch": torch.__version__, "python": platform.python_version()},
        "rows": [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in rows],
        "fitted": {"memory_gb_s": round(peak_bw, 2), "compute_gflop_s": round(peak_flops, 2), "memory_bound_until_batch": ridge},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(config.ROOT)}: {peak_bw:.1f} GB/s, {peak_flops:.1f} GFLOP/s on {out['machine']['cpu']}")
    return 0
