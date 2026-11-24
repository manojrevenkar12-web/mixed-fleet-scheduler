"""Placement as an integer program, solved each hour with HiGHS (scipy.optimize.milp).

Variables, per service m and GPU type g with a feasible shape (tensor-parallel degree, batch) for this hour:
  n[m,g]  replicas (integer)          s[m]  share of m's need left unserved (0..1)
  u[m,g]  replicas started this hour (integer, >= n - kept)
Minimise  W_short * sum s  +  sum n*tp (GPUs used, so the rest can go to fine-tuning)  +  W_start * sum u
Subject to  sum_m n[m,g]*tp[m,g] <= fleet[g]          (GPUs of each type)
            sum_g n[m,g]*cap[m,g] + s[m]*need[m] >= need[m]   (capacity for the forecast, with headroom)
            sum_g n[m,g] <= max replicas per service

The greedy planner fills services one at a time; the program sees all services and GPU types at once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_counts(
    cands: dict[str, list[Any]],
    need: dict[str, float],
    fleet: dict[str, int],
    kept: dict[tuple[str, str], int],
    max_reps: int,
    w_short: float = 1000.0,
    w_start: float = 0.5,
    time_limit_s: float = 2.0,
) -> dict[tuple[str, str], int] | None:
    keys = [(m, c.gpu) for m in cands for c in cands[m]]
    if not keys:
        return {}
    shape = {(m, c.gpu): c for m in cands for c in cands[m]}
    services = list(cands)
    nk, ns = len(keys), len(services)
    # variable order: n (nk), u (nk), s (ns)
    nv = 2 * nk + ns
    c = np.zeros(nv)
    for i, k in enumerate(keys):
        c[i] = shape[k].tp
        c[nk + i] = w_start
    c[2 * nk :] = w_short
    rows, lo, hi = [], [], []
    gpus = sorted(fleet)
    for g in gpus:
        r = np.zeros(nv)
        for i, (m, gg) in enumerate(keys):
            if gg == g:
                r[i] = shape[(m, gg)].tp
        rows.append(r)
        lo.append(-np.inf)
        hi.append(float(fleet[g]))
    for j, m in enumerate(services):
        if need[m] <= 0:
            continue
        r = np.zeros(nv)
        for i, (mm, _) in enumerate(keys):
            if mm == m:
                r[i] = shape[keys[i]].capacity_rps
        r[2 * nk + j] = need[m]
        rows.append(r)
        lo.append(need[m])
        hi.append(np.inf)
        r2 = np.zeros(nv)
        for i, (mm, _) in enumerate(keys):
            if mm == m:
                r2[i] = 1.0
        rows.append(r2)
        lo.append(-np.inf)
        hi.append(float(max_reps))
    for i, k in enumerate(keys):  # u >= n - kept
        r = np.zeros(nv)
        r[nk + i] = 1.0
        r[i] = -1.0
        rows.append(r)
        lo.append(-float(kept.get(k, 0)))
        hi.append(np.inf)
    integrality = np.concatenate([np.ones(2 * nk), np.zeros(ns)])
    ub = np.concatenate([[fleet[g] // max(shape[(m, g)].tp, 1) for m, g in keys], [fleet[g] for _, g in keys], np.ones(ns)])
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality, bounds=Bounds(np.zeros(nv), ub), options={"time_limit": time_limit_s})
    if res.x is None:
        return None
    return {k: int(round(res.x[i])) for i, k in enumerate(keys) if round(res.x[i]) > 0}
