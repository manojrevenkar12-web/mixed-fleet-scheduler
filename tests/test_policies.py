from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mfs import config
from mfs.lab.trace import analyse
from mfs.platform.optimize import solve_counts
from mfs.platform.scheduler import N_FEATURES, backfill_features
from mfs.sim.designs import run_design
from mfs.sim.evaluate import capacity_breakdown, run_metrics, twin_drift
from mfs.world.world import Site


@dataclass
class Cand:
    gpu: str
    tp: int
    capacity_rps: float


def test_the_optimiser_meets_need_within_the_gpus() -> None:
    cands = {"a": [Cand("X", 1, 10.0), Cand("Y", 2, 30.0)], "b": [Cand("Y", 1, 5.0)]}
    out = solve_counts(cands, {"a": 55.0, "b": 9.0}, {"X": 2, "Y": 6}, {}, max_reps=8)
    assert out is not None
    used = {"X": 0, "Y": 0}
    for (m, g), n in out.items():
        used[g] += n * next(c.tp for c in cands[m] if c.gpu == g)
    assert used["X"] <= 2 and used["Y"] <= 6
    assert sum(n * next(c.capacity_rps for c in cands["a"] if c.gpu == g) for (m, g), n in out.items() if m == "a") >= 55.0
    assert sum(n * 5.0 for (m, _), n in out.items() if m == "b") >= 9.0


def test_the_optimiser_keeps_running_replicas_when_starts_cost_something() -> None:
    cands = {"a": [Cand("X", 1, 10.0), Cand("Y", 1, 10.0)]}
    out = solve_counts(cands, {"a": 20.0}, {"X": 4, "Y": 4}, {("a", "Y"): 2}, max_reps=8, w_start=3.0)
    assert out == {("a", "Y"): 2}


def test_shortfall_is_allowed_but_expensive() -> None:
    out = solve_counts({"a": [Cand("X", 1, 10.0)]}, {"a": 1000.0}, {"X": 3}, {}, max_reps=8)
    assert out == {("a", "X"): 3}


def test_ilp_placement_stays_within_the_fleet(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    d = config.designs()["designs"]["platform_ilp"]
    run = run_design(tiny, "platform_ilp", d, fast_cfg, keep_placements=True)
    for p in run.placements:
        for gpu, used in p.gpus_used().items():
            assert used <= tiny.fleet[gpu]
    assert run_metrics(tiny, run)["attainment"] > 0.8


def test_backfill_features_have_a_fixed_length() -> None:
    assert len(backfill_features(10, 8, 4, 48, 30, 20, True, 13)) == N_FEATURES


def test_learned_weights_decide_starts(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    d = config.designs()["designs"]["platform"]
    never = run_design(tiny, "never", d, fast_cfg, backfill_weights=[-1.0] + [0.0] * (N_FEATURES - 1))
    always = run_design(tiny, "always", d, fast_cfg, backfill_weights=[1.0] + [0.0] * (N_FEATURES - 1))
    assert never.train_net_tokens == 0
    assert always.train_net_tokens > 0


def test_capacity_breakdown_adds_up(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    run = run_design(tiny, "platform", config.designs()["designs"]["platform"], fast_cfg)
    shares = capacity_breakdown(tiny, run)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(v >= -1e-9 for v in shares.values())


def test_the_drift_monitor_sees_the_driver_change(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    run = run_design(tiny, "platform", config.designs()["designs"]["platform"], fast_cfg)
    t = twin_drift(tiny, run)
    assert t is not None and t["drift_gpu"] == "L40S"
    assert t["twin_error_after"] > t["twin_error_before"]


def test_a_trace_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text(
        "# fleet: A=4; hours: 4; test\n"
        "JobID,Kind,Partition,AllocGPUs,Submit,Start,End,State\n"
        "svc-1,serve,A,2,2026-11-02T00:00:00,2026-11-02T00:00:00,2026-11-02T04:00:00,COMPLETED\n"
        "ft-1.0,train,A,2,2026-11-02T00:00:00,2026-11-02T01:00:00,2026-11-02T02:00:00,PREEMPTED\n"
        "ft-2.0,train,,2,2026-11-02T00:00:00,,,PENDING\n",
        encoding="utf-8",
    )
    a = analyse(p, checkpoint_hours=0.5)
    assert a["gpu_hours"] == 16
    assert a["serving_allocated"] == pytest.approx(8 / 16)
    assert a["training_lost_to_preemption"] == pytest.approx(1 / 16)
    total = a["serving_allocated"] + a["training_kept"] + a["training_lost_to_preemption"] + a["stranded_jobs_waiting"] + a["idle_nothing_queued"]
    assert total == pytest.approx(1.0)
    assert a["preemptions"] == 1


def test_an_elastic_job_shards_wider_where_it_does_not_fit(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    cfg = config.merged(fast_cfg, {"training": {"elastic_sizes": [1, 2, 0.5]}})
    run = run_design(tiny, "elastic", config.designs()["designs"]["platform"], cfg, keep_placements=True)
    jobs = {j.id: j for j in tiny.training_jobs()}
    sizes = {a.gpus / jobs[a.job].gpus for p in run.placements for a in p.training}
    assert sizes <= {0.5, 1.0, 2.0}
    assert run_metrics(tiny, run)["training_crashes"] == 0
