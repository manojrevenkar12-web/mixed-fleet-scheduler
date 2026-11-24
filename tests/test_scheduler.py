from __future__ import annotations

from typing import Any

from mfs.domain import Observation
from mfs.platform.build import build_platform
from mfs.world.world import Site


def _demand(site: Site, h: int) -> dict[str, float]:
    return {m: float(site.demand[h, j]) for j, m in enumerate(site.models)}


def test_a_placement_respects_the_fleet_and_the_slo(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed)
    slack = float(fast_cfg["guardrails"]["tpot_slack"])
    for h in range(0, 24, 4):
        p = sched.plan(h, _demand(tiny, h), _demand(tiny, h))
        for gpu, used in p.gpus_used().items():
            assert used <= tiny.fleet[gpu]
        for r in p.replicas:
            assert r.step_ms <= tiny.services[r.shape.model].tpot_ms * slack + 1e-9
            assert r.admitted_rps <= float(fast_cfg["guardrails"]["admission_share"]) * r.capacity_rps + 1e-9


def test_overload_is_shed_explicitly(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed)
    huge = {m: 1e5 for m in tiny.models}
    p = sched.plan(0, huge, huge)
    for m in tiny.models:
        admitted = sum(r.admitted_rps for r in p.replicas if r.shape.model == m)
        assert abs(admitted + p.shed_rps[m] - 1e5) < 1e-6
        assert p.shed_rps[m] > 0


def test_steady_traffic_does_not_restart_replicas(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed)
    d = _demand(tiny, 10)
    sched.plan(0, d, d)
    starts = sched.starts
    for h in range(1, 5):
        sched.plan(h, d, d)
    assert sched.starts == starts


def test_replicas_that_all_slow_down_mark_their_gpu_as_drifting(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed)
    d = _demand(tiny, 10)
    p = sched.plan(0, d, d)
    obs = []
    for r in p.replicas:
        mean = sched.est.step(r.shape.model, r.shape.gpu, r.shape.tp, r.shape.batch)[0]
        slow = 1.3 if r.shape.gpu == "L40S" else 1.0
        obs.append(Observation(r.shape, mean * 1000 * slow, False))
    assert any(r.shape.gpu == "L40S" for r in p.replicas)
    sched.observe(0, obs)
    assert "L40S" in sched.drift_until
    assert "H100" not in sched.drift_until
    p2 = sched.plan(1, d, d)
    assert p2.notes["drifting"] == ["L40S"]


def test_feedback_off_learns_nothing(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    sched, _ = build_platform(tiny.services, tiny.fleet, tiny.profiles(), fast_cfg, seed=tiny.seed, without=["feedback"])
    d = _demand(tiny, 10)
    p = sched.plan(0, d, d)
    sched.observe(0, [Observation(r.shape, 999.0, False) for r in p.replicas])
    assert sched.drift_until == {}
