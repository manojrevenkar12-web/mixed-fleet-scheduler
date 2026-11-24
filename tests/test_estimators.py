from __future__ import annotations

import math
from typing import Any

from mfs.platform.estimators import Calibration, Roofline, Uniform, planned_fit
from mfs.world.world import Site


def test_roofline_is_memory_bound_at_small_batches(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    r = Roofline(fast_cfg, tiny.services)
    t1 = r.step("llama-8b", "H100", 1, 1)[0]
    t8 = r.step("llama-8b", "H100", 1, 8)[0]
    assert t8 < 1.3 * t1


def test_uniform_sees_every_gpu_as_an_a100(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    u = Uniform(fast_cfg, tiny.services)
    assert u.step("llama-8b", "H100", 1, 16)[0] == u.step("llama-8b", "L40S", 1, 16)[0]


def test_planned_fit_keeps_a_memory_reserve(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    svc = tiny.services["llama-70b"]
    assert not planned_fit("llama-70b", "H100", 1, 1, svc, fast_cfg)
    assert planned_fit("llama-70b", "H100", 4, 8, svc, fast_cfg)


def test_calibration_learns_per_batch_bucket_then_falls_back() -> None:
    c = Calibration(0.5)
    c.update("llama-8b", "A100", 1, 32, math.log(1.2))
    assert c.factor("llama-8b", "A100", 1, 32) > 1.09
    assert c.factor("llama-8b", "A100", 1, 64) == c.factor("llama-8b", "A100", 1, 32)  # neighbouring bucket
    assert 1.0 < c.factor("llama-8b", "A100", 1, 2) < c.factor("llama-8b", "A100", 1, 32)  # pair, half the step
    assert c.factor("llama-8b", "H100", 1, 32) == 1.0


def test_calibration_converges_on_a_constant_ratio() -> None:
    c = Calibration(0.5)
    for _ in range(12):
        pred = c.factor("m", "g", 1, 16)
        c.update("m", "g", 1, 16, math.log(1.3 / pred))
    assert abs(c.factor("m", "g", 1, 16) - 1.3) < 0.01
