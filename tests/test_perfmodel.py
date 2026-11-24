from __future__ import annotations

import math
from typing import Any

import numpy as np

from mfs import config
from mfs.platform.estimators import Calibration, roofline_parts
from mfs.platform.perfmodel import Learned, PerfModel, step_ratios
from mfs.world.world import Site


def _fit(tiny: Site, cfg: dict[str, Any]) -> PerfModel:
    pm = PerfModel(cfg, tiny.services, seed=tiny.seed)
    pm.fit(tiny.profiles())
    return pm


def test_learned_model_beats_the_roofline_on_measured_pairs(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    pm = _fit(tiny, fast_cfg)
    ratios = step_ratios(tiny.profiles(), fast_cfg, tiny.services)
    err_l, err_r = [], []
    for p in tiny.profiles():
        truth = tiny.step_s(p.model, p.gpu, p.tp, p.batch)
        mean, _ = pm.predict_log(p.model, p.gpu, p.tp, p.batch)
        roof = roofline_parts(p.model, p.gpu, p.tp, p.batch, tiny.services[p.model], fast_cfg)[2] * math.exp(ratios[p.gpu])
        err_l.append(abs(math.exp(mean) / truth - 1))
        err_r.append(abs(roof / truth - 1))
    assert np.mean(err_l) < 0.06
    assert np.mean(err_l) < np.mean(err_r)


def test_training_is_reproducible(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    from mfs.platform import perfmodel

    a = _fit(tiny, fast_cfg).predict_log("llama-8b", "L40S", 1, 16)
    perfmodel._FITTED.clear()
    b = _fit(tiny, fast_cfg).predict_log("llama-8b", "L40S", 1, 16)
    assert a == b


def test_the_model_is_less_sure_on_a_gpu_type_it_never_measured(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    pm = PerfModel(fast_cfg, tiny.services, seed=tiny.seed)
    pm.fit([p for p in tiny.profiles() if p.gpu != "MI300X"])
    grid = [(m, tp, b) for m in tiny.models for tp in (2, 4, 8) for b in (4, 16, 64)]
    seen = np.mean([pm.predict_log(m, "H100", tp, b)[1] for m, tp, b in grid])
    unseen = np.mean([pm.predict_log(m, "MI300X", tp, b)[1] for m, tp, b in grid])
    assert unseen > seen


def test_an_unsure_prediction_falls_back_and_never_plans_below_the_model(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    cfg = config.merged(fast_cfg, {"guardrails": {"fallback_std": 0.0}})
    pm = _fit(tiny, fast_cfg)
    est = Learned(pm, cfg, tiny.services, Calibration(0.5))
    est.fit_fallback(tiny.profiles())
    mean, upper, src = est.step("llama-70b", "L40S", 8, 16)
    assert src == "fallback"
    plain = Learned(pm, fast_cfg, tiny.services, Calibration(0.5), fallback=False)
    assert upper >= plain.step("llama-70b", "L40S", 8, 16)[1]


def test_a_drifting_gpu_is_planned_with_a_margin(tiny: Site, fast_cfg: dict[str, Any]) -> None:
    est = Learned(_fit(tiny, fast_cfg), fast_cfg, tiny.services, Calibration(0.5))
    calm = est.step("llama-8b", "L40S", 1, 16)
    est.drifting = {"L40S"}
    drifting = est.step("llama-8b", "L40S", 1, 16)
    assert drifting[0] == calm[0] and drifting[1] > calm[1]
