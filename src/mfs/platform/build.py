"""Builds the platform's scheduler from measured profiles. The lab and the service both call this, so the
service runs exactly the scheduler the lab graded."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..domain import Profile, Service, TrainingProfile
from .estimators import Calibration, train_ratios
from .perfmodel import Learned, PerfModel, prefill_calibration
from .scheduler import Scheduler


def build_platform(
    services: dict[str, Service],
    fleet: dict[str, int],
    profiles: list[Profile],
    cfg: dict[str, Any],
    *,
    seed: int = 0,
    without: Iterable[str] = (),
    training_profiles: list[TrainingProfile] | None = None,
) -> tuple[Scheduler, dict[str, Any]]:
    off = set(without)
    cal = Calibration(float(cfg["feedback"]["alpha"]))
    cal.prefill = prefill_calibration(profiles, cfg, int(cfg["feedback"]["prefill_ratio_min_profiles"]))
    cal.train = train_ratios(training_profiles or [], cfg)
    pm = PerfModel(cfg, services, seed=seed)
    rep = pm.fit(profiles)
    est = Learned(pm, cfg, services, cal, margin="margin" not in off, fallback="fallback" not in off)
    est.fit_fallback(profiles)
    sched = Scheduler(est, cfg, services, dict(fleet), admission="admission" not in off, feedback="feedback" not in off, calibration=cal, peak_aware="peak_aware" not in off)
    return sched, {"points": rep.points, "members": rep.members, "train_rmse_log": rep.train_rmse_log}
