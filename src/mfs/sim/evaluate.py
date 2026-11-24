"""Grades runs against the site's truth: service quality, GPUs, failures; the performance model's accuracy on
pairs it has and has not measured; and the twin's forecast for a planned expansion."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..domain import Profile
from ..platform.estimators import Calibration, Roofline, roofline_parts
from ..platform.perfmodel import Learned, PerfModel, prefill_calibration
from ..platform.scheduler import Scheduler
from ..reference.hardware import BATCHES, TP_DEGREES
from ..world.world import Site
from .designs import Oracle, Run, evaluate_replica


def run_metrics(site: Site, run: Run) -> dict[str, Any]:
    offered = sum(sum(h.offered.values()) for h in run.hours)
    met = sum(sum(h.met.values()) for h in run.hours)
    shed = sum(sum(h.shed.values()) for h in run.hours)
    service_hours = bad = 0
    for h in run.hours:
        for m, lam in h.offered.items():
            service_hours += 1
            if lam > 0 and h.met[m] / lam < 0.9:
                bad += 1
    gpu_hours = sum(sum(h.gpus.values()) for h in run.hours)
    fleet = sum(site.fleet.values())
    sources: dict[str, int] = {}
    for h in run.hours:
        for k, v in h.sources.items():
            sources[k] = sources.get(k, 0) + v
    reps = sum(sources.values()) or 1
    # recovery after the first driver change: hours until that GPU's service is back to 99% of its demand met
    drift = (site.cfg.get("drift") or [{}])[0]
    recover = None
    if drift:
        start = int(drift["from_hour"])
        for h in run.hours[start:]:
            lam = sum(h.offered.values())
            if lam and sum(h.met.values()) / lam >= 0.97 and h.hour > start:
                recover = h.hour - start
                break
    drift_window = [h for h in run.hours if drift and int(drift["from_hour"]) <= h.hour < int(drift["from_hour"]) + 12]
    dw_off = sum(sum(h.offered.values()) for h in drift_window)
    dw_met = sum(sum(h.met.values()) for h in drift_window)
    return {
        "requests": offered * 3600,
        "attainment": met / offered,
        "violated_share": (offered - met - shed) / offered,
        "shed_share": shed / offered,
        "bad_service_hours": bad,
        "service_hours": service_hours,
        "gpu_utilisation": gpu_hours / (fleet * len(run.hours)),
        "met_per_gpu_hour": met * 3600 / gpu_hours if gpu_hours else 0.0,
        "crashes": sum(h.crashed for h in run.hours),
        "tpot_misses": sum(h.tpot_missed for h in run.hours),
        "overloads": sum(h.overloaded for h in run.hours),
        "replica_starts_per_day": run.starts / (len(run.hours) / 24),
        "fallback_share": sources.get("fallback", 0) / reps,
        "drift_events": len([e for e in run.log if e.get("event") == "drift"]),
        "drift_window_attainment": dw_met / dw_off if dw_off else None,
        "drift_recovery_hours": recover,
        "train_points": run.train.get("points", 0),
        **_training_metrics(site, run, gpu_hours),
        **{f"cap_{k}": v for k, v in capacity_breakdown(site, run).items()},
    }


def capacity_breakdown(site: Site, run: Run) -> dict[str, float]:
    """Where the fleet's GPU-hours went, as shares that add up to 1."""
    fleet = sum(site.fleet.values())
    total = fleet * len(run.hours)
    serving = headroom = train = lost = oom = idle = 0.0
    st = [0.0, 0.0, 0.0]
    for h in run.hours:
        alloc = sum(h.gpus.values())
        serving += h.inf_busy_gpu_h
        headroom += alloc - h.inf_busy_gpu_h
        lost += h.lost_gpu_h
        oom += h.oom_gpu_h
        train += max(0.0, h.train_gpu_hours - h.oom_gpu_h)
        free = max(0.0, fleet - alloc - h.train_gpu_hours)
        s = min(free, sum(h.stranded))
        scale = s / sum(h.stranded) if sum(h.stranded) > 0 else 0.0
        for i in range(3):
            st[i] += h.stranded[i] * scale
        idle += free - s
    useful_train = max(0.0, train - lost)
    return {
        "serving": serving / total,
        "inference_headroom": headroom / total,
        "training_kept": useful_train / total,
        "training_lost_to_preemption": lost / total,
        "training_out_of_memory": oom / total,
        "stranded_job_fit_but_waited": st[0] / total,
        "stranded_too_few_gpus_free": st[1] / total,
        "stranded_no_job_fits_memory": st[2] / total,
        "idle_nothing_queued": idle / total,
    }


def twin_drift(site: Site, run: Run) -> dict[str, Any] | None:
    """How far the Simulator's model drifts from what the Scheduler measures in production, per day, and when an
    alarm (daily error above the threshold on one GPU type) would have fired."""
    if not run.twin:
        return None
    days = len(run.hours) // 24
    gpus = sorted({g for _, g, _, _ in run.twin})
    daily: dict[str, list[float | None]] = {}
    daily_cal: dict[str, list[float | None]] = {}
    for g in gpus:
        daily[g], daily_cal[g] = [], []
        for d in range(days):
            rows = [(e, c) for h, gg, e, c in run.twin if gg == g and d * 24 <= h < (d + 1) * 24]
            daily[g].append(float(np.mean([math.expm1(e) for e, _ in rows])) if rows else None)
            daily_cal[g].append(float(np.mean([math.expm1(c) for _, c in rows])) if rows else None)
    thr = 0.10
    drift = (site.cfg.get("drift") or [{}])[0]
    out: dict[str, Any] = {"threshold": thr, "daily_twin_error": daily, "daily_calibrated_error": daily_cal}
    if drift:
        g, start_day = drift["gpu"], int(drift["from_hour"]) // 24
        series = daily.get(g) or [None] * days  # a GPU type with no replicas has no measurements to compare
        fired = next((d for d in range(start_day, days) if (series[d] or 0) > thr), None)
        out["drift_gpu"] = g
        out["alarm_days_after_change"] = None if fired is None else fired - start_day
        cut = int(drift["from_hour"])
        before = [math.expm1(e) for h, gg, e, _ in run.twin if gg == g and h < cut]
        after = [math.expm1(e) for h, gg, e, _ in run.twin if gg == g and h >= cut]
        cal_after = [math.expm1(c) for h, gg, _, c in run.twin if gg == g and h >= cut]
        out["twin_error_before"] = float(np.mean(before)) if before else None
        out["twin_error_after"] = float(np.mean(after)) if after else None
        out["calibrated_error_after"] = float(np.mean(cal_after)) if cal_after else None
    out["false_alarm_days"] = sum(1 for gg in gpus for d, x in enumerate(daily[gg]) if x is not None and x > thr and not (drift and gg == drift["gpu"] and d >= int(drift["from_hour"]) // 24))
    return out


def _training_metrics(site: Site, run: Run, inference_gpu_hours: float) -> dict[str, Any]:
    fleet_hours = sum(site.fleet.values()) * len(run.hours)
    tg = sum(h.train_gpu_hours for h in run.hours)
    done = [j for j in run.jobs.values() if j["done_at"] is not None]
    return {
        "training_jobs": len(run.jobs),
        "training_done_share": len(done) / len(run.jobs) if run.jobs else None,
        "training_job_hours": sum(j["done_at"] - j["arrival"] for j in done) / len(done) if done else None,
        "training_tokens_b": sum(h.train_tokens for h in run.hours) / 1e9,
        "training_net_tokens_b": run.train_net_tokens / 1e9,
        "training_gpu_share": tg / fleet_hours,
        "training_preemptions": sum(h.preempted for h in run.hours),
        "training_crashes": sum(h.train_crashed for h in run.hours),
        "fleet_busy": (inference_gpu_hours + tg) / fleet_hours,
    }


def _grid(site: Site, pairs: set[tuple[str, str]]) -> list[tuple[str, str, int, int]]:
    out = []
    for m, g in sorted(pairs):
        for tp in TP_DEGREES:
            for b in BATCHES:
                if site.fits(m, g, tp, b):
                    out.append((m, g, tp, b))
    return out


def _scores(site: Site, pm: PerfModel, cfg: dict[str, Any], grid: list[tuple[str, str, int, int]], z: float) -> dict[str, float]:
    if not grid:
        return {}
    err_l, err_r, cover = [], [], []
    for m, g, tp, b in grid:
        true = site.step_s(m, g, tp, b, 0)
        mean, std = pm.predict_log(m, g, tp, b)
        err_l.append(abs(math.exp(mean) / true - 1))
        err_r.append(abs(roofline_parts(m, g, tp, b, site.services[m], cfg)[2] / true - 1))
        cover.append(abs(math.log(true) - mean) <= z * std)
    return {"points": len(grid), "learned_mape": float(np.mean(err_l)), "roofline_mape": float(np.mean(err_r)), "interval_coverage": float(np.mean(cover))}


def model_accuracy(site: Site, cfg: dict[str, Any]) -> dict[str, Any]:
    """MAPE of the decode-step prediction, learned vs roofline, on measured pairs (at points not measured),
    held-back pairs, and a GPU type the model never saw (leave one GPU out)."""
    z = 1.645  # a 90% interval
    profiles = site.profiles()
    measured = {(p.model, p.gpu) for p in profiles}
    taken = {(p.model, p.gpu, p.tp, p.batch) for p in profiles}
    all_pairs = {(m, g) for m in site.models for g in site.fleet}
    pm = PerfModel(cfg, site.services, seed=site.seed)
    pm.fit(profiles)
    seen = [x for x in _grid(site, measured) if x not in taken]
    out: dict[str, Any] = {"measured_pairs": _scores(site, pm, cfg, seen, z), "unmeasured_pairs": _scores(site, pm, cfg, _grid(site, all_pairs - measured), z)}
    # a GPU type the model has never seen: train without it
    holdout = site.cfg["expansion"]["gpu"]
    base = [p for p in profiles if p.gpu != holdout]
    grid = _grid(site, {(m, holdout) for m in site.models})
    pm2 = PerfModel(cfg, site.services, seed=site.seed + 7)
    pm2.fit(base)
    out["unseen_gpu"] = {"gpu": holdout, **_scores(site, pm2, cfg, grid, z)}
    pm3 = PerfModel(cfg, site.services, seed=site.seed + 7)
    pm3.fit(base + calibration_profiles(site, cfg, holdout))
    out["unseen_gpu_calibrated"] = {"gpu": holdout, "extra_profiles": int(cfg["expansion"]["calibration_profiles"]), **_scores(site, pm3, cfg, grid, z)}
    return out


def calibration_profiles(site: Site, cfg: dict[str, Any], gpu: str) -> list[Profile]:
    """A few fresh measurements on a GPU type: one per service, at batch 32, the smallest degree that fits."""
    out: list[Profile] = []
    n = int(cfg["expansion"]["calibration_profiles"])
    for k, m in enumerate(sorted(site.models)[:n]):
        tp = next((t for t in (1, 2, 4, 8) if site.fits(m, gpu, t, 32)), 8)
        out.append(site.measure_profile(m, gpu, tp, 32, 0, ("calibrate", k)))
    return out


def _capacity(site: Site, est: Any, cfg: dict[str, Any], fleet: dict[str, int], truth: bool) -> float:
    """Largest multiple of the weekly peak the fleet can plan for (planner's view), or truly serve at 99%."""
    peak = {m: float(site.demand[:, j].max()) for j, m in enumerate(site.models)}
    target = float(cfg["expansion"]["target_attainment"])

    def ok(scale: float) -> bool:
        sched = Scheduler(est, cfg, site.services, fleet, admission=False, feedback=False)
        dem = {m: v * scale for m, v in peak.items()}
        need = 1 + float(cfg["guardrails"]["headroom"])
        p = sched.plan(0, dem, dem)
        if not truth:
            return all(sum(r.capacity_rps for r in p.replicas if r.shape.model == m) >= dem[m] * need * 0.999 for m in dem)
        served = sum(evaluate_replica(site, r, 0)[0] for r in p.replicas)
        return served >= target * sum(dem.values())

    lo, hi = 0.05, 4.0
    for _ in range(22):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def expansion_forecast(site: Site, cfg: dict[str, Any]) -> dict[str, Any]:
    """The simulator's question: how much more peak traffic does a planned purchase serve? Each estimator plans
    the fleet before and after; the truth is what the oracle can serve at 99% within the SLO."""
    ex = site.cfg["expansion"]
    gpu, count = ex["gpu"], int(ex["count"])
    before = dict(site.fleet)
    after = dict(site.fleet)
    after[gpu] = after.get(gpu, 0) + count
    oracle = Oracle(site)
    true_gain = _capacity(site, oracle, cfg, after, True) - _capacity(site, oracle, cfg, before, True)
    profiles = site.profiles()
    cal = Calibration(float(cfg["feedback"]["alpha"]))
    cal.prefill = prefill_calibration(profiles, cfg, 2)
    roof = Roofline(cfg, site.services)
    pm = PerfModel(cfg, site.services, seed=site.seed)
    pm.fit(profiles)
    learned = Learned(pm, cfg, site.services, cal, margin=False, fallback=False)
    learned.fit_fallback(profiles)
    # calibrate the twin with a few fresh measurements on the new GPU type
    fresh = calibration_profiles(site, cfg, gpu)
    cal2 = Calibration(cal.alpha, prefill={**cal.prefill, **prefill_calibration(fresh, cfg, 1)})
    pm2 = PerfModel(cfg, site.services, seed=site.seed)
    pm2.fit(profiles + fresh)
    calibrated = Learned(pm2, cfg, site.services, cal2, margin=False, fallback=False)
    calibrated.fit_fallback(profiles + fresh)
    out: dict[str, Any] = {"gpu": gpu, "count": count, "true_gain": true_gain, "estimators": {}}
    for name, est in (("roofline", roof), ("learned", learned), ("learned_calibrated", calibrated)):
        gain = _capacity(site, est, cfg, after, False) - _capacity(site, est, cfg, before, False)
        out["estimators"][name] = {"predicted_gain": gain, "error": gain / true_gain - 1 if true_gain else None}
    return out
