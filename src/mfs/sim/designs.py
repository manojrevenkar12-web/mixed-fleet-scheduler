"""Runs one design over one simulated week at one site, and grades every hour against the site's truth."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..domain import Observation, Placement, Replica, TrainingReport
from ..platform.build import build_platform
from ..platform.estimators import Calibration, Roofline, Uniform, train_ratios
from ..platform.perfmodel import prefill_calibration, step_ratios
from ..platform.scheduler import Scheduler
from ..world.world import Site


class Oracle:
    """The site's truth at this hour: the upper bound no scheduler reaches."""

    name = "oracle"

    def __init__(self, site: Site) -> None:
        self.site, self.hour = site, 0

    def step(self, model: str, gpu: str, tp: int, batch: int) -> tuple[float, float, str]:
        t = self.site.step_s(model, gpu, tp, batch, self.hour)
        return t, t, "oracle"

    def prefill(self, model: str, gpu: str, tp: int, prompt: int) -> float:
        return self.site.prefill_s(model, gpu, tp, prompt, self.hour)

    def fits(self, model: str, gpu: str, tp: int, batch: int) -> bool:
        return self.site.fits(model, gpu, tp, batch)

    def train_rate(self, model: str, gpu: str, gpus: int) -> float:
        return self.site.train_rate(model, gpu, gpus, self.hour)

    def train_fits(self, model: str, gpu: str, gpus: int) -> bool:
        return self.site.train_fits(model, gpu, gpus)


@dataclass
class HourResult:
    hour: int
    offered: dict[str, float]
    met: dict[str, float]
    shed: dict[str, float]
    crashed: int
    tpot_missed: int
    overloaded: int
    gpus: dict[str, int]
    sources: dict[str, int]
    train_gpu_hours: float = 0.0
    train_tokens: float = 0.0
    train_done: int = 0
    train_crashed: int = 0
    preempted: int = 0


@dataclass
class Run:
    name: str
    hours: list[HourResult] = field(default_factory=list)
    starts: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)
    train: dict[str, Any] = field(default_factory=dict)
    jobs: dict[int, Any] = field(default_factory=dict)  # job id -> {arrival, done_at}


def evaluate_replica(site: Site, r: Replica, hour: int) -> tuple[float, bool, bool, bool, int]:
    """(requests per second within SLO, crashed, missed TPOT, overloaded, batch actually run)."""
    s, svc = r.shape, site.services[r.shape.model]
    lam = r.admitted_rps
    if not site.fits(s.model, s.gpu, s.tp, s.batch):
        return 0.0, True, False, False, s.batch
    pre = site.prefill_s(s.model, s.gpu, s.tp, svc.prompt, hour)
    # concurrency by Little's law, at most the planned batch
    b = s.batch
    for _ in range(3):
        step = site.step_s(s.model, s.gpu, s.tp, b, hour)
        b = max(1, min(s.batch, math.ceil(lam * (svc.output * step + pre))))
    step = site.step_s(s.model, s.gpu, s.tp, b, hour)
    if step * 1000.0 > svc.tpot_ms:
        return 0.0, False, True, False, b
    full = site.step_s(s.model, s.gpu, s.tp, s.batch, hour)
    mu = s.batch / (svc.output * full + pre)
    if lam >= mu:
        return 0.0, False, False, True, b
    slack = svc.ttft_s - pre
    if slack <= 0:
        return 0.0, False, False, False, b
    rho = lam / mu
    attain = max(0.0, 1.0 - rho * math.exp(-(mu - lam) * slack))
    return lam * attain, False, False, False, b


def make_scheduler(site: Site, d: dict[str, Any], cfg: dict[str, Any], *, profiles: list[Any] | None = None) -> tuple[Scheduler, dict[str, Any]]:
    kind = d["kind"]
    off = set(d.get("without", []))
    if d.get("set"):
        from .. import config as _config

        cfg = _config.merged(cfg, d["set"])
    services = site.services
    profiles = site.profiles() if profiles is None else profiles
    cal = Calibration(float(cfg["feedback"]["alpha"]))
    cal.prefill = prefill_calibration(profiles, cfg, int(cfg["feedback"]["prefill_ratio_min_profiles"]))
    train: dict[str, Any] = {}
    est: Any
    if kind == "uniform":
        est = Uniform(cfg, services)
        return Scheduler(est, cfg, services, dict(site.fleet), admission=False, feedback=False), train
    if kind == "roofline":
        est = Roofline(cfg, services)
        return Scheduler(est, cfg, services, dict(site.fleet), admission=False, feedback=False), train
    if kind == "roofline_calibrated":
        cal.gpu = step_ratios(profiles, cfg, services)
        cal.train = train_ratios(site.training_profiles(), cfg)
        est = Roofline(cfg, services, cal)
        return Scheduler(est, cfg, services, dict(site.fleet), admission=True, feedback=True, calibration=cal), train
    if kind == "oracle":
        est = Oracle(site)
        return Scheduler(est, cfg, services, dict(site.fleet), admission=True, feedback=False), train
    return build_platform(services, site.fleet, profiles, cfg, seed=site.seed, without=off, training_profiles=site.training_profiles())


def run_design(site: Site, name: str, d: dict[str, Any], cfg: dict[str, Any], *, keep_placements: bool = False) -> Run:
    sched, train = make_scheduler(site, d, cfg)
    run = Run(name, train=train)
    models = site.models
    jobs = {j.id: j for j in site.training_jobs()}
    left = {j: job.tokens for j, job in jobs.items()}  # the truth: tokens each job still has to process
    last_gpu: dict[int, str] = {}
    loss_h = float(site.cfg["training"]["checkpoint_loss_hours"]) if jobs else 0.0
    for h in range(site.hours):
        if isinstance(sched.est, Oracle):
            sched.est.hour = h
        for job in jobs.values():
            if job.arrival == h:
                sched.submit(job)
                run.jobs[job.id] = {"arrival": h, "done_at": None}
        demand = {m: float(site.demand[h, j]) for j, m in enumerate(models)}
        forecast = {m: float(site.demand[max(0, h - 1), j]) for j, m in enumerate(models)}
        p = sched.plan(h, forecast, demand)
        met: dict[str, float] = {m: 0.0 for m in models}
        crashed = tpot = over = 0
        obs: list[Observation] = []
        sources: dict[str, int] = {}
        for r in p.replicas:
            ok, cr, tm, ov, b = evaluate_replica(site, r, h)
            met[r.shape.model] += ok
            crashed += int(cr)
            tpot += int(tm)
            over += int(ov)
            sources[r.source] = sources.get(r.source, 0) + 1
            shape = type(r.shape)(r.shape.model, r.shape.gpu, r.shape.tp, b)
            obs.append(site.observe(shape, h, cr) if not cr else Observation(shape, 0.0, True))
        sched.observe(h, obs)
        # training: preempted jobs fall back to their last checkpoint; the rest run for the hour
        for jid in p.preempted:
            job = jobs[jid]
            left[jid] = min(job.tokens, left[jid] + loss_h * 3600.0 * site.train_rate(job.model, last_gpu[jid], job.gpus, h))
        reports: list[TrainingReport] = []
        tg = tt = 0.0
        done = tcrash = 0
        for a in p.training:
            job = jobs[a.job]
            last_gpu[a.job] = a.gpu
            if not site.train_fits(job.model, a.gpu, a.gpus):
                reports.append(TrainingReport(a.job, a.gpu, a.gpus, 0.0, crashed=True))
                tcrash += 1
                tg += a.gpus
                continue
            rate = site.train_rate(job.model, a.gpu, a.gpus, h)
            secs = min(3600.0, left[a.job] / rate)
            left[a.job] -= rate * secs
            tg += a.gpus * secs / 3600.0
            tt += rate * secs
            finished = left[a.job] <= 1e-6
            if finished:
                done += 1
                run.jobs[a.job]["done_at"] = h + secs / 3600.0
            reports.append(TrainingReport(a.job, a.gpu, a.gpus, site.observe_training(a.job, job.model, a.gpu, a.gpus, h), done=finished))
        sched.observe_training(h, reports)
        run.hours.append(HourResult(h, demand, met, dict(p.shed_rps), crashed, tpot, over, p.gpus_used(), sources, tg, tt, done, tcrash, len(p.preempted)))
        if keep_placements:
            run.placements.append(p)
    run.starts = sched.starts
    run.log = sched.log
    return run
