"""Runs one design over one simulated week at one site, and grades every hour against the site's truth."""

from __future__ import annotations

import json
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
    inf_busy_gpu_h: float = 0.0  # GPUs x share of planned capacity actually admitted
    lost_gpu_h: float = 0.0  # fine-tuning work thrown away by preemption, in GPU-hours
    oom_gpu_h: float = 0.0
    waiting_gpus: int = 0  # GPUs asked for by fine-tuning jobs that are queued, not running
    stranded: tuple[float, float, float] = (0.0, 0.0, 0.0)  # free GPUs while jobs waited: (a job fit, too few, none fits in memory)


@dataclass
class Run:
    name: str
    hours: list[HourResult] = field(default_factory=list)
    starts: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)
    train: dict[str, Any] = field(default_factory=dict)
    jobs: dict[int, Any] = field(default_factory=dict)  # job id -> {arrival, done_at}
    twin: list[tuple[int, str, float, float]] = field(default_factory=list)  # (hour, gpu, |log err| of the twin, of the calibrated model)
    train_net_tokens: float = 0.0  # tokens of progress kept at the end of the week (work lost to preemption excluded)


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
        return Scheduler(est, cfg, services, dict(site.fleet), admission=True, feedback=True, calibration=cal, peak_aware="peak_aware" not in off), train
    if kind == "oracle":
        est = Oracle(site)
        return Scheduler(est, cfg, services, dict(site.fleet), admission=True, feedback=False), train
    return build_platform(services, site.fleet, profiles, cfg, seed=site.seed, without=off, training_profiles=site.training_profiles())


def load_backfill_weights() -> list[float]:
    from .. import config as _config

    return [float(x) for x in json.loads((_config.ROOT / "models" / "backfill_policy.json").read_text(encoding="utf-8"))["weights"]]


def run_design(site: Site, name: str, d: dict[str, Any], cfg: dict[str, Any], *, keep_placements: bool = False, backfill_weights: list[float] | None = None) -> Run:
    sched, train = make_scheduler(site, d, cfg)
    if backfill_weights is not None:
        sched.backfill_weights = backfill_weights
    elif d.get("backfill") == "learned":
        sched.backfill_weights = load_backfill_weights()
    run = Run(name, train=train)
    models = site.models
    jobs = {j.id: j for j in site.training_jobs()}
    left = {j: job.tokens for j, job in jobs.items()}  # the truth: tokens each job still has to process
    last_gpu: dict[int, tuple[str, int]] = {}
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
        pm = getattr(sched.est, "pm", None)
        if pm is not None:  # the Simulator's model against what production measured, before this hour's feedback
            for o in obs:
                if o.crashed or o.step_ms <= 0:
                    continue
                s_ = o.shape
                twin_log, _ = pm.predict_log(s_.model, s_.gpu, s_.tp, s_.batch)
                cal_mean, _, _ = sched.est.step(s_.model, s_.gpu, s_.tp, s_.batch)
                lo = math.log(o.step_ms / 1000.0)
                run.twin.append((h, s_.gpu, abs(lo - twin_log), abs(lo - math.log(cal_mean))))
        sched.observe(h, obs)
        # training: preempted jobs fall back to their last checkpoint; the rest run for the hour
        lost = 0.0
        for jid in p.preempted:
            job = jobs[jid]
            before = left[jid]
            lg, ln = last_gpu[jid]
            left[jid] = min(job.tokens, left[jid] + loss_h * 3600.0 * site.train_rate(job.model, lg, ln, h))
            lost += ln * loss_h * (left[jid] - before) / max(loss_h * 3600.0 * site.train_rate(job.model, lg, ln, h), 1e-9)
        reports: list[TrainingReport] = []
        tg = tt = 0.0
        done = tcrash = 0
        for a in p.training:
            job = jobs[a.job]
            last_gpu[a.job] = (a.gpu, a.gpus)
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
        busy = sum(r.shape.tp * min(1.0, r.admitted_rps / r.capacity_rps) for r in p.replicas if r.capacity_rps > 0)
        running = {a.job for a in p.training}
        queued = [j for jid, j in sched.jobs.items() if jid not in running and j.arrival <= h]
        waiting = sum(j.gpus for j in queued)
        on_train: dict[str, int] = {}
        for a in p.training:
            on_train[a.gpu] = on_train.get(a.gpu, 0) + a.gpus
        st = [0.0, 0.0, 0.0]
        if queued:
            used = p.gpus_used()
            for g, n in site.fleet.items():
                f = n - used.get(g, 0) - on_train.get(g, 0)
                if f <= 0:
                    continue
                fit = [j for j in queued if site.train_fits(j.model, g, j.gpus)]
                st[0 if any(j.gpus <= f for j in fit) else 1 if fit else 2] += f
        run.hours.append(
            HourResult(
                h,
                demand,
                met,
                dict(p.shed_rps),
                crashed,
                tpot,
                over,
                p.gpus_used(),
                sources,
                tg,
                tt,
                done,
                tcrash,
                len(p.preempted),
                busy,
                lost,
                float(sum(a.gpus for a in p.training if not site.train_fits(jobs[a.job].model, a.gpu, a.gpus))),
                waiting,
                (st[0], st[1], st[2]),
            )
        )
        if keep_placements:
            run.placements.append(p)
    run.starts = sched.starts
    run.log = sched.log
    run.train_net_tokens = sum(job.tokens - left[j] for j, job in jobs.items() if job.arrival < site.hours)
    return run
