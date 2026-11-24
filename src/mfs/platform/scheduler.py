"""The scheduler: every hour, place replicas of each LLM service on GPU types, pick their tensor-parallel degree
and batch size, route traffic and admit what the fleet can serve; then learn from what the replicas measured.

The plan is the same for every estimator; only the estimator's answers differ, so a comparison between
designs compares performance models and guardrails, not planners.

Safe failure:
- batches are planned on the estimator's upper bound, at a slack below the SLO;
- an unsure learned prediction falls back to the calibrated roofline (estimators.py / perfmodel.py);
- admission control sheds the excess explicitly instead of letting a replica's queue grow until every request
  misses the SLO;
- measured step times flow back; a GPU type whose replicas all moved is planned with a margin while the
  correction settles.

Training jobs fill what inference leaves free, and inference always wins: a job on GPUs that inference needs
again is preempted back to its last checkpoint. To lose less work, a job goes on GPUs that inference did not use
yesterday over the next few hours; a job expected to finish within a few hours may use any free GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..domain import Observation, Placement, Replica, ReplicaShape, Service, TrainingAssignment, TrainingJob, TrainingReport
from ..reference.hardware import BATCHES, MODELS, TP_DEGREES
from .estimators import Calibration, Estimator


@dataclass
class Candidate:
    gpu: str
    tp: int
    batch: int
    step_mean: float
    step_upper: float
    capacity_rps: float
    source: str

    @property
    def per_gpu(self) -> float:
        return self.capacity_rps / self.tp


def best_shape(model: str, gpu: str, tp: int, est: Estimator, svc: Service, slack: float) -> Candidate | None:
    best = None
    for b in BATCHES:
        if not est.fits(model, gpu, tp, b):
            break
        mean, upper, src = est.step(model, gpu, tp, b)
        if upper * 1000.0 > svc.tpot_ms * slack:
            break
        prefill = est.prefill(model, gpu, tp, svc.prompt)
        cap = b / (svc.output * upper + prefill)
        best = Candidate(gpu, tp, b, mean, upper, cap, src)
    return best


def candidates(model: str, fleet: dict[str, int], est: Estimator, svc: Service, slack: float) -> list[Candidate]:
    out = []
    for g, n in fleet.items():
        for tp in TP_DEGREES:
            if tp > min(8, n):
                continue
            c = best_shape(model, g, tp, est, svc, slack)
            if c:
                out.append(c)
    # per GPU type keep the tensor-parallel degree that serves the most per GPU
    best: dict[str, Candidate] = {}
    for c in out:
        if c.gpu not in best or c.per_gpu > best[c.gpu].per_gpu:
            best[c.gpu] = c
    return sorted(best.values(), key=lambda c: -c.per_gpu)


@dataclass
class Scheduler:
    est: Estimator
    cfg: dict[str, Any]
    services: dict[str, Service]
    fleet: dict[str, int]
    admission: bool = True
    feedback: bool = True
    calibration: Calibration | None = None
    previous: Placement | None = None
    drift_until: dict[str, int] = field(default_factory=dict)
    starts: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)
    peak_aware: bool = True
    jobs: dict[int, TrainingJob] = field(default_factory=dict)  # submitted and not finished
    remaining: dict[int, float] = field(default_factory=dict)  # tokens left, as the scheduler estimates
    running: dict[int, TrainingAssignment] = field(default_factory=dict)
    no_fit: set[tuple[str, str, int]] = field(default_factory=set)
    history: list[dict[str, int]] = field(default_factory=list)  # inference GPUs per type, per hour
    preemptions: int = 0

    def submit(self, job: TrainingJob) -> None:
        if job.id not in self.jobs:
            self.jobs[job.id] = job
            self.remaining[job.id] = job.tokens

    def plan(self, hour: int, forecast: dict[str, float], demand: dict[str, float]) -> Placement:
        g = self.cfg["guardrails"]
        if hasattr(self.est, "drifting"):
            self.est.drifting = {gpu for gpu, until in self.drift_until.items() if hour <= until}
        free = dict(self.fleet)
        cands = {m: candidates(m, self.fleet, self.est, self.services[m], float(g["tpot_slack"])) for m in self.services}
        order = sorted(self.services, key=lambda m: (len(cands[m]), -MODELS[m].params_b))
        prev: dict[str, list[Replica]] = {}
        for r in self.previous.replicas if self.previous else []:
            prev.setdefault(r.shape.model, []).append(r)
        placement = Placement(hour)
        for m in order:
            need = forecast[m] * (1 + float(g["headroom"]))
            have = 0.0
            chosen: list[Replica] = []
            by_gpu = {c.gpu: c for c in cands[m]}
            for r in sorted(prev.get(m, []), key=lambda r: -(by_gpu[r.shape.gpu].per_gpu if r.shape.gpu in by_gpu else 0)):
                c = by_gpu.get(r.shape.gpu)
                if c is None or c.tp != r.shape.tp or free[c.gpu] < c.tp or have >= need:
                    continue
                chosen.append(Replica(ReplicaShape(m, c.gpu, c.tp, c.batch), c.capacity_rps, c.step_upper * 1000, c.source))
                free[c.gpu] -= c.tp
                have += c.capacity_rps
            for c in cands[m]:
                while have < need and free[c.gpu] >= c.tp and len(chosen) < int(self.cfg["placement"]["max_replicas_per_service"]):
                    chosen.append(Replica(ReplicaShape(m, c.gpu, c.tp, c.batch), c.capacity_rps, c.step_upper * 1000, c.source))
                    free[c.gpu] -= c.tp
                    have += c.capacity_rps
                    self.starts += 1
            # route this hour's traffic by planned capacity; admit what the plan can serve
            lam = demand[m]
            total = sum(r.capacity_rps for r in chosen)
            shed = 0.0
            for r in chosen:
                share = lam * r.capacity_rps / total if total else 0.0
                if self.admission:
                    r.admitted_rps = min(share, float(g["admission_share"]) * r.capacity_rps)
                    shed += share - r.admitted_rps
                else:
                    r.admitted_rps = share
            if not chosen:
                shed = lam
            placement.shed_rps[m] = shed
            placement.replicas.extend(chosen)
        self._backfill(hour, free, placement)
        placement.notes = {"free": free, "drifting": sorted(gpu for gpu, until in self.drift_until.items() if hour <= until)}
        self.previous = placement
        return placement

    def _peak(self) -> dict[str, int]:
        """GPUs inference is expected to need over the next hours: what it used at those hours yesterday, and now."""
        ahead = int(self.cfg["training"]["lookahead_hours"])
        now = self.history[-1]
        h = len(self.history) - 1
        window = [now] + (self.history[h - 23 : h - 23 + ahead] if h >= 24 else [])
        return {g: max(x.get(g, 0) for x in window) for g in self.fleet}

    def _backfill(self, hour: int, free: dict[str, int], placement: Placement) -> None:
        tc = self.cfg["training"]
        self.history.append(placement.gpus_used())
        peak = self._peak()
        keep: list[TrainingAssignment] = []
        for jid in sorted(self.running):
            a = self.running[jid]
            if free[a.gpu] >= a.gpus:
                free[a.gpu] -= a.gpus
                keep.append(a)
            else:
                placement.preempted.append(jid)
                self.preemptions += 1
        self.running = {a.job: a for a in keep}
        stable = {g: self.fleet[g] - peak[g] - sum(a.gpus for a in keep if a.gpu == g and a.reason == "stable") for g in self.fleet}
        for job in sorted(self.jobs.values(), key=lambda j: (j.arrival, j.id)):
            if job.id in self.running or job.arrival > hour:
                continue
            best: tuple[str, float, str] | None = None
            for g in self.fleet:
                if free[g] < job.gpus or (job.model, g, job.gpus) in self.no_fit or not self.est.train_fits(job.model, g, job.gpus):
                    continue
                rate = self.est.train_rate(job.model, g, job.gpus)
                hours = self.remaining[job.id] / rate / 3600.0
                if not self.peak_aware:
                    reason = "free"
                elif stable[g] >= job.gpus:
                    reason = "stable"
                elif hours <= float(tc["short_job_hours"]):
                    reason = "short"
                else:
                    continue
                if best is None or rate > best[1]:
                    best = (g, rate, reason)
            if best:
                g, rate, reason = best
                self.running[job.id] = TrainingAssignment(job.id, g, job.gpus, rate, reason)
                free[g] -= job.gpus
                if reason == "stable":
                    stable[g] -= job.gpus
        placement.training = list(self.running.values())

    def observe_training(self, hour: int, reports: list[TrainingReport]) -> None:
        for r in reports:
            job = self.jobs.get(r.job)
            if job is None:
                continue
            if r.crashed:
                self.no_fit.add((job.model, r.gpu, r.gpus))
                self.running.pop(r.job, None)
                self.log.append({"hour": hour, "event": "train_oom", "job": r.job, "gpu": r.gpu})
                continue
            if r.done:
                self.jobs.pop(r.job)
                self.remaining.pop(r.job, None)
                self.running.pop(r.job, None)
                continue
            predicted = self.est.train_rate(job.model, r.gpu, r.gpus)
            self.remaining[r.job] = max(0.0, self.remaining[r.job] - r.tokens_per_s * 3600.0)
            if self.feedback and self.calibration is not None and r.tokens_per_s > 0:
                self.calibration.update_train(r.gpu, math.log(r.tokens_per_s / predicted))

    def observe(self, hour: int, observations: list[Observation]) -> None:
        if not self.feedback or self.calibration is None:
            return
        fb = self.cfg["feedback"]
        per_gpu: dict[str, list[float]] = {}
        per_bucket: dict[tuple[str, str, int, int], list[float]] = {}
        # compare every measurement with the prediction made before any of this hour's updates
        for o in observations:
            if o.crashed:
                continue
            s = o.shape
            mean, _, _ = self.est.step(s.model, s.gpu, s.tp, s.batch)
            lr = math.log(o.step_ms / 1000.0 / mean)
            per_bucket.setdefault((s.model, s.gpu, s.tp, s.batch), []).append(lr)
            per_gpu.setdefault(s.gpu, []).append(lr)
        for (m, gpu, tp, b), lrs in per_bucket.items():
            self.calibration.update(m, gpu, tp, b, sum(lrs) / len(lrs))
        self.log.append({"hour": hour, "event": "ratios", **{g: round(sum(v) / len(v), 3) for g, v in per_gpu.items()}})
        for gpu, lrs in per_gpu.items():
            avg = sum(lrs) / len(lrs)
            self.calibration.update_gpu(gpu, avg / 2)
            if len(lrs) >= 2 and all(abs(x) > float(fb["drift_log_ratio"]) and (x > 0) == (avg > 0) for x in lrs):
                self.drift_until[gpu] = hour + int(fb["drift_margin_hours"])
                self.log.append({"hour": hour, "event": "drift", "gpu": gpu, "log_ratio": round(avg, 3)})
