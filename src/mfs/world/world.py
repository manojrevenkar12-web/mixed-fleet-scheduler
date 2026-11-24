"""A simulated site: its fleet, its services and traffic, and how fast each model really runs on each GPU.

The truth is a roofline with hidden efficiencies: how close kernels get to the data sheet per GPU, a
model-on-GPU factor, a site factor (drivers, firmware), per-step launch overhead, a batch penalty, imperfect
overlap of compute and memory, tensor-parallel communication over the node's links, and driver changes that
arrive mid-week. Designs only see measurements of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import numpy as np
import yaml

from .. import config
from ..domain import Observation, Profile, ReplicaShape, Service, TrainingJob, TrainingProfile
from ..reference.hardware import BATCHES, GPUS, MODELS, TP_DEGREES, weights_read_bytes
from ..rng import keyed


@cache
def world_config(variant: str) -> dict[str, Any]:
    raw = yaml.safe_load((config.CONFIG / "world.yml").read_text(encoding="utf-8"))
    base = raw["dev"]
    return base if variant == "dev" else config.merged(base, raw[variant])


@dataclass
class Site:
    seed: int
    variant: str
    cfg: dict[str, Any]
    fleet: dict[str, int]
    services: dict[str, Service]
    eff: dict[str, dict[str, float]]
    pair: dict[tuple[str, str], tuple[float, float]]
    demand: np.ndarray  # [hours, models] requests per second
    models: list[str]
    _profiles: list[Profile] = field(default_factory=list)

    @property
    def hours(self) -> int:
        return int(self.cfg["hours"])

    # ------------------------------------------------------------------ truth
    def _drift(self, gpu: str, hour: int) -> tuple[float, float]:
        mem = comp = 1.0
        for d in self.cfg.get("drift", []):
            if d["gpu"] == gpu and hour >= int(d["from_hour"]):
                mem *= float(d["mem"])
                comp *= float(d["compute"])
        return mem, comp

    def step_s(self, model: str, gpu: str, tp: int, batch: int, hour: int = 0) -> float:
        """True time of one decode step for `batch` sequences."""
        m, g, e = MODELS[model], GPUS[gpu], self.eff[gpu]
        jm, jc = self.pair[(model, gpu)]
        dm, dc = self._drift(gpu, hour)
        svc = self.services[model]
        ctx = svc.prompt + svc.output / 2
        wbytes = weights_read_bytes(m, batch) / tp
        kv = batch * ctx * m.kv_bytes_per_token / tp
        t_mem = (wbytes + kv) / (g.hbm_tbs * 1e12 * e["mem"] * jm * dm)
        t_cmp = 2.0 * m.active_b * 1e9 * batch / tp / (g.tflops * 1e12 * e["compute"] * jc * dc)
        t = max(t_mem, t_cmp) + float(self.cfg["overlap"]) * min(t_mem, t_cmp)
        t += e["overhead_ms"] / 1000.0 * m.layers / 32.0
        if tp > 1:
            lat = float(self.cfg["link_latency_us"][g.link]) * 1e-6
            vol = batch * m.hidden * 2.0 * 2.0 * (tp - 1) / tp
            t += m.layers * 2 * (vol / (g.link_gbs * 1e9) + lat)
        t *= 1.0 + e["batch_penalty"] * max(0.0, math.log2(batch / 64.0))
        return t

    def prefill_s(self, model: str, gpu: str, tp: int, prompt: int, hour: int = 0) -> float:
        m, g, e = MODELS[model], GPUS[gpu], self.eff[gpu]
        _, jc = self.pair[(model, gpu)]
        _, dc = self._drift(gpu, hour)
        t = 2.0 * m.active_b * 1e9 * prompt / tp / (g.tflops * 1e12 * e["compute"] * jc * dc * 1.25)
        t += e["overhead_ms"] / 1000.0 * m.layers / 32.0 * 4
        if tp > 1:
            t += m.layers * 2 * (prompt * m.hidden * 4.0 * (tp - 1) / tp / (g.link_gbs * 1e9))
        return t

    def usable_gb(self, gpu: str) -> float:
        g = GPUS[gpu]
        return g.mem_gb * (1 - float(self.cfg["memory_reserve_share"])) - float(self.cfg["memory_reserve_gb"])

    def needed_gb(self, model: str, tp: int, batch: int) -> float:
        m, svc = MODELS[model], self.services[model]
        kv = batch * (svc.prompt + svc.output) * m.kv_bytes_per_token / tp
        return (m.weight_bytes / tp + kv) / 1e9 + float(self.cfg["activation_gb_per_64"]) * batch / 64.0

    def fits(self, model: str, gpu: str, tp: int, batch: int) -> bool:
        return self.needed_gb(model, tp, batch) <= self.usable_gb(gpu)

    # ------------------------------------------------------------------ training truth
    def train_rate(self, model: str, gpu: str, gpus: int, hour: int = 0) -> float:
        """True tokens per second of a fine-tuning job on `gpus` GPUs of one type."""
        m, g, e, tc = MODELS[model], GPUS[gpu], self.eff[gpu], self.cfg["training"]
        _, jc = self.pair[(model, gpu)]
        _, dc = self._drift(gpu, hour)
        per_gpu = g.tflops * 1e12 * e["compute"] * jc * dc * float(tc["mfu"]) / (6.0 * m.active_b * 1e9)
        comm = 1.0 / (1.0 + float(tc["comm_k"]) * math.sqrt(float(tc["comm_ref_gbs"]) / g.link_gbs)) if gpus > 1 else 1.0
        if gpus > 8:
            comm *= float(tc["cross_node"])
        return gpus * per_gpu * comm

    def train_needed_gb(self, model: str, gpus: int) -> float:
        tc = self.cfg["training"]
        return MODELS[model].params_b * float(tc["bytes_per_param"]) / gpus + float(tc["activation_gb"])

    def train_fits(self, model: str, gpu: str, gpus: int) -> bool:
        return self.train_needed_gb(model, gpus) <= self.usable_gb(gpu)

    # ------------------------------------------------------------------ what a design may see
    def training_jobs(self) -> list[TrainingJob]:
        tc = self.cfg.get("training")
        if not tc:
            return []
        rng = keyed(self.seed, self.variant, "jobs")
        models = sorted(tc["models"])
        out = []
        for i in range(int(tc["jobs"])):
            m = models[int(rng.integers(len(models)))]
            spec = tc["models"][m]
            gpus = int(spec["gpus"][int(rng.integers(len(spec["gpus"])))])
            tokens = float(spec["tokens_b"]) * 1e9 * math.exp(rng.normal() * float(tc["tokens_sigma"]))
            out.append(TrainingJob(i, m, gpus, tokens, int(rng.integers(0, int(tc["arrive_within_hours"])))))
        return sorted(out, key=lambda j: (j.arrival, j.id))

    def training_profiles(self) -> list[TrainingProfile]:
        """One measured step per trainable model and GPU type the site has tried, on 8 GPUs."""
        tc = self.cfg.get("training")
        if not tc:
            return []
        never = set(self.cfg["profiles"].get("no_profiles", []))
        out = []
        for m in sorted(tc["models"]):
            for g in self.fleet:
                if g in never or not self.train_fits(m, g, 8):
                    continue
                z = keyed(self.seed, self.variant, "train-profile", m, g).normal()
                out.append(TrainingProfile(m, g, 8, self.train_rate(m, g, 8) * math.exp(z * float(self.cfg["measurement_sigma"]))))
        return out

    def observe_training(self, job: int, model: str, gpu: str, gpus: int, hour: int) -> float:
        z = keyed(self.seed, self.variant, "train-observe", job, hour).normal()
        return self.train_rate(model, gpu, gpus, hour) * math.exp(z * float(self.cfg["measurement_sigma"]))

    # ------------------------------------------------------------------ profiles
    def profiles(self) -> list[Profile]:
        return list(self._profiles)

    def observe(self, shape: ReplicaShape, hour: int, crashed: bool) -> Observation:
        z = keyed(self.seed, self.variant, "observe", shape, hour).normal()
        t = self.step_s(shape.model, shape.gpu, shape.tp, shape.batch, hour) * math.exp(z * float(self.cfg["measurement_sigma"]))
        return Observation(shape, t * 1000.0, crashed)

    def measure_profile(self, model: str, gpu: str, tp: int, batch: int, hour: int, key: object) -> Profile:
        """A fresh measurement, e.g. when calibrating a GPU type the site just bought."""
        rng = keyed(self.seed, self.variant, "measure", model, gpu, tp, batch, hour, key)
        s = float(self.cfg["measurement_sigma"])
        svc = self.services[model]
        step = self.step_s(model, gpu, tp, batch, hour) * math.exp(rng.normal() * s) * 1000
        pre = self.prefill_s(model, gpu, tp, svc.prompt, hour) * math.exp(rng.normal() * s) * 1000
        return Profile(model, gpu, tp, batch, step, pre, svc.prompt)


def _services(cfg: dict[str, Any]) -> dict[str, Service]:
    return {m: Service(m, int(s["prompt"]), int(s["output"]), float(s["tpot_ms"]), float(s["ttft_s"])) for m, s in cfg["services"].items()}


def _demand(seed: int, variant: str, cfg: dict[str, Any], models: list[str]) -> np.ndarray:
    hours = int(cfg["hours"])
    tr = cfg["traffic"]
    out = np.zeros((hours, len(models)))
    for j, m in enumerate(models):
        rng = keyed(seed, variant, "demand", m)
        phase = rng.uniform(-2, 2)
        peak = float(cfg["services"][m]["peak_rps"])
        for h in range(hours):
            day = 0.5 * (1 - math.cos(2 * math.pi * ((h % 24) - 3 + phase) / 24))
            weekend = 0.8 if (h // 24) % 7 in (5, 6) else 1.0
            level = tr["trough"] + (1 - tr["trough"]) * day
            burst = tr["burst"]["factor"] if rng.random() < tr["burst"]["p"] else 1.0
            out[h, j] = peak * level * weekend * burst * math.exp(rng.normal() * tr["hourly_sigma"])
    return out


def build_site(seed: int, variant: str = "dev") -> Site:
    cfg = world_config(variant)
    services = _services(cfg)
    models = list(services)
    rng = keyed(seed, variant, "efficiency")
    eff: dict[str, dict[str, float]] = {}
    for g in GPUS:
        base = cfg["efficiency"][g]
        site_f = math.exp(rng.normal() * float(cfg["site_sigma"]))
        eff[g] = {
            "mem": base["mem"] * site_f,
            "compute": base["compute"] * math.exp(rng.normal() * float(cfg["site_sigma"])),
            "overhead_ms": base["overhead_ms"],
            "batch_penalty": base["batch_penalty"],
        }
    pair: dict[tuple[str, str], tuple[float, float]] = {}
    for m in MODELS:
        for g in GPUS:
            r = keyed(seed, variant, "pair", m, g)
            s = float(cfg["pair_sigma"])
            pair[(m, g)] = (math.exp(r.normal() * s), math.exp(r.normal() * s))
    site = Site(seed, variant, cfg, dict(cfg["fleet"]), services, eff, pair, _demand(seed, variant, cfg, models), models)
    site._profiles = _profiles(site)
    return site


def _profiles(site: Site) -> list[Profile]:
    pc = site.cfg["profiles"]
    held = {tuple(p) for p in pc.get("held_back_pairs", [])}
    never = set(pc.get("no_profiles", []))
    only_on: dict[str, list[str]] = pc.get("only_on", {})
    out: list[Profile] = []
    gpus = [g for g in site.fleet if g not in never]
    for m in site.models:
        for g in gpus:
            if (m, g) in held or (m in only_on and g not in only_on[m]):
                continue
            rng = keyed(site.seed, site.variant, "profiles", m, g)
            grid = [(tp, b) for tp in TP_DEGREES for b in BATCHES if site.fits(m, g, tp, b)]
            if not grid:
                continue
            idx = rng.choice(len(grid), size=min(len(grid), int(pc["points_per_pair"])), replace=False)
            for i in sorted(int(x) for x in idx):
                tp, b = grid[i]
                out.append(site.measure_profile(m, g, tp, b, 0, ("initial", i)))
    return out
