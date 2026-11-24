"""The learned performance model: a small PyTorch ensemble that predicts the decode-step time of a model on a
GPU at a tensor-parallel degree and batch size, as a correction on top of the roofline.

Learning the residual log(measured / roofline) keeps the physics the data sheet already explains and lets
the network learn what it misses: kernel maturity per vendor, launch overhead, large-batch slowdowns. The
members' spread is the uncertainty the planner uses for its margin and its fallback. An unseen GPU or model
gets a prediction from its public figures and a wide spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..domain import Profile, Service
from ..reference.hardware import GPUS, MODELS
from .estimators import Calibration, planned_fit, planned_train_fit, roofline_parts, roofline_prefill, roofline_train_rate

LINKS = ("nvlink", "pcie", "xgmi")
_FITTED: dict[tuple[object, ...], tuple[dict[str, Any], float, TrainReport]] = {}


def features(model: str, gpu: str, tp: int, batch: int, svc: Service, cfg: dict[str, Any]) -> list[float]:
    m, g = MODELS[model], GPUS[gpu]
    t_mem, t_cmp, t = roofline_parts(model, gpu, tp, batch, svc, cfg)
    return [
        math.log(m.params_b),
        math.log(m.active_b),
        m.layers / 80.0,
        m.hidden / 8192.0,
        m.kv_bytes_per_token / 1e6,
        1.0 if m.family == "moe" else 0.0,
        math.log(g.tflops),
        math.log(g.hbm_tbs),
        g.mem_gb / 192.0,
        math.log(g.link_gbs),
        *[1.0 if g.link == k else 0.0 for k in LINKS],
        1.0 if g.vendor == "amd" else 0.0,
        g.generation / 10.0,
        math.log(tp),
        math.log(batch),
        math.log(svc.prompt + svc.output),
        math.log(t_mem),
        math.log(t_cmp),
        math.log(t_mem / t_cmp),
    ]


class _Mlp(nn.Module):
    def __init__(self, n_in: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainReport:
    points: int
    members: int
    train_rmse_log: float


class PerfModel:
    def __init__(self, cfg: dict[str, Any], services: dict[str, Service], seed: int = 0) -> None:
        self.cfg = cfg
        self.pc = cfg["perf_model"]
        self.services = services
        self.seed = seed
        self.members: list[_Mlp] = []
        self.mu = np.zeros(1)
        self.sd = np.ones(1)
        self._cache: dict[tuple[str, str, int, int], tuple[float, float]] = {}
        self.std_scale = 1.0

    def _x(self, rows: list[list[float]]) -> torch.Tensor:
        a = (np.asarray(rows, dtype=np.float64) - self.mu) / self.sd
        return torch.as_tensor(a, dtype=torch.float32)

    def fit(self, profiles: list[Profile]) -> TrainReport:
        """Train on all profiles. The spread is first calibrated on held-out (model, GPU) pairs, a fifth of them:
        the members' std is scaled so that 90% of measurements on pairs the model did not see fall inside
        mean +- 1.645 std (split-conformal style). The scheduler leans on the spread exactly where it has no data."""
        key = (self.seed, tuple(profiles))
        if key in _FITTED:
            state, scale, rep = _FITTED[key]
            self.load_state_dict(state)
            self.std_scale = scale
            return rep
        gen = np.random.default_rng(self.seed + 991)
        pairs = sorted({(p.model, p.gpu) for p in profiles})
        held = {pairs[i] for i in gen.choice(len(pairs), size=max(1, int(len(pairs) * float(self.pc["calibration_share"]))), replace=False)} if pairs else set()
        mask = np.array([(p.model, p.gpu) in held for p in profiles])
        if mask.sum() >= 10 and (~mask).sum() >= 20:
            self._train([p for p, m in zip(profiles, mask, strict=True) if not m])
            z = []
            for p in (p for p, m in zip(profiles, mask, strict=True) if m):
                mean, std = self.predict_log(p.model, p.gpu, p.tp, p.batch)
                z.append(abs(math.log(p.step_ms / 1000.0) - mean) / max(std, 1e-6))
            self.std_scale = float(np.clip(np.quantile(z, 0.9) / 1.645, 1.0, 4.0))
        rep = self._train(profiles)
        _FITTED[key] = (self.state_dict(), self.std_scale, rep)
        return rep

    def _train(self, profiles: list[Profile]) -> TrainReport:
        torch.set_num_threads(int(self.pc["threads"]))
        scale = self.std_scale
        self.std_scale = 1.0
        rows, ys = [], []
        for p in profiles:
            svc = self.services[p.model]
            rows.append(features(p.model, p.gpu, p.tp, p.batch, svc, self.cfg))
            ys.append(math.log(p.step_ms / 1000.0) - math.log(roofline_parts(p.model, p.gpu, p.tp, p.batch, svc, self.cfg)[2]))
        x = np.asarray(rows)
        self.mu, self.sd = x.mean(axis=0), x.std(axis=0) + 1e-6
        y = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
        xt = self._x(rows)
        self.members = []
        errs = []
        for k in range(int(self.pc["members"])):
            torch.manual_seed(self.seed * 101 + k)
            gen = np.random.default_rng(self.seed * 101 + k)
            idx = gen.integers(0, len(ys), len(ys)) if self.pc["bootstrap"] else np.arange(len(ys))
            net = _Mlp(xt.shape[1], int(self.pc["hidden"]))
            opt = torch.optim.Adam(net.parameters(), lr=float(self.pc["lr"]), weight_decay=float(self.pc["weight_decay"]))
            xi, yi = xt[idx], y[idx]
            for _ in range(int(self.pc["epochs"])):
                opt.zero_grad()
                loss = torch.mean((net(xi) - yi) ** 2)
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                errs.append(float(torch.sqrt(torch.mean((net(xt) - y) ** 2))))
            self.members.append(net)
        self._cache.clear()
        self.std_scale = scale
        return TrainReport(len(ys), len(self.members), float(np.mean(errs)))

    def predict_log(self, model: str, gpu: str, tp: int, batch: int) -> tuple[float, float]:
        """(mean, std) of log step seconds."""
        key = (model, gpu, tp, batch)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        svc = self.services[model]
        x = self._x([features(model, gpu, tp, batch, svc, self.cfg)])
        with torch.no_grad():
            preds = np.array([float(net(x)[0]) for net in self.members])
        base = math.log(roofline_parts(model, gpu, tp, batch, svc, self.cfg)[2])
        out = (base + float(preds.mean()), float(preds.std()) * self.std_scale)
        self._cache[key] = out
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "members": [{k: v.tolist() for k, v in net.state_dict().items()} for net in self.members]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mu, self.sd = np.asarray(state["mu"]), np.asarray(state["sd"])
        self.members = []
        for s in state["members"]:
            net = _Mlp(len(self.mu), int(self.pc["hidden"]))
            net.load_state_dict({k: torch.as_tensor(v) for k, v in s.items()})
            net.eval()
            self.members.append(net)
        self._cache.clear()


class Learned:
    """The platform's estimator: ensemble mean, an upper bound from its spread, the calibrated roofline where
    the ensemble is unsure, and measured feedback on top of both."""

    name = "learned"

    def __init__(self, model: PerfModel, cfg: dict[str, Any], services: dict[str, Service], calibration: Calibration, *, margin: bool = True, fallback: bool = True) -> None:
        self.pm, self.cfg, self.services, self.cal = model, cfg, services, calibration
        self.margin, self.fallback = margin, fallback
        self.drifting: set[str] = set()
        self.roof_ratio: dict[str, float] = {}  # per GPU: median log(measured / roofline) over the site's profiles

    def fit_fallback(self, profiles: list[Profile]) -> None:
        self.roof_ratio = step_ratios(profiles, self.cfg, self.services)

    def step(self, model: str, gpu: str, tp: int, batch: int) -> tuple[float, float, str]:
        g = self.cfg["guardrails"]
        mean_log, std_log = self.pm.predict_log(model, gpu, tp, batch)
        factor = self.cal.factor(model, gpu, tp, batch)
        z = float(g["margin_z"]) if self.margin else 0.0
        if gpu in self.drifting:
            z = max(z, float(g["drift_margin_z"]))
        mean = math.exp(mean_log) * factor
        upper = math.exp(mean_log + z * std_log) * factor
        if self.fallback and std_log > float(g["fallback_std"]):
            # far outside what it was trained on: plan on the slower of the model and the site-calibrated
            # roofline, never on less than the model's own upper bound
            ratio = self.roof_ratio.get(gpu, float(np.median(list(self.roof_ratio.values()))) if self.roof_ratio else 0.0)
            roof = roofline_parts(model, gpu, tp, batch, self.services[model], self.cfg)[2] * math.exp(ratio) * factor
            t = max(roof, mean)
            return t, max(t * (1 + float(g["fallback_margin"])), upper), "fallback"
        return mean, upper, "learned"

    def prefill(self, model: str, gpu: str, tp: int, prompt: int) -> float:
        t = roofline_prefill(model, gpu, tp, prompt, self.cfg)
        lr = self.cal.prefill.get(gpu)
        if lr is None and self.cal.prefill:
            lr = float(np.median(list(self.cal.prefill.values())))
        return t * math.exp(lr or 0.0)

    def fits(self, model: str, gpu: str, tp: int, batch: int) -> bool:
        return planned_fit(model, gpu, tp, batch, self.services[model], self.cfg)

    def train_rate(self, model: str, gpu: str, gpus: int) -> float:
        return roofline_train_rate(model, gpu, gpus, self.cfg) * self.cal.train_factor(gpu)

    def train_fits(self, model: str, gpu: str, gpus: int) -> bool:
        return planned_train_fit(model, gpu, gpus, self.cfg)


def prefill_calibration(profiles: list[Profile], cfg: dict[str, Any], min_profiles: int) -> dict[str, float]:
    by_gpu: dict[str, list[float]] = {}
    for p in profiles:
        if p.prefill_ms:
            r = math.log(p.prefill_ms / 1000.0 / roofline_prefill(p.model, p.gpu, p.tp, p.prompt, cfg))
            by_gpu.setdefault(p.gpu, []).append(r)
    return {g: float(np.median(v)) for g, v in by_gpu.items() if len(v) >= min_profiles}


def step_ratios(profiles: list[Profile], cfg: dict[str, Any], services: dict[str, Service]) -> dict[str, float]:
    """The roofline's one free parameter per GPU: how far the site's measured steps sit from it (median)."""
    by_gpu: dict[str, list[float]] = {}
    for p in profiles:
        t = roofline_parts(p.model, p.gpu, p.tp, p.batch, services[p.model], cfg)[2]
        by_gpu.setdefault(p.gpu, []).append(math.log(p.step_ms / 1000.0 / t))
    return {g: float(np.median(v)) for g, v in by_gpu.items()}
