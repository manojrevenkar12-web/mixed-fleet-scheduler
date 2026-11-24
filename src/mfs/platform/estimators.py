"""How long will a decode step take? Three answers behind one interface.

- `Roofline`: the data sheet times one efficiency for every GPU. Cheap, explainable, no data needed.
- `Uniform`: what a scheduler built for identical hardware assumes: every GPU is an A100.
- `Learned` (perfmodel.py): a PyTorch ensemble trained on the site's measurements, with an uncertainty, a
  fallback to the calibrated roofline where it is unsure, and measured feedback.

Every estimator returns (mean, upper) seconds, where `upper` is what the planner plans on.

Training throughput (tokens per second of a fine-tuning job) is the calibrated roofline in every design that
has measurements: the learned ensemble covers inference decode, where the money is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..domain import Service, TrainingProfile
from ..reference.hardware import GPUS, MODELS, weights_read_bytes


class Estimator(Protocol):
    name: str

    def step(self, model: str, gpu: str, tp: int, batch: int) -> tuple[float, float, str]: ...

    def prefill(self, model: str, gpu: str, tp: int, prompt: int) -> float: ...

    def fits(self, model: str, gpu: str, tp: int, batch: int) -> bool: ...

    def train_rate(self, model: str, gpu: str, gpus: int) -> float: ...

    def train_fits(self, model: str, gpu: str, gpus: int) -> bool: ...


def roofline_parts(model: str, gpu: str, tp: int, batch: int, svc: Service, cfg: dict[str, Any]) -> tuple[float, float, float]:
    """(memory time, compute time, total) for one decode step from public figures only."""
    m, g, rc = MODELS[model], GPUS[gpu], cfg["roofline"]
    ctx = svc.prompt + svc.output / 2
    t_mem = (weights_read_bytes(m, batch) / tp + batch * ctx * m.kv_bytes_per_token / tp) / (g.hbm_tbs * 1e12 * rc["mem_efficiency"])
    t_cmp = 2.0 * m.active_b * 1e9 * batch / tp / (g.tflops * 1e12 * rc["compute_efficiency"])
    t = max(t_mem, t_cmp) + rc["overhead_ms"] / 1000.0 * m.layers / 32.0
    if tp > 1:
        t += m.layers * 2 * (batch * m.hidden * 4.0 * (tp - 1) / tp / (g.link_gbs * 1e9) + 10e-6)
    return t_mem, t_cmp, t


def roofline_prefill(model: str, gpu: str, tp: int, prompt: int, cfg: dict[str, Any]) -> float:
    m, g, rc = MODELS[model], GPUS[gpu], cfg["roofline"]
    return 2.0 * m.active_b * 1e9 * prompt / tp / (g.tflops * 1e12 * rc["prefill_efficiency"]) + rc["overhead_ms"] / 1000.0 * m.layers / 8.0


def planned_fit(model: str, gpu: str, tp: int, batch: int, svc: Service, cfg: dict[str, Any]) -> bool:
    m, g, mc = MODELS[model], GPUS[gpu], cfg["memory"]
    need = (m.weight_bytes / tp + batch * (svc.prompt + svc.output) * m.kv_bytes_per_token / tp) / 1e9 + mc["activation_gb_per_64"] * batch / 64.0
    return need <= g.mem_gb * (1 - mc["reserve_share"]) - mc["reserve_gb"]


def roofline_train_rate(model: str, gpu: str, gpus: int, cfg: dict[str, Any]) -> float:
    """Tokens per second of a fine-tuning job from public figures: 6 FLOPs per parameter per token."""
    m, g, tc = MODELS[model], GPUS[gpu], cfg["training"]
    return gpus * g.tflops * 1e12 * float(cfg["roofline"]["compute_efficiency"]) * float(tc["assumed_mfu"]) / (6.0 * m.active_b * 1e9)


def planned_train_fit(model: str, gpu: str, gpus: int, cfg: dict[str, Any]) -> bool:
    m, g, mc, tc = MODELS[model], GPUS[gpu], cfg["memory"], cfg["training"]
    need = m.params_b * float(tc["bytes_per_param"]) / gpus + float(tc["activation_gb"])
    return need <= g.mem_gb * (1 - mc["reserve_share"]) - mc["reserve_gb"]


@dataclass
class Calibration:
    """Measured/predicted ratios that flow back from running replicas and from fresh profiles.

    Kept per (model, GPU, tensor-parallel degree, batch bucket), because a prediction can be right at small
    batches and wrong at large ones; a query falls back to the nearest bucket, then the pair, then the GPU.
    """

    alpha: float
    bucket: dict[tuple[str, str, int, int], float] = field(default_factory=dict)
    pair: dict[tuple[str, str, int], float] = field(default_factory=dict)
    gpu: dict[str, float] = field(default_factory=dict)
    prefill: dict[str, float] = field(default_factory=dict)
    train: dict[str, float] = field(default_factory=dict)  # per GPU: log(measured / roofline) training throughput

    @staticmethod
    def _b(batch: int) -> int:
        return int(round(math.log2(max(batch, 1))))

    def log_factor(self, model: str, gpu: str, tp: int, batch: int) -> float:
        k = self._b(batch)
        for d in (0, -1, 1):
            v = self.bucket.get((model, gpu, tp, k + d))
            if v is not None:
                return v
        return self.pair.get((model, gpu, tp), self.gpu.get(gpu, 0.0))

    def factor(self, model: str, gpu: str, tp: int, batch: int = 32) -> float:
        return math.exp(self.log_factor(model, gpu, tp, batch))

    def update(self, model: str, gpu: str, tp: int, batch: int, log_ratio: float) -> None:
        """`log_ratio` is log(measured / predicted), the prediction already including the current factor."""
        old = self.log_factor(model, gpu, tp, batch)
        self.bucket[(model, gpu, tp, self._b(batch))] = old + self.alpha * log_ratio
        key = (model, gpu, tp)
        self.pair[key] = self.pair.get(key, self.gpu.get(gpu, 0.0)) + self.alpha * log_ratio / 2

    def update_gpu(self, gpu: str, log_ratio: float) -> None:
        self.gpu[gpu] = self.gpu.get(gpu, 0.0) + self.alpha * log_ratio

    def train_factor(self, gpu: str) -> float:
        if gpu in self.train:
            return math.exp(self.train[gpu])
        return math.exp(sorted(self.train.values())[len(self.train) // 2]) if self.train else 1.0

    def update_train(self, gpu: str, log_ratio: float) -> None:
        """`log_ratio` is log(measured / predicted), the prediction already including the current factor."""
        self.train[gpu] = math.log(self.train_factor(gpu)) + self.alpha * log_ratio


def train_ratios(profiles: list[TrainingProfile], cfg: dict[str, Any]) -> dict[str, float]:
    by_gpu: dict[str, list[float]] = {}
    for p in profiles:
        by_gpu.setdefault(p.gpu, []).append(math.log(p.tokens_per_s / roofline_train_rate(p.model, p.gpu, p.gpus, cfg)))
    return {g: sorted(v)[len(v) // 2] for g, v in by_gpu.items()}


class Roofline:
    name = "roofline"

    def __init__(self, cfg: dict[str, Any], services: dict[str, Service], calibration: Calibration | None = None) -> None:
        self.cfg, self.services, self.cal = cfg, services, calibration

    def step(self, model: str, gpu: str, tp: int, batch: int) -> tuple[float, float, str]:
        t = roofline_parts(model, gpu, tp, batch, self.services[model], self.cfg)[2]
        if self.cal:
            t *= self.cal.factor(model, gpu, tp, batch)
        return t, t, "roofline"

    def prefill(self, model: str, gpu: str, tp: int, prompt: int) -> float:
        t = roofline_prefill(model, gpu, tp, prompt, self.cfg)
        if self.cal and gpu in self.cal.prefill:
            t *= math.exp(self.cal.prefill[gpu])
        return t

    def fits(self, model: str, gpu: str, tp: int, batch: int) -> bool:
        return planned_fit(model, gpu, tp, batch, self.services[model], self.cfg)

    def train_rate(self, model: str, gpu: str, gpus: int) -> float:
        r = roofline_train_rate(model, gpu, gpus, self.cfg)
        return r * self.cal.train_factor(gpu) if self.cal else r

    def train_fits(self, model: str, gpu: str, gpus: int) -> bool:
        return planned_train_fit(model, gpu, gpus, self.cfg)


class Uniform(Roofline):
    """A scheduler built for identical hardware: every GPU is treated as an A100, memory included."""

    name = "uniform"

    def step(self, model: str, gpu: str, tp: int, batch: int) -> tuple[float, float, str]:
        t = roofline_parts(model, "A100", tp, batch, self.services[model], self.cfg)[2]
        return t, t, "uniform"

    def prefill(self, model: str, gpu: str, tp: int, prompt: int) -> float:
        return roofline_prefill(model, "A100", tp, prompt, self.cfg)

    def fits(self, model: str, gpu: str, tp: int, batch: int) -> bool:
        return planned_fit(model, "A100", tp, batch, self.services[model], self.cfg)

    def train_rate(self, model: str, gpu: str, gpus: int) -> float:
        return roofline_train_rate(model, "A100", gpus, self.cfg)

    def train_fits(self, model: str, gpu: str, gpus: int) -> bool:
        return planned_train_fit(model, "A100", gpus, self.cfg)
