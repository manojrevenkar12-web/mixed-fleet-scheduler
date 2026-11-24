"""Types the world, the platform and the service share. Nothing here carries hidden truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Profile:
    """One measurement a site took: a decode step of `batch` sequences, or a prefill of `prompt` tokens."""

    model: str
    gpu: str
    tp: int
    batch: int
    step_ms: float
    prefill_ms: float | None = None
    prompt: int = 0


@dataclass(frozen=True)
class Service:
    model: str
    prompt: int
    output: int
    tpot_ms: float
    ttft_s: float


@dataclass(frozen=True)
class ReplicaShape:
    model: str
    gpu: str
    tp: int
    batch: int


@dataclass
class Replica:
    shape: ReplicaShape
    capacity_rps: float  # what the scheduler expects one replica to serve within the SLO
    step_ms: float  # what the scheduler expects a decode step to take at this batch
    source: str  # learned | fallback | roofline | uniform | oracle
    admitted_rps: float = 0.0


@dataclass(frozen=True)
class TrainingJob:
    """A fine-tuning job: `tokens` to process on `gpus` GPUs of one type. It runs on GPUs inference does not need
    and gives them back when inference does (preemption from its last checkpoint)."""

    id: int
    model: str
    gpus: int
    tokens: float
    arrival: int


@dataclass(frozen=True)
class TrainingProfile:
    model: str
    gpu: str
    gpus: int
    tokens_per_s: float


@dataclass(frozen=True)
class TrainingAssignment:
    job: int
    gpu: str
    gpus: int
    tokens_per_s: float  # what the scheduler expects
    reason: str  # stable | short


@dataclass(frozen=True)
class TrainingReport:
    """What a running job reports after an hour: its measured throughput, or that it ran out of memory."""

    job: int
    gpu: str
    gpus: int
    tokens_per_s: float
    crashed: bool = False
    done: bool = False


@dataclass
class Placement:
    hour: int
    replicas: list[Replica] = field(default_factory=list)
    shed_rps: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)
    training: list[TrainingAssignment] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)

    def gpus_used(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.replicas:
            out[r.shape.gpu] = out.get(r.shape.gpu, 0) + r.shape.tp
        return out

    def training_gpus(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.training:
            out[a.gpu] = out.get(a.gpu, 0) + a.gpus
        return out


@dataclass(frozen=True)
class Observation:
    """What the scheduler sees after an hour: the measured decode step of each running replica."""

    shape: ReplicaShape
    step_ms: float
    crashed: bool
