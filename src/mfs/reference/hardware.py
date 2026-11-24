"""Public reference data: accelerator data-sheet figures and model architectures.

Both the world and the platform may read this module. It is what anyone can look up; how close real kernels
get to these numbers on a given site is the hidden truth the world keeps.
Figures are approximate dense BF16/FP16 data-sheet values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gpu:
    name: str
    vendor: str
    generation: float  # rough architecture generation, comparable across vendors
    tflops: float  # dense BF16/FP16 tensor TFLOP/s
    hbm_tbs: float  # memory bandwidth TB/s
    mem_gb: float
    link_gbs: float  # GPU-to-GPU bandwidth inside a node, GB/s
    link: str  # nvlink | pcie | xgmi


GPUS: dict[str, Gpu] = {
    "V100": Gpu("V100", "nvidia", 7.0, 125.0, 0.90, 32.0, 300.0, "nvlink"),
    "A100": Gpu("A100", "nvidia", 8.0, 312.0, 2.04, 80.0, 600.0, "nvlink"),
    "L40S": Gpu("L40S", "nvidia", 8.9, 362.0, 0.864, 48.0, 64.0, "pcie"),
    "H100": Gpu("H100", "nvidia", 9.0, 989.0, 3.35, 80.0, 900.0, "nvlink"),
    "MI300X": Gpu("MI300X", "amd", 9.0, 1307.0, 5.30, 192.0, 896.0, "xgmi"),
    "B200": Gpu("B200", "nvidia", 10.0, 2250.0, 8.00, 180.0, 1800.0, "nvlink"),
}


@dataclass(frozen=True)
class LlmArch:
    name: str
    family: str  # dense | moe
    params_b: float  # total parameters, billions
    active_b: float  # parameters used per token, billions
    layers: int
    hidden: int
    kv_heads: int
    head_dim: int

    @property
    def kv_bytes_per_token(self) -> float:
        return 2.0 * self.layers * self.kv_heads * self.head_dim * 2.0  # K and V, 2 bytes each

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9 * 2.0


MODELS: dict[str, LlmArch] = {
    "llama-8b": LlmArch("llama-8b", "dense", 8.0, 8.0, 32, 4096, 8, 128),
    "llama-13b": LlmArch("llama-13b", "dense", 13.0, 13.0, 40, 5120, 40, 128),
    "codellama-34b": LlmArch("codellama-34b", "dense", 34.0, 34.0, 48, 8192, 8, 128),
    "llama-70b": LlmArch("llama-70b", "dense", 70.0, 70.0, 80, 8192, 8, 128),
    "mixtral-8x7b": LlmArch("mixtral-8x7b", "moe", 46.7, 12.9, 32, 4096, 8, 128),
}

TP_DEGREES = (1, 2, 4, 8)
BATCHES = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256)


def weights_read_bytes(m: LlmArch, batch: int) -> float:
    """Bytes of weights one decode step reads. A mixture of experts touches more experts as the batch grows."""
    if m.family != "moe":
        return m.weight_bytes
    import math

    share = 1.0 - math.exp(-batch / 4.0)
    return (m.active_b + (m.params_b - m.active_b) * share) * 1e9 * 2.0
