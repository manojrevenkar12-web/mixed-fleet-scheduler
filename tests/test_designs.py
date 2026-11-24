from __future__ import annotations

from typing import Any

import pytest

from mfs import config
from mfs.sim.designs import run_design
from mfs.sim.evaluate import run_metrics
from mfs.world.world import Site


@pytest.fixture(scope="module")
def results(tiny: Site, fast_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    d = config.designs()["designs"]
    return {n: run_metrics(tiny, run_design(tiny, n, d[n], fast_cfg)) for n in ("uniform", "roofline_calibrated", "platform", "no_admission", "oracle")}


def test_the_platform_beats_the_scheduler_built_for_identical_gpus(results: dict[str, dict[str, Any]]) -> None:
    assert results["platform"]["attainment"] > results["uniform"]["attainment"] + 0.1


def test_the_platform_is_close_to_the_oracle_and_never_runs_out_of_memory(results: dict[str, dict[str, Any]]) -> None:
    assert results["platform"]["attainment"] > results["oracle"]["attainment"] - 0.05
    assert results["platform"]["crashes"] == 0


def test_admission_control_turns_misses_into_explicit_sheds(results: dict[str, dict[str, Any]]) -> None:
    assert results["platform"]["violated_share"] <= results["no_admission"]["violated_share"]


def test_every_design_meets_the_same_traffic(results: dict[str, dict[str, Any]]) -> None:
    assert len({round(r["requests"], 6) for r in results.values()}) == 1


def test_every_design_in_config_has_a_known_kind() -> None:
    d = config.designs()
    kinds = {"uniform", "roofline", "roofline_calibrated", "learned", "oracle"}
    for name, spec in d["designs"].items():
        assert spec["kind"] in kinds, name
        assert set(spec.get("without", [])) <= {"margin", "fallback", "feedback", "admission", "peak_aware"}
    assert set(d["main"]) | set(d["ablations"]) <= set(d["designs"])
