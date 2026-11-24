from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mfs import config
from mfs.lab import report, study


def test_heldout_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study, "REPORTS", tmp_path)
    monkeypatch.setattr(study, "LOCK", tmp_path / "heldout.lock")
    fake: dict[str, Any] = {"designs": {}, "gates": {"platform_pass": True}}
    monkeypatch.setattr(study, "run_split", lambda split, workers=3: fake)
    study.run_heldout_once()
    assert (tmp_path / "heldout.lock").exists()
    with pytest.raises(study.HeldOutLocked):
        study.run_heldout_once()


def test_gates_compare_with_their_limits() -> None:
    row = {"attainment": 0.96, "violated_share": 0.004, "crashes": 0, "bad_service_hours": 5, "service_hours": 100, "gpu_utilisation": 0.5, "fleet_busy": 0.7, "training_crashes": 0}
    d = {k: dict(row) for k in ("platform", "roofline_calibrated", "uniform", "oracle")}
    d["uniform"]["attainment"] = 0.5
    d["roofline_calibrated"]["violated_share"] = 0.02
    acc = {"unmeasured_pairs": {"learned_mape": 0.08, "interval_coverage": 0.9}}
    gates = {"platform": config.gates()["platform"]}
    assert study.check_gates({"designs": d, "accuracy": acc}, gates)["platform_pass"]
    d["platform"]["crashes"] = 1
    assert not study.check_gates({"designs": d, "accuracy": acc}, gates)["platform_pass"]


def test_the_readme_block_is_generated_from_the_reports() -> None:
    dev = report._load("dev")
    assert dev is not None
    block = report.readme_block(dev, None, None)
    assert block.startswith(report.START) and block.endswith(report.END)
    assert f"{100 * dev['designs']['platform']['attainment']:.1f}%" in block
