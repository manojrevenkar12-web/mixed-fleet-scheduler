from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from mfs import service as service_mod
from mfs.service import BadRequest, PlacementService, make_handler
from mfs.world.world import Site


@pytest.fixture()
def svc(tiny: Site, fast_cfg: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PlacementService:
    monkeypatch.setattr(service_mod.config, "platform", lambda: fast_cfg)
    s = PlacementService(tmp_path / "state.json")
    s.put_site({"fleet": tiny.fleet, "services": {m: {k: v for k, v in asdict(x).items() if k != "model"} for m, x in tiny.services.items()}, "seed": tiny.seed})
    s.add_profiles({"profiles": [asdict(p) for p in tiny.profiles()]})
    return s


def _demand(site: Site, h: int) -> dict[str, float]:
    return {m: float(site.demand[h, j]) for j, m in enumerate(site.models)}


def test_place_and_observe(svc: PlacementService, tiny: Site) -> None:
    out = svc.place({"hour": 3, "forecast": _demand(tiny, 3)})
    assert out["replicas"] and all(r["source"] in {"learned", "fallback"} for r in out["replicas"])
    obs = [{"model": r["model"], "gpu": r["gpu"], "tp": r["tp"], "batch": r["batch"], "step_ms": r["planned_step_ms"]} for r in out["replicas"]]
    assert svc.observe({"hour": 3, "observations": obs})["applied"]
    again = svc.observe({"hour": 3, "observations": obs})
    assert not again["applied"] and svc.counters["duplicate_observations"] == 1


def test_jobs_are_kept_once_and_placed(svc: PlacementService, tiny: Site) -> None:
    jobs = [asdict(j) for j in tiny.training_jobs()]
    assert svc.submit_jobs({"jobs": jobs})["added"] == len(jobs)
    assert svc.submit_jobs({"jobs": jobs[:2]}) == {"added": 0, "duplicates": 2}
    quiet = {m: v * 0.2 for m, v in _demand(tiny, 13).items()}
    out = svc.place({"hour": 13, "forecast": quiet})
    assert out["training"], "free GPUs at night go to the queued jobs"
    t = out["training"][0]
    svc.observe({"hour": 13, "observations": [], "training": [{"job": t["job"], "gpu": t["gpu"], "gpus": t["gpus"], "tokens_per_s": 1.0, "done": True}]})
    assert t["job"] not in {j.id for j in svc.jobs}


def test_duplicate_profiles_are_not_added_twice(svc: PlacementService, tiny: Site) -> None:
    r = svc.add_profiles({"profiles": [asdict(tiny.profiles()[0])]})
    assert r == {"added": 0, "duplicates": 1, "profiles": len(tiny.profiles())}


def test_restart_gives_the_same_placement(svc: PlacementService, tiny: Site) -> None:
    first = svc.place({"hour": 5, "forecast": _demand(tiny, 5)})
    assert svc.path is not None
    restarted = PlacementService(svc.path)
    assert restarted.place({"hour": 5, "forecast": _demand(tiny, 5)}) == first


def test_bad_requests_are_refused(svc: PlacementService) -> None:
    with pytest.raises(BadRequest):
        svc.place({"hour": 1})
    with pytest.raises(BadRequest):
        svc.put_site({"fleet": {"TPU": 4}, "services": {"llama-8b": {}}})
    with pytest.raises(BadRequest):
        svc.add_profiles({"profiles": [{"model": "llama-8b", "gpu": "A100", "tp": 1, "batch": 1, "step_ms": -1}]})


def test_http_api(svc: PlacementService, tiny: Site) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(svc))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/healthz") as r:
            assert json.loads(r.read())["ready"] is True
        req = urllib.request.Request(base + "/v1/place", data=json.dumps({"hour": 2, "forecast": _demand(tiny, 2)}).encode(), method="POST")
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["hour"] == 2
        with urllib.request.urlopen(base + "/metrics") as r:
            assert "mfs_placements_total 1" in r.read().decode()
        bad = urllib.request.Request(base + "/v1/place", data=b"[1]", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        server.shutdown()
