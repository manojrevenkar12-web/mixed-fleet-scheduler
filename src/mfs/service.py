"""The placement service: the platform behind a small HTTP API, for a cluster controller to call every hour.

PUT  /v1/site       {"fleet", "services", "seed"}             start a site
POST /v1/profiles   {"profiles": [...], "training_profiles": [...]}   add measured profiles, retrain
POST /v1/jobs       {"jobs": [...]}                           submit fine-tuning jobs (kept once per id)
POST /v1/place      {"hour", "forecast": {...}, "demand": {...}}  replicas, training assignments, preemptions
POST /v1/observe    {"hour", "observations": [...], "training": [...]}  what replicas and jobs measured
GET  /healthz       GET /metrics (Prometheus text)

The fleet and services come from the request that creates the state (PUT /v1/site). State (site, profiles,
observed hours) is kept in a JSON file; on restart the model is retrained from the same profiles with the same
seed, so the service gives the same answer. An hour's observations are kept once: a retried POST /v1/observe for
an hour already seen changes nothing and says so.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import config
from .domain import Observation, Profile, ReplicaShape, Service, TrainingJob, TrainingProfile, TrainingReport
from .platform.build import build_platform
from .platform.scheduler import Scheduler
from .reference.hardware import GPUS, MODELS


class BadRequest(ValueError):
    pass


class PlacementService:
    def __init__(self, state_path: str | Path | None = None) -> None:
        self.path = Path(state_path) if state_path else None
        self.lock = threading.Lock()
        self.site: dict[str, Any] | None = None
        self.profiles: list[Profile] = []
        self.training_profiles: list[TrainingProfile] = []
        self.jobs: list[TrainingJob] = []
        self.observed: list[int] = []
        self.sched: Scheduler | None = None
        self.counters = {"placements": 0, "observations": 0, "duplicate_observations": 0, "retrains": 0}
        self.last_plan_ms = 0.0
        if self.path and self.path.exists():
            st = json.loads(self.path.read_text(encoding="utf-8"))
            self.site, self.observed = st["site"], st["observed"]
            self.profiles = [Profile(**p) for p in st["profiles"]]
            self.training_profiles = [TrainingProfile(**p) for p in st.get("training_profiles", [])]
            self.jobs = [TrainingJob(**j) for j in st.get("jobs", [])]
            if self.profiles:
                self._retrain()

    # --------------------------------------------------------------- state
    def _save(self) -> None:
        if not self.path:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "site": self.site,
                    "profiles": [asdict(p) for p in self.profiles],
                    "training_profiles": [asdict(p) for p in self.training_profiles],
                    "jobs": [asdict(j) for j in self.jobs],
                    "observed": self.observed,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def _services(self) -> dict[str, Service]:
        assert self.site
        return {m: Service(m, **s) for m, s in self.site["services"].items()}

    def _retrain(self) -> None:
        assert self.site
        self.sched, _ = build_platform(self._services(), self.site["fleet"], self.profiles, config.platform(), seed=int(self.site.get("seed", 0)), training_profiles=self.training_profiles)
        for j in self.jobs:
            self.sched.submit(j)
        self.counters["retrains"] += 1

    # --------------------------------------------------------------- API
    def put_site(self, body: dict[str, Any]) -> dict[str, Any]:
        fleet, services = body.get("fleet"), body.get("services")
        if not isinstance(fleet, dict) or not fleet or not isinstance(services, dict) or not services:
            raise BadRequest("site needs a fleet {gpu: count} and services {model: {prompt, output, tpot_ms, ttft_s}}")
        for g, n in fleet.items():
            if g not in GPUS or not isinstance(n, int) or n < 0:
                raise BadRequest(f"unknown GPU type or count: {g}")
        for m, s in services.items():
            if m not in MODELS or set(s) != {"prompt", "output", "tpot_ms", "ttft_s"}:
                raise BadRequest(f"unknown model or service fields: {m}")
        with self.lock:
            self.site = {"fleet": fleet, "services": services, "seed": int(body.get("seed", 0))}
            self.profiles, self.training_profiles, self.jobs, self.observed, self.sched = [], [], [], [], None
            self._save()
        return {"ok": True}

    def add_profiles(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.site:
            raise BadRequest("PUT /v1/site first")
        try:
            new = [Profile(**p) for p in body["profiles"]]
            new_train = [TrainingProfile(**p) for p in body.get("training_profiles", [])]
        except (KeyError, TypeError) as e:
            raise BadRequest(f"profiles: {e}") from e
        for p in new:
            if p.model not in self.site["services"] or p.gpu not in self.site["fleet"] or p.step_ms <= 0:
                raise BadRequest(f"profile for an unknown model or GPU, or a non-positive step time: {p.model} on {p.gpu}")
        with self.lock:
            seen = set(self.profiles)
            added = [p for p in new if p not in seen]
            self.profiles.extend(added)
            added_train = [p for p in new_train if p not in set(self.training_profiles)]
            self.training_profiles.extend(added_train)
            if added or added_train:
                self._retrain()
            self._save()
        return {"added": len(added), "duplicates": len(new) - len(added), "profiles": len(self.profiles)}

    def submit_jobs(self, body: dict[str, Any]) -> dict[str, Any]:
        if not self.site:
            raise BadRequest("PUT /v1/site first")
        try:
            new = [TrainingJob(int(j["id"]), str(j["model"]), int(j["gpus"]), float(j["tokens"]), int(j["arrival"])) for j in body["jobs"]]
        except (KeyError, TypeError, ValueError) as e:
            raise BadRequest(f"jobs: {e}") from e
        for j in new:
            if j.gpus <= 0 or j.tokens <= 0:
                raise BadRequest(f"job {j.id}: gpus and tokens must be positive")
        with self.lock:
            known = {j.id for j in self.jobs}
            added = [j for j in new if j.id not in known]
            self.jobs.extend(added)
            if self.sched:
                for j in added:
                    self.sched.submit(j)
            self._save()
        return {"added": len(added), "duplicates": len(new) - len(added)}

    def place(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.sched is None:
            raise BadRequest("no profiles yet: POST /v1/profiles first")
        try:
            hour = int(body["hour"])
            forecast = {m: float(body["forecast"][m]) for m in self.sched.services}
            demand = {m: float(body.get("demand", body["forecast"])[m]) for m in self.sched.services}
        except (KeyError, TypeError, ValueError) as e:
            raise BadRequest(f"place needs hour and forecast per service: {e}") from e
        with self.lock:
            t = time.perf_counter()
            p = self.sched.plan(hour, forecast, demand)
            self.last_plan_ms = (time.perf_counter() - t) * 1000
            self.counters["placements"] += 1
        return {
            "hour": hour,
            "replicas": [
                {**asdict(r.shape), "planned_rps": round(r.capacity_rps, 3), "admit_rps": round(r.admitted_rps, 3), "planned_step_ms": round(r.step_ms, 3), "source": r.source} for r in p.replicas
            ],
            "shed_rps": {m: round(v, 3) for m, v in p.shed_rps.items()},
            "gpus_used": p.gpus_used(),
            "drifting": p.notes.get("drifting", []),
            "training": [asdict(a) for a in p.training],
            "preempted": p.preempted,
        }

    def observe(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.sched is None:
            raise BadRequest("no profiles yet")
        try:
            hour = int(body["hour"])
            obs = [Observation(ReplicaShape(o["model"], o["gpu"], int(o["tp"]), int(o["batch"])), float(o.get("step_ms", 0.0)), bool(o.get("crashed", False))) for o in body["observations"]]
            train = [
                TrainingReport(int(t["job"]), t["gpu"], int(t["gpus"]), float(t.get("tokens_per_s", 0.0)), bool(t.get("crashed", False)), bool(t.get("done", False))) for t in body.get("training", [])
            ]
        except (KeyError, TypeError, ValueError) as e:
            raise BadRequest(f"observations: {e}") from e
        with self.lock:
            if hour in self.observed:
                self.counters["duplicate_observations"] += 1
                return {"hour": hour, "applied": False, "reason": "this hour was already observed"}
            self.sched.observe(hour, [o for o in obs if o.crashed or o.step_ms > 0])
            self.sched.observe_training(hour, train)
            done = {t.job for t in train if t.done}
            self.jobs = [j for j in self.jobs if j.id not in done]
            self.observed.append(hour)
            self.counters["observations"] += len(obs)
            self._save()
        return {"hour": hour, "applied": True, "drifting": sorted(g for g, u in self.sched.drift_until.items() if hour <= u)}

    def metrics(self) -> str:
        lines = [f"mfs_{k}_total {v}" for k, v in self.counters.items()]
        lines += [f"mfs_profiles {len(self.profiles)}", f"mfs_last_plan_ms {self.last_plan_ms:.3f}", f"mfs_ready {int(self.sched is not None)}"]
        return "\n".join(lines) + "\n"


def make_handler(svc: PlacementService) -> type[BaseHTTPRequestHandler]:
    routes = {
        ("PUT", "/v1/site"): svc.put_site,
        ("POST", "/v1/profiles"): svc.add_profiles,
        ("POST", "/v1/jobs"): svc.submit_jobs,
        ("POST", "/v1/place"): svc.place,
        ("POST", "/v1/observe"): svc.observe,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # quiet
            return

        def _send(self, code: int, body: Any, ctype: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"ok": True, "ready": svc.sched is not None})
            elif self.path == "/metrics":
                self._send(200, svc.metrics(), "text/plain; version=0.0.4")
            else:
                self._send(404, {"error": "not found"})

        def _body(self) -> None:
            fn = routes.get((self.command, self.path))
            if fn is None:
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 8_000_000:
                    raise BadRequest("body too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("body must be a JSON object")
                self._send(200, fn(body))
            except (BadRequest, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)})

        do_POST = do_PUT = _body

    return Handler


def serve(port: int, state: str) -> None:
    svc = PlacementService(state)
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(svc))  # noqa: S104 - container port
    print(f"serving on :{port}, state in {state}")
    server.serve_forever()
