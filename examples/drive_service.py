"""Replays a day of a simulated site through the placement service's HTTP API, the way a cluster controller would:
profiles first, then every hour a placement, and the measured step times of what ran.

    python examples/drive_service.py            # starts the service in-process on a free port
"""

from __future__ import annotations

import json
import threading
import urllib.request
from dataclasses import asdict
from http.server import ThreadingHTTPServer

from mfs.domain import ReplicaShape
from mfs.service import PlacementService, make_handler
from mfs.world.world import build_site


def call(base: str, method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(), method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main() -> None:
    site = build_site(21, "dev")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(PlacementService()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    services = {m: {k: v for k, v in asdict(s).items() if k != "model"} for m, s in site.services.items()}
    call(base, "PUT", "/v1/site", {"fleet": site.fleet, "services": services, "seed": site.seed})
    print(call(base, "POST", "/v1/profiles", {"profiles": [asdict(p) for p in site.profiles()], "training_profiles": [asdict(p) for p in site.training_profiles()]}))
    jobs = {j.id: j for j in site.training_jobs() if j.arrival < 24}
    print(call(base, "POST", "/v1/jobs", {"jobs": [asdict(j) for j in jobs.values()]}))
    for h in range(24):
        demand = {m: float(site.demand[h, j]) for j, m in enumerate(site.models)}
        plan = call(base, "POST", "/v1/place", {"hour": h, "forecast": demand})
        obs = []
        for r in plan["replicas"]:
            o = site.observe(ReplicaShape(r["model"], r["gpu"], r["tp"], r["batch"]), h, False)
            obs.append({"model": r["model"], "gpu": r["gpu"], "tp": r["tp"], "batch": r["batch"], "step_ms": o.step_ms})
        train = [{"job": t["job"], "gpu": t["gpu"], "gpus": t["gpus"], "tokens_per_s": site.observe_training(t["job"], jobs[t["job"]].model, t["gpu"], t["gpus"], h)} for t in plan["training"]]
        out = call(base, "POST", "/v1/observe", {"hour": h, "observations": obs, "training": train})
        used = sum(plan["gpus_used"].values())
        print(
            f"hour {h:2d}: {len(plan['replicas']):3d} replicas on {used:3d} GPUs, {len(plan['training'])} jobs training, shed {sum(plan['shed_rps'].values()):6.1f} rps, drifting {out.get('drifting', [])}"
        )
    server.shutdown()


if __name__ == "__main__":
    main()
