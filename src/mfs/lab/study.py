"""Runs every design on a split's sites, grades them, checks the gates and writes reports/<split>.json."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..sim.designs import run_design
from ..sim.evaluate import expansion_forecast, model_accuracy, run_metrics
from ..world.world import build_site

REPORTS = config.REPORTS
LOCK = REPORTS / "heldout.lock"
SUMMED = {"training_crashes", "training_preemptions", "training_jobs", "crashes", "tpot_misses", "overloads", "bad_service_hours", "service_hours", "drift_events", "requests", "train_points"}


class HeldOutLocked(RuntimeError):
    pass


def _round(x: Any) -> Any:
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return None
        return round(x, 5)
    if isinstance(x, dict):
        return {str(k): _round(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v) for v in x]
    return x


def config_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(config.CONFIG.glob("*.yml")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _samples(site: Any, runs: dict[str, Any]) -> dict[str, Any]:
    series = {}
    for name in ("roofline_calibrated", "platform", "oracle", "no_feedback"):
        r = runs[name]
        series[name] = [
            {
                "hour": h.hour,
                "attainment": sum(h.met.values()) / sum(h.offered.values()),
                "gpus": sum(h.gpus.values()),
                "shed": sum(h.shed.values()) / sum(h.offered.values()),
                "train_gpus": h.train_gpu_hours,
                "preempted": h.preempted,
            }
            for h in r.hours
        ]
    occupancy = []
    for pl in runs["platform"].placements:
        inf: dict[str, dict[str, int]] = {}
        for rep in pl.replicas:
            g = inf.setdefault(rep.shape.gpu, {})
            g[rep.shape.model] = g.get(rep.shape.model, 0) + rep.shape.tp
        occupancy.append({"inf": inf, "train": pl.training_gpus(), "drifting": pl.notes.get("drifting", []), "preempted": len(pl.preempted)})
    demand = [float(site.demand[h].sum()) for h in range(site.hours)]
    peak = int(np.argmax(demand))
    p = runs["platform"].placements[peak]
    mix: dict[str, dict[str, int]] = {}
    for rep in p.replicas:
        mix.setdefault(rep.shape.model, {})
        mix[rep.shape.model][rep.shape.gpu] = mix[rep.shape.model].get(rep.shape.gpu, 0) + rep.shape.tp
    shapes = sorted({(r.shape.model, r.shape.gpu, r.shape.tp, r.shape.batch, r.source) for r in p.replicas})
    return {
        "hours": series,
        "demand_rps": demand,
        "peak_hour": peak,
        "peak_mix_gpus": mix,
        "peak_shapes": [{"model": m, "gpu": g, "tp": tp, "batch": b, "source": s} for m, g, tp, b, s in shapes][:24],
        "fleet": site.fleet,
        "drift": site.cfg.get("drift", []),
        "events": [e for e in runs["platform"].log if e.get("event") == "drift"][:12],
        "train": runs["platform"].train,
        "occupancy": occupancy,
        "models": list(site.models),
    }


def _seed_job(args: tuple[int, str, bool]) -> dict[str, Any]:
    seed, variant, first = args
    site = build_site(seed, variant)
    cfg = config.platform()
    dcfg = config.designs()
    out: dict[str, Any] = {"seed": seed, "designs": {}}
    runs = {}
    for name, d in dcfg["designs"].items():
        runs[name] = run_design(site, name, d, cfg, keep_placements=first and name == "platform")
        out["designs"][name] = run_metrics(site, runs[name])
    out["accuracy"] = model_accuracy(site, cfg)
    out["expansion"] = expansion_forecast(site, cfg)
    out["site"] = {"gpus": sum(site.fleet.values()), "gpu_types": len(site.fleet), "services": len(site.models), "hours": site.hours, "profiles": len(site.profiles())}
    if first:
        out["samples"] = _samples(site, runs)
    return _round(out)


def _mean(rows: list[dict[str, Any]]) -> dict[str, Any]:
    m: dict[str, Any] = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if isinstance(vals[0], dict):
            m[k] = _mean(vals)
        elif isinstance(vals[0], str):
            m[k] = vals[0]
        elif k in SUMMED:
            m[k] = sum(v or 0 for v in vals)
        else:
            xs = [v for v in vals if v is not None]
            m[k] = float(np.mean(xs)) if xs else None
    return m


def check_gates(res: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
    d = res["designs"]
    p, rc, u, o = d["platform"], d["roofline_calibrated"], d["uniform"], d["oracle"]
    acc = res["accuracy"]
    gp = g["platform"]
    rows = [
        ("Requests served within the SLO, share of offered (min)", p["attainment"], gp["attainment_min"], "min"),
        ("Attainment, points below the oracle", 100 * (o["attainment"] - p["attainment"]), gp["below_oracle_pts_max"], "max"),
        ("Requests that missed the SLO (not shed), share", p["violated_share"], gp["violated_share_max"], "max"),
        ("SLO misses vs calibrated roofline, ratio", p["violated_share"] / max(rc["violated_share"], 1e-9), gp["violated_vs_calibrated_roofline_max"], "max"),
        ("Replicas that ran out of memory", p["crashes"], gp["crashes_max"], "max"),
        ("Service-hours below 90% within SLO, share", p["bad_service_hours"] / p["service_hours"], gp["bad_service_hours_share_max"], "max"),
        ("Attainment over the GPU-uniform scheduler, points (min)", 100 * (p["attainment"] - u["attainment"]), gp["over_uniform_pts_min"], "min"),
        ("Step-time error on unmeasured pairs, MAPE", acc["unmeasured_pairs"]["learned_mape"], gp["unmeasured_mape_max"], "max"),
        ("90% interval coverage on unmeasured pairs (min)", acc["unmeasured_pairs"]["interval_coverage"], gp["coverage_min"], "min"),
        ("Training jobs that ran out of memory", p.get("training_crashes", 0), gp["training_crashes_max"], "max"),
        ("Fleet busy with training on top of inference, points (min)", 100 * (p.get("fleet_busy", 0) - p["gpu_utilisation"]), gp["fleet_busy_over_inference_pts_min"], "min"),
    ]
    out = [{"gate": n, "value": v, "limit": lim, "kind": k, "pass": (v <= lim) if k == "max" else (v >= lim)} for n, v, lim, k in rows]
    return {"platform": out, "platform_pass": all(r["pass"] for r in out)}


def run_split(split: str, workers: int = 3) -> dict[str, Any]:
    seeds: list[int] = config.seeds()[split]
    variant = "heldout" if split == "heldout" else "dev"
    jobs = [(s, variant, i == 0) for i, s in enumerate(seeds)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            per_seed = list(ex.map(_seed_job, jobs))
    else:
        per_seed = [_seed_job(j) for j in jobs]
    dcfg = config.designs()
    designs = {n: _mean([s["designs"][n] for s in per_seed]) for n in dcfg["designs"]}
    res: dict[str, Any] = {
        "split": split,
        "variant": variant,
        "seeds": seeds,
        "designs": designs,
        "labels": {n: d["label"] for n, d in dcfg["designs"].items()},
        "main": dcfg["main"],
        "ablations": dcfg["ablations"],
        "accuracy": _mean([s["accuracy"] for s in per_seed]),
        "expansion": {"per_seed": [s["expansion"] for s in per_seed]},
        "site": per_seed[0]["site"],
        "per_seed": [{"seed": s["seed"], **{n: s["designs"][n]["attainment"] for n in dcfg["main"]}} for s in per_seed],
        "samples": per_seed[0].get("samples", {}),
    }
    ex = res["expansion"]["per_seed"]
    res["expansion"]["mean_abs_error"] = {k: float(np.mean([abs(e["estimators"][k]["error"]) for e in ex if e["estimators"][k]["error"] is not None])) for k in ex[0]["estimators"]}
    res["gates"] = check_gates(res, config.gates())
    return _round(res)


def write(split: str, res: dict[str, Any]) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{split}.json"
    path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return path


def run_heldout_once(workers: int = 3) -> dict[str, Any]:
    if LOCK.exists():
        raise HeldOutLocked(f"{LOCK} exists: the held-out sites have been run. Add new seeds under a new name and an ADR.")
    REPORTS.mkdir(exist_ok=True)
    LOCK.write_text(json.dumps({"seeds": config.seeds()["heldout"], "config_sha256": config_hash()}, indent=1) + "\n", encoding="utf-8")
    res = run_split("heldout", workers)
    write("heldout", res)
    return res
