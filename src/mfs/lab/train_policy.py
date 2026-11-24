"""Learns the start/wait policy for fine-tuning jobs by policy search (the cross-entropy method).

Each candidate policy is a weight vector over `backfill_features`; it is scored by running whole simulated weeks on
training sites (their own seeds, never the development or held-out ones) and counting the training tokens kept at
the end of the week, so work lost to preemption counts against it. The inference planner during training is the
calibrated roofline, which is fast; the policy only decides when a job starts.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from .. import config
from ..platform.scheduler import N_FEATURES
from ..sim.designs import run_design
from ..world.world import build_site


def _score(args: tuple[list[float], list[int]]) -> float:
    w, seeds = args
    cfg = config.platform()
    d = config.designs()["designs"]["roofline_calibrated"]
    total = 0.0
    for s in seeds:
        site = build_site(s, "dev")
        total += run_design(site, "train", d, cfg, backfill_weights=w).train_net_tokens / 1e9
    return total / len(seeds)


def train(workers: int = 2) -> dict[str, Any]:
    pc = config.platform()["training"]["policy_search"]
    seeds = [int(s) for s in config.seeds()["policy_train"]]
    rng = np.random.default_rng(int(pc["seed"]))
    mean, std = np.zeros(N_FEATURES), np.full(N_FEATURES, float(pc["init_std"]))
    mean[0] = 0.5  # start out willing to start jobs
    history = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for it in range(int(pc["iterations"])):
            pop = [mean + std * rng.normal(size=N_FEATURES) for _ in range(int(pc["population"]))]
            scores = list(ex.map(_score, [(list(map(float, p)), seeds) for p in pop]))
            order = np.argsort(scores)[::-1][: int(pc["elite"])]
            elite = np.array([pop[i] for i in order])
            mean, std = elite.mean(axis=0), elite.std(axis=0) + float(pc["min_std"])
            history.append({"iteration": it, "best": float(max(scores)), "mean": float(np.mean(scores))})
            print(f"iteration {it}: best {max(scores):.3f} bn tokens kept, population mean {np.mean(scores):.3f}")
    baseline = {
        "look-ahead rule": _score_rule(seeds, True),
        "any free GPU": _score_rule(seeds, False),
    }
    out = {
        "weights": [round(float(x), 5) for x in mean],
        "features": ["bias", "hours_left/24", "slack_after_start", "inference_growth_ahead", "stable", "sin_hour", "cos_hour", "gpus/16"],
        "trained_on_seeds": seeds,
        "history": history,
        "train_score_bn": _score((list(map(float, mean)), seeds)),
        "baselines_bn": baseline,
    }
    path = config.ROOT / "models" / "backfill_policy.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def _score_rule(seeds: list[int], peak_aware: bool) -> float:
    cfg = config.platform()
    d = dict(config.designs()["designs"]["roofline_calibrated"])
    total = 0.0
    for s in seeds:
        site = build_site(s, "dev")
        r = run_design(site, "rule", d if peak_aware else {**d, "without": ["peak_aware"]}, cfg)
        total += r.train_net_tokens / 1e9
    return total / len(seeds)
