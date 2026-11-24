"""Writes REPORT.md, the README results block and the console data from reports/*.json. Never edited by hand:
CI regenerates them and fails on a diff."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import config

ROOT = config.ROOT
START, END = "<!-- results:start -->", "<!-- results:end -->"
NAMES = {"uniform": "Built for identical GPUs", "roofline": "Roofline", "roofline_calibrated": "Calibrated roofline", "platform": "The platform", "oracle": "Oracle"}


def _load(name: str) -> dict[str, Any] | None:
    p = config.REPORTS / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{d}f}%"


def num(x: float | None, d: int = 0) -> str:
    return "–" if x is None else f"{x:,.{d}f}"


ROWS: list[tuple[str, str, Any]] = [
    ("Requests served within the SLO", "attainment", lambda x: pct(x, 1)),
    ("Requests that missed the SLO", "violated_share", lambda x: pct(x, 2)),
    ("Requests shed by admission control", "shed_share", lambda x: pct(x, 1)),
    ("Service-hours below 90% within SLO", "bad_service_hours", num),
    ("GPU-hours used by inference, share of fleet", "gpu_utilisation", lambda x: pct(x, 0)),
    ("Requests within SLO per GPU-hour", "met_per_gpu_hour", num),
    ("Replicas out of memory (replica-hours)", "crashes", num),
    ("Replica-hours overloaded", "overloads", num),
    ("12 h after a driver update: served within SLO", "drift_window_attainment", lambda x: pct(x, 1)),
    ("GPU-hours busy, inference + fine-tuning, share of fleet", "fleet_busy", lambda x: pct(x, 0)),
    ("Fine-tuning tokens processed, billions", "training_tokens_b", lambda x: num(x, 1)),
    ("Fine-tuning jobs finished within the week", "training_done_share", lambda x: pct(x, 0)),
    ("Fine-tuning jobs preempted by inference", "training_preemptions", num),
    ("Fine-tuning jobs out of memory", "training_crashes", num),
]


def _table(r: dict[str, Any], names: list[str], heads: dict[str, str]) -> list[str]:
    d = r["designs"]
    out = ["| | " + " | ".join(heads.get(n, n) for n in names) + " |", "|---|" + "---:|" * len(names)]
    for label, key, fmt in ROWS:
        cells = [fmt(d[n][key]) for n in names]
        if "platform" in names:
            i = names.index("platform")
            cells[i] = f"**{cells[i]}**"
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return out


def _accuracy(r: dict[str, Any]) -> list[str]:
    a = r["accuracy"]
    rows = [
        ("Measured pairs, unmeasured batch and parallelism", a["measured_pairs"]),
        ("Pairs never measured (model on GPU)", a["unmeasured_pairs"]),
        (f"A GPU type never seen ({a['unseen_gpu']['gpu']})", a["unseen_gpu"]),
        (f"... after {int(a['unseen_gpu_calibrated']['extra_profiles'])} calibration measurements", a["unseen_gpu_calibrated"]),
    ]
    out = ["| Decode-step prediction | Roofline error | Learned error | 90% interval holds |", "|---|---:|---:|---:|"]
    for label, s in rows:
        out.append(f"| {label} | {pct(s['roofline_mape'])} | {pct(s['learned_mape'])} | {pct(s['interval_coverage'], 0)} |")
    return out


def _expansion(r: dict[str, Any]) -> list[str]:
    ex = r["expansion"]["per_seed"]
    e0 = ex[0]
    out = [
        f"| Planned purchase: {e0['count']} x {e0['gpu']} | " + " | ".join(f"site {i + 1}" for i in range(len(ex))) + " | mean absolute error |",
        "|---|" + "---:|" * (len(ex) + 1),
    ]
    out.append("| True extra peak traffic served at 99% | " + " | ".join(f"+{100 * e['true_gain']:.0f}%" for e in ex) + " | |")
    labels = {"roofline": "Roofline twin", "learned": "Learned twin", "learned_calibrated": "Learned twin + 3 measurements"}
    for k, lab in labels.items():
        out.append(f"| {lab} | " + " | ".join(f"+{100 * e['estimators'][k]['predicted_gain']:.0f}%" for e in ex) + f" | {pct(r['expansion']['mean_abs_error'][k], 0)} |")
    return out


def _gates(r: dict[str, Any]) -> list[str]:
    out = ["| Gate | Value | Limit | |", "|---|---:|---:|---|"]
    for x in r["gates"]["platform"]:
        v = x["value"]
        out.append(f"| {x['gate']} | {v:.3f} | {'≥' if x['kind'] == 'min' else '≤'} {x['limit']} | {'pass' if x['pass'] else '**fail**'} |")
    return out


def _ablations(r: dict[str, Any]) -> list[str]:
    d = r["designs"]
    out = [
        "| Variant | Within SLO | Missed SLO | Shed | Inference GPU-hours | After a driver update | Fleet busy | Fine-tuning tokens (bn) | Preemptions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in ["platform", *r["ablations"]]:
        m = d[n]
        name = "**The platform**" if n == "platform" else r["labels"][n]
        out.append(
            f"| {name} | {pct(m['attainment'])} | {pct(m['violated_share'], 2)} | {pct(m['shed_share'])} | {pct(m['gpu_utilisation'], 0)} | {pct(m['drift_window_attainment'])} | {pct(m['fleet_busy'], 0)} | {num(m['training_tokens_b'], 1)} | {num(m['training_preemptions'])} |"
        )
    return out


CAP = [
    ("Serving requests", "serving"),
    ("Inference headroom (allocated, idle)", "inference_headroom"),
    ("Fine-tuning work kept", "training_kept"),
    ("Fine-tuning work lost to preemption", "training_lost_to_preemption"),
    ("Fine-tuning out of memory", "training_out_of_memory"),
    ("Free while jobs waited: a job fitted, the policy waited", "stranded_job_fit_but_waited"),
    ("Free while jobs waited: fits in memory, too few GPUs free", "stranded_too_few_gpus_free"),
    ("Free while jobs waited: no waiting job fits in memory", "stranded_no_job_fits_memory"),
    ("Free, nothing queued", "idle_nothing_queued"),
]


def _policy_table(r: dict[str, Any], names: list[str]) -> list[str]:
    d = r["designs"]
    out = [
        "| Policy | Within SLO | Missed SLO | Inference GPU-hours | Within SLO per GPU-hour | Replica starts per day | Fine-tuning tokens kept (bn) | Jobs finished | Preemptions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in names:
        m = d[n]
        out.append(
            f"| {r['labels'][n]} | {pct(m['attainment'])} | {pct(m['violated_share'], 2)} | {pct(m['gpu_utilisation'], 0)} | {num(m['met_per_gpu_hour'])} | {num(m['replica_starts_per_day'])} | {num(m.get('training_net_tokens_b'), 1)} | {pct(m['training_done_share'], 0)} | {num(m['training_preemptions'])} |"
        )
    return out


def _capacity(r: dict[str, Any], names: list[str]) -> list[str]:
    d = r["designs"]
    if any(f"cap_{k}" not in d[n] for n in names for _, k in CAP):
        return []
    out = ["| Where the fleet's GPU-hours went | " + " | ".join(r["labels"][n] if n != "platform" else "The platform" for n in names) + " |", "|---|" + "---:|" * len(names)]
    for label, k in CAP:
        out.append(f"| {label} | " + " | ".join(pct(d[n][f"cap_{k}"]) for n in names) + " |")
    return out


def _twin(r: dict[str, Any]) -> list[str]:
    t = r.get("twin")
    if not t:
        return []
    days = ", ".join("not fired" if x is None else "same day" if x == 0 else f"after {x} d" for x in t["alarm_days_after_change"])
    return [
        "| Simulator vs what the Scheduler measures live | |",
        "|---|---:|",
        f"| Twin step-time error before the driver change | {pct(t['twin_error_before'])} |",
        f"| Twin step-time error after the change, on that GPU type | {pct(t['twin_error_after'])} |",
        f"| The Scheduler's calibrated model after the change | {pct(t['calibrated_error_after'])} |",
        f"| Drift alarm (daily error above {pct(t['threshold'], 0)}), days after the change | {days} |",
        f"| Alarms on days without a change, all sites | {t['false_alarm_days']} |",
    ]


def readme_block(dev: dict[str, Any], held: dict[str, Any] | None, host: dict[str, Any] | None, bench: dict[str, Any] | None = None) -> str:
    s = dev["site"]
    lines = [
        START,
        "",
        f"_Generated by `python -m mfs.cli report` from `reports/dev.json`{', `reports/heldout.json`' if held else ''}{' and `reports/host_profile.json`' if host else ''}. Do not edit by hand._",
        "",
        f"**Development sites:** {len(dev['seeds'])} simulated sites, each a week of hourly placements on {s['gpus']} GPUs of {s['gpu_types']} types "
        f"for {s['services']} LLM services, {s['profiles']} measured profiles per site, and {int(dev['designs']['platform']['training_jobs'] / len(dev['seeds']))} fine-tuning jobs queued for the GPUs inference leaves free. "
        "Every design meets the same traffic, the same jobs, the same hardware and the same measurement noise.",
        "",
    ]
    lines += _table(dev, dev["main"], NAMES)
    lines += ["", "**The performance model** (development sites; error is mean absolute percentage error against the true step time):", ""]
    lines += _accuracy(dev)
    lines += ["", "**The simulator's question:** how much more peak traffic does a planned purchase serve?", ""]
    lines += _expansion(dev)
    lines += ["", "**What each guardrail is worth** (one changed at a time):", ""]
    lines += _ablations(dev)
    if dev.get("placement_policies"):
        lines += [
            "",
            "**Scheduling policies, head to head on the same traces** (placement: greedy pass vs integer optimisation; fine-tuning start/wait: any free GPU vs the look-ahead rule vs the policy learned on separate training sites):",
            "",
        ]
        lines += _policy_table(dev, dev["placement_policies"] + [n for n in dev["start_policies"] if n not in dev["placement_policies"]])
    cap = _capacity(dev, ["platform", "naive_backfill", "learned_backfill", "elastic_jobs", "ilp_elastic"]) if "learned_backfill" in dev["designs"] else []
    if cap:
        lines += ["", "**Capacity-loss breakdown** (shares of all GPU-hours in the week; the same analysis runs on any Slurm-style job trace with `mfs analyse-trace`):", ""] + cap
    tw = _twin(dev)
    if tw:
        lines += [
            "",
            "**Simulator drift monitor:** each hour the Simulator's model predicts the step times the Scheduler then measures; a daily error above the threshold on one GPU type raises an alarm.",
            "",
        ] + tw
    lines += ["", f"**Release gates, development sites:** {'all pass' if dev['gates']['platform_pass'] else 'some fail'}", ""]
    lines += _gates(dev)
    if held:
        lines += ["", "**Held-out sites, run once** (a GPU type never measured, a new mixture-of-experts model measured on one GPU type only, two driver changes, burstier traffic):", ""]
        lines += _table(held, held["main"], NAMES)
        lines += [""]
        lines += _accuracy(held)
        lines += [""]
        lines += _expansion(held)
        lines += ["", f"Held-out gates: {sum(x['pass'] for x in held['gates']['platform'])} of {len(held['gates']['platform'])} pass.", ""]
        lines += _gates(held)
    hp = _load("heldout_policies")
    if hp:
        lines += ["", f"**Held-out policy sites, run once** (seeds {', '.join(str(x) for x in hp['seeds'])}, new sites never used to choose or train a policy):", ""]
        lines += _policy_table(hp, hp["placement_policies"] + [n for n in hp["start_policies"] if n not in hp["placement_policies"]])
        lines += [""] + _capacity(hp, ["platform", "naive_backfill", "learned_backfill", "elastic_jobs", "ilp_elastic"])
        tw = _twin(hp)
        if tw:
            lines += [""] + tw
    if host:
        f, mch = host["fitted"], host["machine"]
        lines += [
            "",
            f"**Measured on real hardware:** `mfs profile-host` timed decode-like steps with torch.profiler on the build machine ({mch['cpu']}, {mch['threads']} threads, PyTorch {mch['torch']}): "
            f"{f['memory_gb_s']} GB/s when memory-bound, {f['compute_gflop_s']} GFLOP/s when compute-bound, memory-bound up to batch {f['memory_bound_until_batch']}. "
            "On a GPU node the same harness would take the profiles the model trains on; here it checks the measuring code against a real chip.",
        ]
    if bench:
        lines += [
            "",
            f"**Decision time, measured** (`mfs bench`, one CPU thread): training the ensemble on {bench['train_points']} profiles takes {bench['train_s']} s; "
            f"one hourly placement for {bench['site']['gpus']} GPUs and {bench['site']['services']} services takes {bench['plan_ms_median']} ms (p95 {bench['plan_ms_p95']} ms).",
        ]
    lines += ["", END]
    return "\n".join(lines)


def report_md(dev: dict[str, Any], held: dict[str, Any] | None) -> str:
    out = ["# Results", "", "Generated from `reports/*.json` by `python -m mfs.cli report`.", ""]
    for title, r in (("Development sites", dev), ("Held-out sites (run once)", held)):
        if not r:
            continue
        names = r["main"] + r["ablations"]
        out += [f"## {title}: seeds {', '.join(str(s) for s in r['seeds'])}", "", "### All designs", ""]
        keys = list(r["designs"]["platform"])
        out.append("| metric | " + " | ".join(names) + " |")
        out.append("|---|" + "---:|" * len(names))
        for k in keys:
            cells = ["–" if r["designs"][n][k] is None else (f"{r['designs'][n][k]:.4g}" if isinstance(r["designs"][n][k], float) else str(r["designs"][n][k])) for n in names]
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        out += ["", "### Per site", "", "| seed | " + " | ".join(r["main"]) + " |", "|---|" + "---:|" * len(r["main"])]
        for s in r["per_seed"]:
            out.append(f"| {s['seed']} | " + " | ".join(pct(s[n]) for n in r["main"]) + " |")
        out += ["", "### Performance model", ""] + _accuracy(r) + ["", "### Expansion forecast", ""] + _expansion(r) + ["", "### Gates", ""] + _gates(r) + [""]
    return "\n".join(out) + "\n"


def write_all() -> list[Path]:
    dev = _load("dev")
    if dev is None:
        raise SystemExit("run `python -m mfs.cli run dev` first")
    held, host, bench = _load("heldout"), _load("host_profile"), _load("bench")
    hp = _load("heldout_policies")
    paths = []
    p = ROOT / "REPORT.md"
    p.write_text(report_md(dev, held), encoding="utf-8")
    paths.append(p)
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if START in text and END in text:
            head, rest = text.split(START, 1)
            _, tail = rest.split(END, 1)
            readme.write_text(head + readme_block(dev, held, host, bench) + tail, encoding="utf-8")
            paths.append(readme)
    data = ROOT / "web" / "public" / "data" / "report.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps({"dev": dev, "heldout": held, "heldout_policies": hp, "host": host, "bench": bench}, separators=(",", ":")) + "\n", encoding="utf-8")
    paths.append(data)
    return paths
