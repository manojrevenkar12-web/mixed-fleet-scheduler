# Working on this repository

## Fast feedback

- `pytest -q` runs in about 10 s on one CPU: a tiny simulated site, a shorter training run of the ensemble.
- `ruff check . && ruff format --check . && mypy src tests` for lint and types.
- `python -m mfs.cli run dev && python -m mfs.cli gate dev && python -m mfs.cli report` runs every design on the
  development sites (a few minutes on 2 to 3 cores) and regenerates REPORT.md, the README block and the console data.
- `python -m mfs.cli bench` and `python -m mfs.cli profile-host` measure on this machine (labelled measured).
- `cd web && npm test && npm run typecheck && npm run lint` for the console.

## Invariants

Each one has a test or a CI step.

- **The truth stays in the world.**
  - What designs get: the site's measured profiles, the measured step times of their running replicas, the
    throughput their training jobs report, out-of-memory errors, and demand as it arrives.
  - What only the evaluation (`sim/designs.py`, `sim/evaluate.py`) may read: the hidden efficiencies, the
    model-on-GPU factors, the driver changes and the true step times (`world/world.py`).
  - Test: `tests/test_boundaries.py`.
- **The world does not read the platform's settings.** `world/` loads `world.yml` only; `platform/` and
  `service.py` never import `world`, `sim` or `lab`. Test: `tests/test_boundaries.py`.
- **Same luck for every design.** Every coin is keyed by (seed, purpose, id) in `rng.py`: the same traffic, jobs,
  profiles and measurement noise reach every design. Tests: `tests/test_world.py`, `tests/test_designs.py`.
- **The model reads and code decides.** The ensemble only predicts a step time and its spread. The planner, the
  margins, the fallback, admission control, preemption and the memory check are code with their thresholds in
  `config/platform.yml`. Tests: `tests/test_scheduler.py`, `tests/test_perfmodel.py`, `tests/test_training.py`.
- **Inference always wins.** Training gets only GPUs inference does not use this hour; a GPU inference needs again is
  taken back. Test: `tests/test_training.py`.
- **An observation counts once.** The service applies an hour's measurements once and keeps a profile or a job once,
  whatever the retries. Test: `tests/test_service.py`.
- **The lab and the service run the same scheduler** (`platform/build.py`).
- **Settings live in `config/`**, by owner: the world in `world.yml`; the platform in `platform.yml`; designs in
  `designs.yml`; seeds in `seeds.yml`; thresholds in `gates.yml`.
- **Reports are generated, never edited.** CI regenerates REPORT.md, the README block and
  `web/public/data/report.json` and fails on a diff.

## Held-out sites

- `config/seeds.yml` lists them; they run in the held-out world (`world.yml: heldout`): a GPU type that was never
  measured (B200), a mixture-of-experts model measured on one GPU type only, two driver changes (one faster),
  burstier traffic, a different fleet.
- `python -m mfs.cli run heldout` writes `reports/heldout.lock` before it runs. If the lock exists it exits with
  code 3. CI checks the lock was committed exactly once.
- Gates were set from the development sites before the held-out run. Never loosen a gate after it. The held-out
  numbers are reported as they came out (ADR 0006).
- If a change needs new held-out evidence, add new seeds under a new name and record why in an ADR.

## Data

Everything about a site is simulated. GPU and model figures are public data-sheet values (`reference/hardware.py`).
