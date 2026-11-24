# Repeatable tasks

## Add a GPU type
1. Add its data-sheet figures to `src/mfs/reference/hardware.py` (`GPUS`).
2. Give it a hidden efficiency in `config/world.yml: efficiency` and put it in a fleet.
3. `pytest -q`, then `python -m mfs.cli run dev`. Look at the "GPU type never seen" row: that is how the platform
   does on day one, before anyone profiles the card.

## Add a model
1. Add the architecture to `MODELS` in `reference/hardware.py` (dense or mixture of experts).
2. Add a service for it in `config/world.yml: services` with its SLOs and peak traffic.
3. Run dev and read the accuracy table: a model measured on few GPU types should show up as "fallback" placements.

## Change a guardrail
1. Change the value in `config/platform.yml: guardrails` (or `training`).
2. Add a variant in `config/designs.yml` with `set:` (a value) or `without:` (a guardrail off) and list it under
   `ablations`, so the README shows what it is worth.
3. `python -m mfs.cli run dev && python -m mfs.cli gate dev && python -m mfs.cli report`.

## Answer "what does buying N of GPU X give us?"
1. Set `config/world.yml: expansion` to the GPU and count.
2. `python -m mfs.cli run dev`; the expansion table compares the twin's forecast with the site's truth, before and
   after 3 calibration measurements on the new card.

## Profile a real machine
`python -m mfs.cli profile-host` times decode-like matmuls with torch.profiler and writes
`reports/host_profile.json`. On a GPU node, run the same harness to produce the profiles the ensemble trains on.

## Run the service
`python -m mfs.cli serve --port 8080 --state scheduler-state.json`, then `PUT /v1/site`, `POST /v1/profiles`,
`POST /v1/jobs`, and every hour `POST /v1/place` and `POST /v1/observe`. `examples/drive_service.py` replays a
simulated site through the HTTP API.

## Regenerate the console data and screenshots
`python -m mfs.cli report`; `cd web && npm run build && cd .. && python scripts/screenshots.py`.

## Run the held-out sites
Only once, after the gates are set: `python -m mfs.cli run heldout && python -m mfs.cli report`.
