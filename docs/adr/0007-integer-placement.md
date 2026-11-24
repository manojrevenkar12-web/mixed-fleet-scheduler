# 0007. Replica placement by integer optimisation, compared with the greedy pass

## Context
The greedy pass places each service in turn, on the GPU type with the most requests per GPU within the SLO, and
keeps yesterday's replicas where it can. It is fast and easy to explain, but it decides one service at a time: a
service placed early can take the cards a later service fits best on, and the order of the services matters.

## Decision
`platform/optimize.py` solves one hour's placement for all services at once as a mixed-integer programme (SciPy's
`milp`, the HiGHS solver). Variables: replicas per (service, GPU type), replicas started this hour, and the share of
each service's need left unmet. Objective: unmet need (weight 1000) + GPUs used + replica starts (weight
`placement.ilp_start_weight`). Constraints: GPUs per type, capacity at the admission share covers the need unless
the shortfall is paid for, a cap on replicas per service. The candidate shapes, margins, fallback and memory checks
are the same ones the greedy pass uses: the solver only chooses counts. If HiGHS finds no solution within the time
limit, the greedy pass places the hour.

`placement.policy` in `config/platform.yml` selects the policy; `platform_ilp` in `designs.yml` runs it beside the
greedy platform on the same traces.

## Consequences
The study's policy table shows the trade-off on every run. The start weight is the lever: at 0.5 the solver
restarts replicas freely and loses attainment to cold starts; at 3.0 it starts fewer replicas than the greedy pass.
The solver takes tens of milliseconds an hour on this fleet, still well inside an hourly cycle.
