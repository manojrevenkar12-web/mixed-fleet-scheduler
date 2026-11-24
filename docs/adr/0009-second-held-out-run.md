# 0009. A second held-out run for the new policies

## Context
The held-out sites in `heldout` were run once (ADR 0006) before the integer placement (ADR 0007) and the learned
start policy (ADR 0008) existed. Rerunning them would turn them into development sites.

## Decision
New seeds `heldout_policies` in `seeds.yml`, in the held-out world, with their own lock
`reports/heldout_policies.lock`. `mfs run heldout_policies` writes the lock first and refuses a second run with
exit code 3; CI checks each lock was committed once. The policies, the start weight and the training seeds were
fixed on the development and training sites before this run. The numbers are reported as they came out.

## Consequences
The first held-out report stays as it was. Any later policy change needs another set of seeds and another ADR.

## Record
The first attempt (24.11.2026) wrote the lock and then stopped with an IndexError in the drift monitor before any
site finished: the held-out world's first driver change is on a GPU type with no running replicas, so it had no
measurements to compare. No result was produced or seen. The uncommitted lock was removed, the monitor fixed (no
measurements = no alarm), and the run started again with the same seeds and settings.

The second attempt completed. Results are in `reports/heldout_policies.json` and the README, unchanged by any later
tuning: integer placement lost 0.6 points of attainment, the learned start/wait policy finished fewer jobs, and the
drift monitor's single threshold false-alarmed on GPU types the model never measured well.
