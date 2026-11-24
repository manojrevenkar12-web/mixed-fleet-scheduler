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
