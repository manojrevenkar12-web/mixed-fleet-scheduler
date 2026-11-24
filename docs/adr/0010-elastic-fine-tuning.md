# 0010. Elastic fine-tuning jobs, chosen from the capacity-loss breakdown

## Context
The first capacity-loss breakdown showed about a fifth of the fleet's GPU-hours free while fine-tuning jobs waited.
Splitting that share by cause showed the larger part was not the start/wait policy: the free cards were V100 and
L40S, and no waiting job fitted on them in memory at the GPU count it asked for (a 34B model on 16 L40S needs more
than 48 GB a card when fully sharded over 16). A better start/wait policy cannot use those GPUs.

## Decision
A job may run on a different number of GPUs of one type than it asked for: `training.elastic_sizes` lists the
multiples tried in order (`[1, 2, 0.5]`: as asked, sharded over twice as many cards so it fits, or on half when the
cards are short). The memory check and the measured throughput decide as before; the communication cost of the wider
job is in the world's truth, not assumed by the scheduler. `elastic_jobs` and `ilp_elastic` (with integer placement)
run beside the other policies. The platform's default stays `[1]` so the first held-out report keeps its meaning;
the policies are confirmed on the second held-out run (ADR 0009).

## Consequences
The breakdown splits "free while jobs waited" into three causes, so a change can be aimed at the right one. A wider
job is slower per GPU; the policy table reports tokens, not GPU-hours, so that cost is visible.
