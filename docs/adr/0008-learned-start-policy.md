# 0008. A start/wait policy for fine-tuning jobs learned by policy search

## Context
ADR 0004's look-ahead rule waits for GPUs inference will not need over the next 3 hours. It was written by hand. It
cuts preemptions but leaves GPUs free while jobs wait. A start/wait decision with delayed reward (the cost of a
preemption shows up hours later) is a reinforcement-learning problem, and the simulator is the environment.

## Decision
`backfill_features` describes one decision (start this job on this GPU type now?) by 8 numbers: hours of work
left, GPUs free after the start, inference growth expected in the next hours from yesterday's traffic, whether the
type is stable, the hour of day, and the job's size. The policy starts the job if a linear score is positive.

The weights are found by the cross-entropy method (`lab/train_policy.py`): each candidate runs whole simulated
weeks, and its reward is the fine-tuning tokens kept at the end of the week, so work lost to preemption counts
against it. It trains on its own seeds (`policy_train` in `seeds.yml`), never on development or held-out sites.
The weights are committed in `models/backfill_policy.json` with the training curve and both hand-written rules
scored on the same training sites.

Linear policy search, not a deep RL agent: eight weights can be read, the search needs no gradient through the
simulator, and a few hundred simulated weeks are enough. Inference planning is unchanged, so the policy cannot
trade SLOs for tokens.

## Consequences
The policy table shows tokens kept, jobs finished and preemptions for the three start policies on the same traces.
The reward counts only tokens; a policy that finishes fewer jobs but more tokens would win. That is stated in the
README's limits.
