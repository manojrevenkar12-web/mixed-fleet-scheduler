# 0001. Simulated sites with a hidden truth

## Context
A scheduler for mixed GPU fleets is judged by what it does to latency SLOs, GPU-hours and training throughput. We
have no GPU fleet to run it on, and a benchmark on one fleet says little about the next one.

## Decision
Each site is simulated from public data-sheet figures plus hidden factors: per-GPU kernel efficiency, a
model-on-GPU factor, a site factor, launch overhead, a batch penalty, imperfect overlap of compute and memory,
tensor-parallel communication over the node's links, memory limits, and driver changes mid-week. Designs see only
measurements (profiles, the step times their replicas measure, job throughput, out-of-memory errors). The
evaluation grades every hour against the truth. Every random draw is keyed by (seed, purpose, id), so every design
meets the same traffic, jobs and noise.

## Consequences
Numbers are simulated and labelled so. The roofline the truth is built on is the textbook model of LLM decode
(memory-bound at small batches, compute-bound at large ones); the hidden factors are what a real fleet adds to it.
`mfs profile-host` measures the same shape of curve on a real chip to keep the model honest.
