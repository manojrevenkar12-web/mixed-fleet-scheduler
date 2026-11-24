# 0004. Fine-tuning jobs backfill what inference leaves free

## Context
Inference at a latency SLO leaves half of the fleet idle at night. Fine-tuning jobs can use it, but a job that is
preempted loses the work since its last checkpoint, and a job placed on a card it does not fit on crashes.

## Decision
Inference is planned first, on the whole fleet. Running jobs keep their GPUs while inference does not need them and
are preempted otherwise (oldest kept first). Queued jobs, in arrival order, start on the GPU type with the highest
predicted throughput among those that fit, but only on GPUs inference did not use yesterday over the next 3 hours,
unless the job is expected to finish within 3 hours. Throughput is the roofline calibrated by the site's training
profiles and corrected by what running jobs report. A job that ran out of memory is never placed on that
(model, GPU, count) again.

## Consequences
On development sites the look-ahead cuts preemptions by a quarter (143 vs 191 over three site-weeks) for 4% fewer tokens; the variant
that puts jobs on any free GPU stays in the ablation table so the trade-off is visible. Training throughput uses the
calibrated roofline, not the ensemble: its error matters less than a decode-step error, which decides SLOs.
