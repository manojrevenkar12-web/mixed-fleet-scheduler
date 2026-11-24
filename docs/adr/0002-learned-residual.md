# 0002. Learn the residual over the roofline, and calibrate the spread on pairs the model has not seen

## Context
A plain roofline is wrong by 15% on average and by more on some model-on-GPU pairs; a model trained from scratch
needs far more profiles than a site has, and is confidently wrong on a GPU type it has never seen.

## Decision
A PyTorch ensemble of 5 small MLPs predicts log(step time / roofline) from the roofline's own parts, the batch,
the tensor-parallel degree, the GPU's data sheet and the model's architecture. Members are trained on bootstrap
samples. Their spread is scaled so that 90% of measurements on held-out (model, GPU) pairs fall inside
mean ± 1.645 std (split-conformal style); held-out points chosen at random did not calibrate it, because the
planner leans on the spread exactly where a whole pair is missing.

## Consequences
On development sites the learned model's error on measured pairs is about 2%, on pairs never measured about 8%,
against 15% for the roofline. On a GPU type never measured it is worse than the roofline (26% vs 19%), which is
why the guardrails in ADR 0003 exist; three measurements on the new card bring it to 15%.
