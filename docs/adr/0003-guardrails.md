# 0003. Guardrails: fallback, drift margin, admission control, feedback

## Decision
- Where the calibrated spread is above 0.20 (log), the planner uses the slower of the model and the site-calibrated
  roofline, plus 5%, and never less than the model's own upper bound.
- Measured step times flow back per (model, GPU, tensor-parallel degree, batch bucket); a GPU type whose replicas
  all moved by more than 12% in one direction is planned with a 1-std margin for 6 hours.
- Admission control admits at most 90% of a replica's planned capacity and sheds the rest explicitly, instead of
  letting a queue grow until every request on it misses the SLO.
- No margin elsewhere (`margin_z: 0`).

## Evidence (development sites)
Planning every batch on mean + 1 std cost 1.9 points of SLO attainment and 9 points more GPU-hours and avoided no
misses, so the margin is kept for drifting GPUs only and the variant stays in the ablation table. Without feedback
attainment drops by 4 points; without admission control by 11 points.
