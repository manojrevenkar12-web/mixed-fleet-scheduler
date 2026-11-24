# Results

Generated from `reports/*.json` by `python -m mfs.cli report`.

## Development sites: seeds 21, 22, 23

### All designs

| metric | uniform | roofline | roofline_calibrated | platform | oracle | with_margin | no_fallback | no_feedback | no_admission | naive_backfill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| requests | 1.668e+09 | 1.668e+09 | 1.668e+09 | 1.668e+09 | 1.668e+09 | 1.668e+09 | 1.668e+09 | 1.668e+09 | 1.668e+09 | 1.668e+09 |
| attainment | 0.4492 | 0.5453 | 0.9517 | 0.9638 | 0.966 | 0.9455 | 0.9632 | 0.9229 | 0.8521 | 0.9638 |
| violated_share | 0.5508 | 0.4547 | 0.02001 | 0.00451 | 0.00212 | 0.00638 | 0.00497 | 0.04484 | 0.1479 | 0.00451 |
| shed_share | 0 | 0 | 0.02825 | 0.03169 | 0.03194 | 0.04818 | 0.03183 | 0.03222 | 0 | 0.03169 |
| bad_service_hours | 1653 | 1377 | 243 | 183 | 176 | 224 | 189 | 305 | 285 | 183 |
| service_hours | 2016 | 2016 | 2016 | 2016 | 2016 | 2016 | 2016 | 2016 | 2016 | 2016 |
| gpu_utilisation | 0.5632 | 0.3567 | 0.418 | 0.4602 | 0.4577 | 0.5486 | 0.4628 | 0.4488 | 0.4595 | 0.4602 |
| met_per_gpu_hour | 1.375e+04 | 2.631e+04 | 3.939e+04 | 3.633e+04 | 3.651e+04 | 3e+04 | 3.609e+04 | 3.574e+04 | 3.212e+04 | 3.633e+04 |
| crashes | 6708 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tpot_misses | 1795 | 12 | 0 | 0 | 0 | 0 | 0 | 92 | 0 | 0 |
| overloads | 1217 | 6757 | 109 | 21 | 0 | 37 | 24 | 638 | 1727 | 21 |
| replica_starts_per_day | 66.67 | 62.19 | 103.1 | 96.52 | 83.62 | 126 | 100.4 | 85.19 | 96.67 | 96.52 |
| fallback_share | 0 | 0 | 0 | 0.00272 | 0 | 0.0018 | 0 | 0.00034 | 0.00176 | 0.00272 |
| drift_events | 0 | 0 | 25 | 10 | 0 | 13 | 8 | 0 | 10 | 10 |
| drift_window_attainment | 0.4503 | 0.4508 | 0.9261 | 0.9412 | 0.9415 | 0.9343 | 0.9376 | 0.8487 | 0.7378 | 0.9412 |
| drift_recovery_hours | – | 47.67 | 1 | 1 | 1 | 1 | 1 | 8 | 1 | 1 |
| train_points | 0 | 0 | 0 | 702 | 0 | 702 | 702 | 702 | 702 | 702 |
| training_jobs | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 |
| training_done_share | 0.9074 | 0.5741 | 0.4907 | 0.4722 | 0.4537 | 0.3889 | 0.4722 | 0.5 | 0.4907 | 0.5 |
| training_job_hours | 29.64 | 55.72 | 61 | 63.62 | 62.3 | 73.27 | 65.45 | 64.82 | 61.54 | 60.95 |
| training_tokens_b | 34 | 27.84 | 26.24 | 25.85 | 25.87 | 23.49 | 25.6 | 26.18 | 26.14 | 26.89 |
| training_gpu_share | 0.1935 | 0.2891 | 0.2703 | 0.2523 | 0.2654 | 0.2328 | 0.2483 | 0.2509 | 0.2487 | 0.267 |
| training_preemptions | 86 | 89 | 141 | 143 | 133 | 181 | 159 | 143 | 151 | 191 |
| training_crashes | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fleet_busy | 0.7568 | 0.6458 | 0.6883 | 0.7125 | 0.7231 | 0.7814 | 0.7111 | 0.6997 | 0.7081 | 0.7272 |

### Per site

| seed | uniform | roofline | roofline_calibrated | platform | oracle |
|---|---:|---:|---:|---:|---:|
| 21 | 45.2% | 43.7% | 94.3% | 96.2% | 96.2% |
| 22 | 50.0% | 65.9% | 95.5% | 96.5% | 97.1% |
| 23 | 39.6% | 54.0% | 95.7% | 96.4% | 96.5% |

### Performance model

| Decode-step prediction | Roofline error | Learned error | 90% interval holds |
|---|---:|---:|---:|
| Measured pairs, unmeasured batch and parallelism | 14.5% | 1.8% | 99% |
| Pairs never measured (model on GPU) | 14.8% | 8.3% | 98% |
| A GPU type never seen (MI300X) | 18.5% | 25.8% | 95% |
| ... after 3 calibration measurements | 18.5% | 14.9% | 100% |

### Expansion forecast

| Planned purchase: 32 x MI300X | site 1 | site 2 | site 3 | mean absolute error |
|---|---:|---:|---:|---:|
| True extra peak traffic served at 99% | +30% | +29% | +40% | |
| Roofline twin | +48% | +43% | +44% | 37% |
| Learned twin | +32% | +35% | +41% | 9% |
| Learned twin + 3 measurements | +30% | +30% | +40% | 1% |

### Gates

| Gate | Value | Limit | |
|---|---:|---:|---|
| Requests served within the SLO, share of offered (min) | 0.964 | ≥ 0.94 | pass |
| Attainment, points below the oracle | 0.216 | ≤ 1.5 | pass |
| Requests that missed the SLO (not shed), share | 0.005 | ≤ 0.01 | pass |
| SLO misses vs calibrated roofline, ratio | 0.225 | ≤ 0.5 | pass |
| Replicas that ran out of memory | 0.000 | ≤ 0 | pass |
| Service-hours below 90% within SLO, share | 0.091 | ≤ 0.12 | pass |
| Attainment over the GPU-uniform scheduler, points (min) | 51.456 | ≥ 30 | pass |
| Step-time error on unmeasured pairs, MAPE | 0.083 | ≤ 0.12 | pass |
| 90% interval coverage on unmeasured pairs (min) | 0.978 | ≥ 0.8 | pass |
| Training jobs that ran out of memory | 0.000 | ≤ 0 | pass |
| Fleet busy with training on top of inference, points (min) | 25.231 | ≥ 10 | pass |

