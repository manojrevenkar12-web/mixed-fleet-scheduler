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

## Held-out sites (run once): seeds 3101, 3102, 3103

### All designs

| metric | uniform | roofline | roofline_calibrated | platform | oracle | with_margin | no_fallback | no_feedback | no_admission | naive_backfill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| requests | 1.913e+09 | 1.913e+09 | 1.913e+09 | 1.913e+09 | 1.913e+09 | 1.913e+09 | 1.913e+09 | 1.913e+09 | 1.913e+09 | 1.913e+09 |
| attainment | 0.5905 | 0.491 | 0.9456 | 0.9538 | 0.9542 | 0.9502 | 0.9516 | 0.8955 | 0.8314 | 0.9538 |
| violated_share | 0.4095 | 0.509 | 0.01286 | 0.00284 | 0.00206 | 0.0021 | 0.00374 | 0.06075 | 0.1686 | 0.00284 |
| shed_share | 0 | 0 | 0.04151 | 0.04333 | 0.04377 | 0.04768 | 0.04471 | 0.04375 | 0 | 0.04333 |
| bad_service_hours | 2045 | 1482 | 351 | 314 | 336 | 335 | 330 | 566 | 334 | 314 |
| service_hours | 2520 | 2520 | 2520 | 2520 | 2520 | 2520 | 2520 | 2520 | 2520 | 2520 |
| gpu_utilisation | 0.6457 | 0.2295 | 0.3026 | 0.3366 | 0.3053 | 0.4294 | 0.327 | 0.3179 | 0.3314 | 0.3366 |
| met_per_gpu_hour | 1.808e+04 | 4.229e+04 | 6.194e+04 | 5.643e+04 | 6.206e+04 | 4.381e+04 | 5.807e+04 | 5.593e+04 | 5.009e+04 | 5.643e+04 |
| crashes | 8380 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tpot_misses | 556 | 415 | 31 | 1 | 0 | 5 | 0 | 0 | 19 | 1 |
| overloads | 1466 | 6004 | 101 | 7 | 0 | 12 | 17 | 924 | 1998 | 7 |
| replica_starts_per_day | 129.4 | 71.71 | 116.9 | 128.4 | 109.1 | 155.7 | 126 | 104.7 | 128.4 | 128.4 |
| fallback_share | 0 | 0 | 0 | 0.2683 | 0 | 0.2201 | 0 | 0.2796 | 0.2731 | 0.2683 |
| drift_events | 0 | 0 | 12 | 4 | 0 | 14 | 5 | 0 | 6 | 4 |
| drift_window_attainment | 0.5475 | 0.3782 | 0.9374 | 0.9433 | 0.9456 | 0.9341 | 0.942 | 0.8736 | 0.8009 | 0.9433 |
| drift_recovery_hours | – | 46 | 1 | 1 | 1 | 3.333 | 1 | 9 | 1 | 1 |
| train_points | 0 | 0 | 0 | 642 | 0 | 642 | 642 | 642 | 642 | 642 |
| training_jobs | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 | 108 |
| training_done_share | 0.9722 | 0.9167 | 0.8426 | 0.8704 | 0.8518 | 0.713 | 0.8426 | 0.8518 | 0.8611 | 0.8982 |
| training_job_hours | 33.43 | 29.06 | 34.11 | 39.48 | 37.15 | 48.77 | 38.99 | 38.43 | 39.37 | 36.24 |
| training_tokens_b | 38.12 | 37.65 | 36.13 | 35.8 | 36.49 | 32.94 | 35.91 | 35.82 | 35.7 | 36.84 |
| training_gpu_share | 0.1727 | 0.2935 | 0.3236 | 0.3164 | 0.3354 | 0.3139 | 0.3225 | 0.3281 | 0.3242 | 0.3223 |
| training_preemptions | 166 | 42 | 63 | 84 | 56 | 134 | 69 | 61 | 77 | 117 |
| training_crashes | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| fleet_busy | 0.8184 | 0.523 | 0.6262 | 0.6529 | 0.6408 | 0.7433 | 0.6495 | 0.646 | 0.6556 | 0.6589 |

### Per site

| seed | uniform | roofline | roofline_calibrated | platform | oracle |
|---|---:|---:|---:|---:|---:|
| 3101 | 62.8% | 48.2% | 95.3% | 95.6% | 95.4% |
| 3102 | 60.6% | 49.9% | 94.1% | 95.3% | 95.2% |
| 3103 | 53.7% | 49.1% | 94.3% | 95.3% | 95.6% |

### Performance model

| Decode-step prediction | Roofline error | Learned error | 90% interval holds |
|---|---:|---:|---:|
| Measured pairs, unmeasured batch and parallelism | 13.8% | 2.2% | 96% |
| Pairs never measured (model on GPU) | 15.6% | 19.0% | 84% |
| A GPU type never seen (B200) | 14.4% | 19.2% | 98% |
| ... after 3 calibration measurements | 14.4% | 19.0% | 88% |

### Expansion forecast

| Planned purchase: 32 x B200 | site 1 | site 2 | site 3 | mean absolute error |
|---|---:|---:|---:|---:|
| True extra peak traffic served at 99% | +47% | +39% | +38% | |
| Roofline twin | +46% | +44% | +56% | 20% |
| Learned twin | +39% | +30% | +41% | 16% |
| Learned twin + 3 measurements | +42% | +29% | +30% | 19% |

### Gates

| Gate | Value | Limit | |
|---|---:|---:|---|
| Requests served within the SLO, share of offered (min) | 0.954 | ≥ 0.94 | pass |
| Attainment, points below the oracle | 0.034 | ≤ 1.5 | pass |
| Requests that missed the SLO (not shed), share | 0.003 | ≤ 0.01 | pass |
| SLO misses vs calibrated roofline, ratio | 0.221 | ≤ 0.5 | pass |
| Replicas that ran out of memory | 0.000 | ≤ 0 | pass |
| Service-hours below 90% within SLO, share | 0.125 | ≤ 0.12 | **fail** |
| Attainment over the GPU-uniform scheduler, points (min) | 36.336 | ≥ 30 | pass |
| Step-time error on unmeasured pairs, MAPE | 0.190 | ≤ 0.12 | **fail** |
| 90% interval coverage on unmeasured pairs (min) | 0.839 | ≥ 0.8 | pass |
| Training jobs that ran out of memory | 0.000 | ≤ 0 | pass |
| Fleet busy with training on top of inference, points (min) | 31.637 | ≥ 10 | pass |

