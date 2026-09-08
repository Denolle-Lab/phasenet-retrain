### External results (QuakeScope notebooks) with bootstrap intervals on recall

| sequence | region | phase | analyst | recall_0p3_v7 | recall_0p3_parent | diff_0p3_v7_minus_parent | budget_mid | diff_at_matched_budget | MAE_v7_s | MAE_parent_s |
|---|---|---|---|---|---|---|---|---|---|---|
| Kaikoura 2016 | non-US | P | 357 | 0.661 [0.613, 0.711] | 0.703 [0.655, 0.751] | -0.042 [-0.112, 0.025] | 1,367 | -0.016 | 0.139 | 0.139 |
| Kaikoura 2016 | non-US | S | 458 | 0.502 [0.454, 0.546] | 0.520 [0.474, 0.561] | -0.017 [-0.083, 0.044] | 1,034 | -0.006 | 0.162 | 0.163 |
| Norcia 2016 | non-US | P | 701 | 0.763 [0.730, 0.793] | 0.795 [0.763, 0.823] | -0.031 [-0.076, 0.013] | 982.000 | -0.014 | 0.049 | 0.048 |
| Norcia 2016 | non-US | S | 656 | 0.744 [0.710, 0.776] | 0.777 [0.744, 0.806] | -0.034 [-0.079, 0.012] | 975.000 | -0.018 | 0.064 | 0.069 |
| Thessaly 2021 | non-US | P | 395 | 0.719 [0.671, 0.762] | 0.747 [0.701, 0.790] | -0.028 [-0.094, 0.033] | 1,439 | -0.022 | 0.129 | 0.135 |
| Thessaly 2021 | non-US | S | 331 | 0.390 [0.338, 0.441] | 0.402 [0.353, 0.450] | -0.012 [-0.085, 0.060] | 1,181 | -0.003 | 0.129 | 0.132 |
| Ridgecrest | western US | P | 347 | 0.605 [0.548, 0.651] | 0.582 [0.527, 0.631] | 0.023 [-0.049, 0.095] | n/a | n/a | 0.040 | 0.033 |
| Ridgecrest | western US | S | 298 | 0.517 [0.460, 0.570] | 0.487 [0.430, 0.544] | 0.030 [-0.047, 0.107] | 370.000 | 0.024 | 0.053 | 0.064 |
| San Simeon | western US | P | 62 | 0.823 [0.726, 0.919] | 0.855 [0.758, 0.935] | -0.032 [-0.161, 0.097] | n/a | n/a | 0.067 | 0.066 |
| San Simeon | western US | S | 16 | 0.750 [0.562, 0.938] | 0.750 [0.562, 0.938] | 0.000 [-0.250, 0.252] | n/a | n/a | 0.020 | 0.035 |
| Monte Cristo | western US | P | 16 | 0.750 [0.562, 0.938] | 0.812 [0.625, 1.000] | -0.062 [-0.312, 0.250] | n/a | n/a | 0.047 | 0.051 |
| Monte Cristo | western US | S | 12 | 0.417 [0.167, 0.667] | 0.500 [0.250, 0.750] | -0.083 [-0.500, 0.333] | n/a | n/a | 0.219 | 0.236 |
| Mendocino 2024 | western US | P | 200 | 0.830 [0.780, 0.880] | 0.900 [0.855, 0.940] | -0.070 [-0.140, -0.005] | n/a | n/a | 0.071 | 0.074 |
| Mendocino 2024 | western US | S | 113 | 0.619 [0.522, 0.699] | 0.681 [0.593, 0.770] | -0.062 [-0.195, 0.053] | 380.000 | -0.008 | 0.086 | 0.101 |

Unpaired difference intervals (independent resampling of the two 0/1 vectors; the true paired interval would be narrower). At matched budget the fine-tune is below the parent in 6 of 6 non-US sequence x phase pairs; a sign test gives p = 0.016 one-sided. Matched-budget differences have no interval: the per-pick data is not stored by the notebooks.
