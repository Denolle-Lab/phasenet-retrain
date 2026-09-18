# Label audit of the benchmark test set (41B, laptop part)

*Branch `issue/41b-benchmark-labels`, from PR #83 (`issue/41b-label-audit`,
commit 4d03c11). Produced by `scripts/audit_benchmark_labels.py` from the
committed `notebooks/benchmark_manifest.csv` (35,392 rows, sha256
`df8d6fbf…`) and `notebooks/step3_results.parquet` (1,952,410 rows, sha256
`f769cecc…`); no waveform and no model were touched. Every fraction carries a
95 % percentile-bootstrap interval from `scripts/metrics.py::bootstrap_ci`
(n = 1000, seed 0). `tables.md` holds the full tables, `provenance.json`
the constants, hashes and report checksums.*

```bash
python scripts/audit_benchmark_labels.py --benchmark notebooks/benchmark_manifest.csv \
    --results notebooks/step3_results.parquet --out-dir docs/audit_2026-09-16_benchmark_labels
```

## Verdict

The label problems found here do not change the ranking of the 2026
fine-tunes or the sizes of the documented gaps (H1 to H4 of
`docs/audit_2026-09-07/README.md`): paired on the same traces, v7 minus the
parent is −0.030 [−0.032, −0.027] P recall on all 31,880 P labels and −0.025
[−0.028, −0.023] on the 28,842 labels no screen flags; S is −0.044 [−0.047,
−0.040] on all 16,967 S labels and −0.048 [−0.052, −0.043] (S1/S2/C1 removed)
or −0.041 [−0.045, −0.037] (S1/C1 removed); the conditional P timing gain of
v7 is −0.019 [−0.023, −0.015] s on all picks both models make and −0.018
[−0.022, −0.014] s on the clean ones. What the
problems do change is the level of the S numbers: on the labels no screen
flags, the conditional S MAE of the parent is 0.312 s instead of 0.631 s
and of v7 0.248 s instead of 0.586 s, because 11.3 % of the S labels (1,924)
carry a consensus of at least six independent public pickers on an S at
another time (median +1.28 s, 78 % later than the label) and 15.0 % (2,543)
sit where no more than one independent picker finds an S at a P SNR above
10 dB. For P, S1 flags 3.5 % (1,124) of the labels; with C1 the clean subset
drops 9.5 % of the P labels and the parent's conditional P MAE goes from
0.237 to 0.199 s.

Two artefacts of the benchmark construction, not of the labels, were
found on the way and belong to 34C: (1) notebook 05 pads the 30 s window
with zeros when the P sits less than 27 s into the source trace (every
STEAD and TXED trace, 97 % of INSTANCE), and on 514 P labels (1.6 %; 12.4 %
of TXED, 2.5 % of STEAD) the pickers' consensus is the step from the padding
into the data, so the stored `p_prob` is the probability at that step; on
those TXED rows the parent's P recall is 0.975 and its own argmax is at the
step in 97 % of them (TXED P recall 0.827 with, 0.806 without those rows).
(2) AQ2009GM was cut at an assumed 100 Hz although its trace layout (8751
samples, P at sample 3125 in 71 % of rows) is 70 s and 25 s at 125 Hz, and
its S−P slope is the only anomalous one (details below); ETHZ was cut at
100 Hz although its picks are at 120 to 500 Hz (already in the roadmap).

## Per dataset

`S1` = consensus offset (≥ 6 independent public PhaseNet weights detect at
prob ≥ 0.3, MAD of their residuals ≤ 0.15 s, consensus > 0.3 s from the
label, consensus not within 1 s of the source trace or window edge). `edge`
= the excluded edge cases. `S2` = consensus miss (P SNR > 10 dB, ≤ 1
independent detection). `C1` = S−P against hypocentral distance, Theil–Sen,
|residual| > max(4 MAD, 1 s). Percentages of the labels of that phase;
obst2024 has no results (notebook 05 skipped it).

| dataset | traces (manifest / results) | S1 P flag % | S1 P edge % | S2 P % | S labels | S1 S flag % | of which late % | S2 S % | C1 flag % | S−P slope s/km (epi.) | Vp/Vs (epi. / hyp.) | Aguilar rows still in report |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| aq2009gm | 1,324 / 1,300 | 6.3 [4.9, 7.6] | 0.0 | 0.0 | 419 | 33.2 [28.6, 37.7] | 51 | 1.0 [0.2, 1.9] | 18.5 [15.1, 22.5] | 0.152 | 1.91 / 2.00 | 1,075 by name (ambiguous, see below) |
| ceed | 5,435 / 4,530 | 4.7 [4.1, 5.3] | 0.0 | 0.0 | 2,280 | 4.7 [3.9, 5.6] | 56 | 2.2 [1.7, 2.9] | 9.3 [8.5, 10.1] | 0.107 | 1.64 / 1.66 | 95 (report published after the notebook ran) |
| cwa | 200 / 195 | 7.2 [3.6, 10.8] | 0.0 | 0.0 | 122 | 16.4 [9.8, 23.0] | 55 | 1.6 [0.0, 4.1] | no distance | n/a | n/a | no report |
| ethz | 1,458 / 1,365 | 1.9 [1.2, 2.6] | 0.0 | 0.0 | 888 | 7.2 [5.5, 9.0] | 53 | 0.1 [0.0, 0.3] | 2.4 [1.6, 3.2] | 0.113 | 1.68 / 1.70 | 0 |
| instancecounts | 9,776 / 8,406 | 2.8 [2.4, 3.1] | 0.0 | 0.0 | 3,037 | 11.5 [10.4, 12.7] | 87 | 3.4 [2.8, 4.1] | 1.4 [1.2, 1.6] | 0.096 | 1.58 / 1.53 | 0 |
| mlaapde | 1,302 / 1,298 | 3.5 [2.5, 4.5] | 0.2 | 0.0 | 0 | n/a | n/a | n/a | no S | n/a | n/a | no report |
| pisdl | 765 / 762 | 2.1 [1.2, 3.1] | 0.0 | 0.0 | 451 | 2.0 [0.9, 3.3] | 22 | 0.0 | 18.6 [15.8, 21.3] | 0.114 | 1.69 / 1.69 | no report |
| pnw | 2,841 / 2,333 | 1.8 [1.3, 2.4] | 0.0 | 0.0 | 1,430 | 5.3 [4.1, 6.5] | 0 | 0.7 [0.3, 1.1] | 17.0 [15.7, 18.5] | 0.109 | 1.65 / 1.67 | 0 |
| stead | 9,360 / 9,360 | 4.5 [4.1, 4.9] | 2.5 [2.1, 2.8] | 0.0 | 6,457 | 14.1 [13.2, 15.0] | 89 | 29.1 [27.9, 30.1] | 6.6 [6.1, 7.1] | 0.096 | 1.58 / 1.59 | 0 |
| txed | 2,261 / 2,261 | 1.5 [1.0, 2.0] | 12.4 [11.1, 13.7] | 0.0 | 1,772 | 13.4 [11.9, 15.2] | 88 | 27.9 [25.6, 30.0] | 8.6 [7.5, 9.8] | 0.117 | 1.70 / 1.70 | 0 |
| vcseis | 200 / 113 | 3.5 [0.9, 7.1] | 0.0 | 0.0 | 111 | 11.7 [6.3, 18.0] | 15 | 0.0 | no distance | n/a | n/a | no report |
| all | 35,392 / 31,923 | 3.5 (1,124 of 31,880) | 1.6 (514) | 0.0 | 16,967 | 11.3 (1,924) | 78 | 15.0 (2,543) | 6.3 (2,017 of 31,923 result traces) | | | |

Before the edge exclusion the raw S1 P rates were 7.0 % for STEAD and
13.8 % for TXED; the excluded cases sit at 0.0 to 0.25 s of the source
trace in nominal seconds (the first real sample after the zero padding), not
at the label. No S2 P flag fires anywhere: at a P SNR above 10 dB at least
three independent weights always detect the P (15,768 rows).

## Sensitivity of the benchmark conclusions

Weights `jma_wc` (parent), `instance` (strongest public in-domain weight)
and `jma_wc_ft_global_v7`; recall at 0.3 and conditional MAE on all labels
versus the labels with no S1, S2 or C1 flag (`s1_s2_c1`) or no S1 or C1 flag
(`s1_c1`), the difference with a paired percentile bootstrap (the same
resampled traces evaluate both). Per dataset in `sensitivity.csv`.

| weight | phase | metric | all | clean (s1_s2_c1) | all − clean | clean (s1_c1) | all − clean |
|---|---|---|--:|--:|--:|--:|--:|
| jma_wc | P | recall | 0.884 (n 31,880) | 0.887 (n 28,842) | −0.003 [−0.005, −0.002] | 0.887 | −0.003 [−0.005, −0.002] |
| jma_wc | P | cond. MAE s | 0.237 | 0.199 | +0.038 [+0.034, +0.042] | 0.199 | +0.038 [+0.034, +0.042] |
| jma_wc | S | recall | 0.549 (n 16,967) | 0.606 (n 12,129) | −0.057 [−0.062, −0.052] | 0.506 (n 14,590) | +0.043 [+0.040, +0.046] |
| jma_wc | S | cond. MAE s | 0.631 | 0.312 | +0.319 [+0.300, +0.337] | 0.318 | +0.314 [+0.296, +0.331] |
| instance | P | recall | 0.879 | 0.882 | −0.003 [−0.004, −0.001] | 0.882 | −0.003 [−0.004, −0.001] |
| instance | P | cond. MAE s | 0.195 | 0.163 | +0.032 [+0.029, +0.035] | 0.163 | +0.032 [+0.029, +0.035] |
| instance | S | recall | 0.392 | 0.432 | −0.039 [−0.044, −0.035] | 0.359 | +0.033 [+0.030, +0.036] |
| instance | S | cond. MAE s | 0.584 | 0.215 | +0.369 [+0.344, +0.392] | 0.216 | +0.368 [+0.343, +0.391] |
| jma_wc_ft_global_v7 | P | recall | 0.854 | 0.862 | −0.008 [−0.009, −0.006] | 0.862 | −0.008 [−0.009, −0.006] |
| jma_wc_ft_global_v7 | P | cond. MAE s | 0.198 | 0.164 | +0.035 [+0.031, +0.038] | 0.164 | +0.035 [+0.031, +0.038] |
| jma_wc_ft_global_v7 | S | recall | 0.505 | 0.558 | −0.053 [−0.057, −0.048] | 0.465 | +0.040 [+0.037, +0.043] |
| jma_wc_ft_global_v7 | S | cond. MAE s | 0.586 | 0.248 | +0.337 [+0.317, +0.356] | 0.250 | +0.335 [+0.315, +0.355] |

Paired gaps on the same traces (`sensitivity_gap.csv`), v7 minus parent:

| phase, bin | all rows | no S1/S2/C1 flag | no S1/C1 flag |
|---|--:|--:|--:|
| P all recall | −0.030 [−0.032, −0.027] (n 31,880) | −0.025 [−0.028, −0.023] (n 28,842) | same |
| P local | −0.012 [−0.015, −0.009] | −0.010 [−0.014, −0.007] | same |
| P regional | −0.043 [−0.047, −0.038] | −0.035 [−0.040, −0.031] | same |
| P teleseismic | −0.101 [−0.119, −0.084] | −0.102 [−0.121, −0.081] | same |
| P cond. MAE on picks both make | −0.019 [−0.023, −0.015] s | −0.018 [−0.022, −0.014] s | same |
| S all recall | −0.044 [−0.047, −0.040] (n 16,967) | −0.048 [−0.052, −0.043] (n 12,129) | −0.041 [−0.045, −0.037] (n 14,590) |
| S local | −0.032 [−0.037, −0.028] | −0.033 [−0.038, −0.028] | −0.030 [−0.034, −0.025] |
| S regional | −0.073 [−0.081, −0.065] | −0.094 [−0.106, −0.082] | −0.070 [−0.079, −0.060] |
| S cond. MAE on picks both make | −0.001 [−0.004, +0.001] s | −0.007 [−0.010, −0.004] s | −0.007 [−0.010, −0.004] s |

v7 minus `instance`: P all −0.025 → −0.020 clean; S all +0.113 → +0.126
(s1_s2_c1) / +0.106 (s1_c1). Every sign and every ordering of
`docs/audit_2026-09-07/README.md` task 2 survives; the regional P deficit
shrinks from 0.043 to 0.035 and the regional S deficit moves from 0.073 to
0.094 or 0.070 depending on whether S2 is applied, so H2's "grows with
distance" holds under both rules. H1 and H4 rest on probability levels and
SNR bins that the screens do not touch; H3 is a contamination count and is
unaffected.

## Findings behind the numbers

**S1 on S: the late-S consensus in STEAD, TXED and INSTANCE.** Of the S1 S
flags, 89 % (STEAD), 88 % (TXED) and 87 % (INSTANCE) have the pickers'
S after the label; the consensus residual has quartiles +0.34, +1.28 and
+2.99 s and 30 flags saturate at the +5 s search boundary. In STEAD it is a
local-distance effect: 18.1 % of the 4,038 local S labels against 7.4 % of
the 2,419 regional ones, 20 % of the labels with S−P of 10 to 15 s against
5 % above 20 s. The parent detects an S on 93 % of the flagged rows (median
residual +1.20 s) and on 50 % of the others, so these rows are where the
models see a clear S that is not at the label; whether the label or the
sixteen pickers name the S phase is what the server run must settle on the
waveforms (`review_sheet.csv`, 30 rows per dataset ranked by |consensus
residual| / (MAD + 0.05 s) × detecting share). AQ2009GM is different: 33.2 %
of its 419 S labels flag with a symmetric residual (51 % early), and 33 of
its rows have no P and an S at exactly sample 5000 (40 s at 125 Hz), 24 of
which no independent weight detects: placeholder S labels.

**S2 on S rests on the P SNR.** `snr_db` is notebook 05's ratio of 2 s after
to 2 s before the P; an S at regional distance can be invisible at a high P
SNR. That is why the S2 S rate is 29.1 % in STEAD and 27.9 % in TXED
(regional, S−P 20 to 30 s) and why the `s1_c1` rule is reported alongside.
Under `s1_c1` the S recall of every weight is lower on the clean rows (the
removed S1 rows are ones the models detect) and the gap is unchanged.

**C1: hypocentral, not epicentral.** Against `distance_km` (epicentral), a
single Theil–Sen line flags 20.0 % of INSTANCE (its deep events at short
epicentral distance, depth to 300 km); against sqrt(D² + z²) it flags 1.4 %.
The hypocentral residual is the flag used for the clean subsets; both are in
`per_dataset_summary.csv`. PNW (17.0 %) and PiSDL (18.6 %) keep high rates
with MADs of 0.44 and 0.26 s: their distances are Haversine from the
notebook's lat/lon, and their S−P scatter at fixed distance is the server
question (C2 on the S onset). Slopes: 0.096 to 0.117 s/km (Vp/Vs 1.58 to
1.70 at Vp = 6 km/s) everywhere except AQ2009GM.

**AQ2009GM: the distance is epicentral and consistent with the catalogue;
the time axis is the anomaly.** Its epicentral slope is 0.152 s/km on the
405 rows with S−P and distance (Vp/Vs 1.91); against hypocentral distance
the intercept is −0.03 s and the slope 0.167 s/km (Vp/Vs 2.00), so the
distances share an origin with the times (a hypocentral column mislabelled
as epicentral would lower the slope, not raise it, and a unit other than km
would not give a zero intercept). Compared with the INSTANCE rows within
about 0.6° of L'Aquila at the same epicentral distance and depth, AQ2009GM's S−P
is 1.15 to 1.23 times longer (3.04 vs 2.62 s at 10 to 20 km, 8.63 vs 7.04 s
at 40 to 60 km). The manifest's `ts_tp_s` for AQ2009GM was computed at
100 Hz because `trace_sampling_rate_hz` is absent from its metadata
(notebook 04 cell 20 printed the fallback), and notebook 05 cut its windows
at 100 Hz. The trace layout says 125 Hz: 8751 samples = 70 s × 125 + 1, P at
sample 3125 = 25 s × 125 in 71 % of rows (95 % within ±1 sample), S-only
rows at sample 5000 = 40 s × 125. At 125 Hz the hypocentral slope becomes
0.133 s/km (Vp/Vs 1.80), inside the range of the other datasets, and the
INSTANCE ratio is explained. Consequences if confirmed on the server (the
`benchmark` mode of `audit_source_labels.py` reports `rate_mismatch` per
row): the AQ2009GM benchmark windows are 24 s of data stretched to 30 s,
its residuals are in 0.8 × real seconds, and its `ts_tp_s` and C1 line are
25 % too long. Its labels themselves are not shown wrong by this.

**AQ2009GM and MLAAPDE names are positional and repeat across chunks.**
The benchmark manifest has no chunk column; 44 AQ2009GM and 95 MLAAPDE
(dataset, trace_name) pairs repeat, and notebook 05's index took the last
shard that carries a name. The Aguilar cross-check for AQ2009GM is therefore
by name only: 1,075 of 1,324 names occur in the 26,739 flagged names of
`aq2009_report.csv`, which is what chance gives when a name lives in up to
ten shards (the S1 S rate is 36.6 % on matched names and 20.9 % on the
rest, P 6.1 % against 7.3 %). The server run reads through the loader,
which rejects a name it cannot place in one chunk; the count of such
rejections is the number of AQ2009GM benchmark rows whose waveform may not
belong to their label. Six STEAD names also occur in TXED (`bucket0$…,:3,:6000`
style); the HDF5 keeps one waveform per name, so those six TXED rows scored a
STEAD waveform.

**Aguilar cross-check.** The five reports the notebook applied (stead,
instance, pnw, txed, ethz) leave 0 benchmark rows in the reports. The CEED
report (253,309 names, published later) covers 95 benchmark rows (1.7 %);
their S1 rates are 5.3 % P / 9.7 % S against 4.7 % / 4.6 % on the other
rows, and none of them is an S2 S flag. Report files and sha256 in
`provenance.json`; the aq2009 report has 85,520 rows for 26,739 names
(`docs/LABEL_ERROR_FILTERING.md` quotes 26,739).

**C4.** P sits at a fixed sample in AQ2009GM (3125, 73 %), PiSDL (3000,
63 %), MLAAPDE (2400, 58 %), VCSEIS (6000, 55 %), ETHZ (11999, 20 %) and
STEAD (500, 16 %); within 1 s of the source trace edge in 0.2 % of TXED and
0.3 % of MLAAPDE, none elsewhere (CEED and CWA carry no length in the
name). The manifest's `p_in_s_window` (S within 15 s of the P, 36.7 % of
rows) marks the rows where the S-centred window contains the P: S1 P flags
are two to four times rarer inside it (STEAD 1.7 % vs 6.0 %, INSTANCE 1.4 %
vs 3.3 %), S1 S flags more common (STEAD 19.2 % vs 8.9 %, INSTANCE 13.3 %
vs 7.3 %). `multi_arrival` is a dataset constant (AQ2009GM and VCSEIS only)
and separates nothing.

**ETHZ.** Its picks imply 120, 200, 250 and 500 Hz metadata rates, notebook
05 cut its windows at 100 Hz, and the S−P interval in the window is 1.2 to
5 times the manifest's `ts_tp_s`. Its S1 rates (1.9 % P, 7.2 % S) are in
nominal window seconds. Already listed under 34C in
`docs/2026-09-10_revised_issue_roadmap.md`.

## Files

| file | content |
|---|---|
| `per_trace_flags.parquet` (not committed, regenerated by the script) | 31,923 result traces: label positions, `n_independent`, `n_detect`, `consensus_residual_s`, `consensus_mad_s`, `consensus_position_s`, `consensus_source_position_s`, `consensus_at_edge`, `consensus_saturated`, `s1_flag`, `s2_flag` (P) and the same with `_s`, `c1_residual_s`, `c1_flag`, `c1_epi_flag`, `aguilar_flagged`, the manifest columns, `clean_p`, `clean_s`, `clean_p_s1_c1`, `clean_s_s1_c1` |
| `per_dataset_summary.csv` | one row per dataset, every rate with its interval, C1 both ways, C4, Aguilar split |
| `sensitivity.csv`, `sensitivity_gap.csv` | the two tables above, per dataset and per distance bin |
| `review_sheet.csv` | 312 rows, ≤ 30 per dataset: the strongest S1 disagreements (P and S), with the source label sample, window position, consensus residual and position, `n_detect`, MAD, SNR, the other flags |
| `tables.md` | the script's full tables |
| `provenance.json` | commit, input hashes, constants, weight-to-corpus map, report paths and sha256, totals |

## What the server run adds (runbook step 6c)

`python scripts/audit_source_labels.py benchmark --benchmark notebooks/benchmark_manifest.csv --sample 3000`
reads a stratified sample of the benchmark rows through the loader at the
stored rate and runs C1 to C6 on the waveforms: the C2 onset test on the S
labels the pickers place later (STEAD, TXED, INSTANCE), `rate_mismatch` for
AQ2009GM and ETHZ, and the loader's rejections of chunk-ambiguous AQ2009GM
names. `manifest --manifest data/manifests_v2/{train,val,test}.csv` does
the same for the rows v7 was trained, validated and tested on.
