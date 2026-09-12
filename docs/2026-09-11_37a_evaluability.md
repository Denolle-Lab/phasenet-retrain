# 37A: evaluability of the built held-out cases

Checkpoint 37A of issue #37, branch `issue/37a-suite-evaluability`, stacked on
`issue/44a-suite-policy` (PR #51). Every number below is read from
`data/heldout_testset/evaluability.csv` and the per-case
`data/heldout_testset/<key>/evaluability_stations.csv`, both written by

```sh
python scripts/certify_evaluability.py \
    --waveform-root /Users/marinedenolle/GitHub/phasenet-retrain/data/heldout_testset
```

with the default rule parameters. The waveform root is the original clone,
whose `waveforms/*.mseed` are not committed; the script reads them with
`obspy.read(headonly=True)` and writes nothing there. Each case run appends a
`reference_qa` record (role, policy hash, source hashes, rule parameters) to
`data/evaluation/access.jsonl` through `evaluation_policy.record_access`; the
`access_id` column of the CSV points at that record. No model, checkpoint or
prediction is read anywhere in this checkpoint.

## What 37A certifies, and what it does not

37A certifies, for the seven regression and development cases of
`configs/evaluation_suites.json`, whether the committed references and the
fetched waveforms can support each of two endpoints: `pick_scoring` (P and S
recall against reference arrivals on covered stations) and
`network_event_scoring` (event recovery from multi-station picks). The
verdict is metadata only: pick counts by provenance tier, station coverage
measured from the MiniSEED files, event support. The `certified_by` column
carries `37A` for those seven cases.

The twelve acceptance candidates get the same numbers and the same rule
applied, but `provisional = True` and no certification. Their verdicts are
inputs to 37B and 44B, not decisions.

37A does not certify: the 37B panel (immutable eligibility table, minimum
support per endpoint, predeclared fallbacks); the scientific quality of any
reference beyond its provenance tier (pick timing, onset, uncertainty,
completeness are not validated, only counted); the scorer's correctness
(#35); the loader's sampling-rate handling (#34; stations below 100 Hz are
counted, not excluded); the reviewed or preliminary status of the ISC
bulletins, which is taken from the existing docs and not re-verified. Hash
pinning in `manifest.json` identifies the files; it is not validation.

## Rules

| Parameter | Default | Meaning |
|---|---:|---|
| `min_refs_per_phase` (N) | 100 | reviewed-or-manual references per phase, in coverage, after duplicate collapse |
| `min_events_3sta` (M) | 10 | events with reviewed-or-manual P references in coverage on at least 3 covered stations |
| `min_covered_stations` (K) | 3 | covered stations carrying reviewed-or-manual references, network endpoint |
| `min_stations_pick` | 2 | covered stations carrying reviewed-or-manual references, pick endpoint |
| `min_coverage_fraction` | 0.5 | three-component common coverage over window length for a station to count as covered |
| `match_tol_s` | 0.5 | duplicate collapse per station and phase, `heldout_testset_registry.MATCH_TOL_S`, the scorer's rule |

Definitions, as implemented in `scripts/certify_evaluability.py`:

- Provenance tier of a pick: `mode` x `reference_ok`, so `manual_ok`,
  `unknown_ok` (ISC unknown-mode rows accepted by the build's
  `reference_ok`), `automatic_rejected`, `unknown_rejected` (NOA unknown-mode
  rows), `other` (none found). Reviewed-or-manual (RM) is `manual_ok` plus
  `unknown_ok` when the case's bulletin is recorded as reviewed in
  `BULLETIN_REVIEWED`: Petrinja, La Palma and Fagradalsfjall (ISC reviewed)
  yes; Kahramanmaraş, Noto 2024, Hualien and Reykjanes (ISC preliminary) no.
  Source: `docs/2026-09-08_heldout_test_cases.md`, "Data status", and the
  registry notes.
- Station-code resolution of a pick: `fetched` (station is a fetched
  `NET.STA`), `unfetched` (has a network code, not fetched), `ambiguous`
  (the code maps to more than one network in `station_map.csv`; quarantined,
  never counted), `unresolved` (bare code, no network). A further count
  flags unresolved codes equal to a fetched station's code; they stay
  unresolved.
- Coverage: per channel the trace spans are merged with a one-sample
  tolerance and clipped to the window; per location code the three channels
  with the longest coverage are intersected; the station's covered intervals
  are the union over locations. Gaps are the uncovered parts of the window
  in that three-component coverage, so a two-component file is one gap the
  length of the window. A reference counts "in coverage" when its time falls
  inside a covered interval of its station.
- `not_evaluable`: `no_waveforms` (no fetched station), `zero_overlap` (no
  `reference_ok` pick in coverage), `single_station` (fewer than two covered
  stations carry `reference_ok` picks). Any of these sets both endpoints to
  ineligible with that reason.
- `pick_scoring`: not `not_evaluable`, at least 2 covered stations with RM
  picks, and RM P >= N and RM S >= N in coverage after duplicate collapse.
  `pick_scoring_P_ok` and `pick_scoring_S_ok` are also written so a P-only
  use stays visible; the endpoint requires both.
- `network_event_scoring`: not `not_evaluable`, at least K covered stations
  with RM picks, and at least M events with RM P on at least 3 covered
  stations.

N = 100 was fixed before any count was seen: the 95 % binomial half-width at
recall 0.5 is 1.96 sqrt(0.25 / N), 0.098 at N = 100, which is the coarsest
resolution at which a per-case recall difference of 0.1 is distinguishable
from zero. M = 10 is the floor for a case to contribute events at all; the
panel-level requirement belongs to 37B. Nothing was adjusted after the
verdicts came out; the parameters are CLI flags and the CSV records them in
`rule_params`.

## Verdicts

"With picks" is the number of covered stations carrying RM picks in
coverage. RM P / S are after duplicate collapse.

| Case | Role | Windows (h) | Fetched / covered / with picks | RM P / S in coverage | Events with 3-station P | Pick scoring | Network/event scoring | Verdict | Certified |
|---|---|---:|---|---:|---:|---|---|---|---|
| kaikoura_2016 | regression | 1 x 3 | 6 / 6 / 6 | 357 / 458 | 67 | yes | yes | both | 37A |
| norcia_2016 | regression | 1 x 2 | 6 / 6 / 6 | 701 / 656 | 143 | yes | yes | both | 37A |
| thessaly_2021 | regression | 1 x 3 | 6 / 6 / 6 | 395 / 331 | 72 | yes | yes | both | 37A |
| samos_2020 | dev | 1 x 3 | 6 / 6 / 3 | 34 / 24 | 6 | no | no | neither | 37A |
| adriatic_2022 | dev | 1 x 3 | 6 / 6 / 6 | 400 / 357 | 71 | yes | yes | both | 37A |
| etna_2022_2024 | dev | 2 x 3 | 6 / 6 / 6 | 597 / 395 | 108 | yes | yes | both | 37A |
| corinth_thiva_2020 | dev | 2 x 3 | 6 / 6 / 6 | 108 / 87 | 20 | no | yes | network_event_scoring | 37A |
| kahramanmaras_2023 | acceptance | 1 x 3 | 7 / 7 / 5 | 101 / 37 | 20 | no | yes | network_event_scoring | provisional |
| noto_2024 | acceptance | 1 x 3 | 2 / 2 / 2 | 49 / 23 | 0 | no | no | neither | provisional |
| hualien_2024 | acceptance | 1 x 3 | 1 / 1 / 1 | 38 / 15 | 0 | no | no | neither (single_station) | provisional |
| petrinja_2020 | acceptance | 1 x 3 | 6 / 6 / 6 | 152 / 142 | 19 | yes | yes | both | provisional |
| reykjanes_2023 | acceptance | 2 x 3 | 6 / 6 / 1 | 25 / 18 | 0 | no | no | neither (single_station) | provisional |
| fagradalsfjall_2021 | acceptance | 2 x 3 | 6 / 6 / 2 | 16 / 8 | 0 | no | no | neither | provisional |
| la_palma_2021 | acceptance | 2 x 3 | 6 / 6 / 6 | 940 / 951 | 163 | yes | yes | both | provisional |
| santorini_2025 | acceptance | 2 x 3 | 6 / 6 / 6 | 1132 / 1067 | 199 | yes | yes | both | provisional |
| west_bohemia_2018 | acceptance | 2 x 3 | 6 / 6 / 1 | 137 / 157 | 0 | no | no | neither (single_station) | provisional |
| maurienne_2017 | acceptance | 2 x 3 | 6 / 6 / 6 | 290 / 308 | 61 | yes | yes | both | provisional |
| noto_swarm_2023 | acceptance | 1 x 3 | 2 / 2 / 0 | 0 / 0 | 0 | no | no | neither (zero_overlap) | provisional |
| campi_flegrei_2023 | acceptance | 2 x 3 | 6 / 6 / 6 | 30 / 26 | 5 | no | no | neither | provisional |

The rule text written per case (`pick_scoring_reason`,
`network_event_scoring_reason`):

| Case | Pick scoring rule | Network/event rule |
|---|---|---|
| kaikoura_2016 | P 357 and S 458 >= 100 on 6 stations | 67 events with 3-station P support on 6 stations |
| norcia_2016 | P 701 and S 656 >= 100 on 6 stations | 143 events with 3-station P support on 6 stations |
| thessaly_2021 | P 395 and S 331 >= 100 on 6 stations | 72 events with 3-station P support on 6 stations |
| samos_2020 | P 34 < 100; S 24 < 100 | events with 3-station P support 6 < 10 |
| adriatic_2022 | P 400 and S 357 >= 100 on 6 stations | 71 events with 3-station P support on 6 stations |
| etna_2022_2024 | P 597 and S 395 >= 100 on 6 stations | 108 events with 3-station P support on 6 stations |
| corinth_thiva_2020 | S 87 < 100 | 20 events with 3-station P support on 6 stations |
| kahramanmaras_2023 | S 37 < 100 | 20 events with 3-station P support on 5 stations |
| noto_2024 | P 49 < 100; S 23 < 100 | covered stations with reviewed-or-manual picks 2 < 3; events with 3-station P support 0 < 10 |
| hualien_2024 | single_station | single_station |
| petrinja_2020 | P 152 and S 142 >= 100 on 6 stations | 19 events with 3-station P support on 6 stations |
| reykjanes_2023 | single_station | single_station |
| fagradalsfjall_2021 | P 16 < 100; S 8 < 100 | covered stations with reviewed-or-manual picks 2 < 3; events with 3-station P support 0 < 10 |
| la_palma_2021 | P 940 and S 951 >= 100 on 6 stations | 163 events with 3-station P support on 6 stations |
| santorini_2025 | P 1132 and S 1067 >= 100 on 6 stations | 199 events with 3-station P support on 6 stations |
| west_bohemia_2018 | single_station | single_station |
| maurienne_2017 | P 290 and S 308 >= 100 on 6 stations | 61 events with 3-station P support on 6 stations |
| noto_swarm_2023 | zero_overlap | zero_overlap |
| campi_flegrei_2023 | P 30 < 100; S 26 < 100 | events with 3-station P support 5 < 10 |

What this means for the certified cases:

- Kaikōura, Norcia, Thessaly, Adriatic and Etna are eligible for both
  endpoints. Norcia's window is 2 h (the registry's 120 min), the others 3 h.
- Corinth-Thiva is eligible for network/event scoring only: 87 manual S
  references in coverage is under N. P alone passes (108), which
  `pick_scoring_P_ok` records; it is not the endpoint.
- Samos is eligible for neither: 34 P and 24 S manual references in
  coverage, 6 events with three-station support, 16 catalogue events in the
  window (NOA, M >= 2.0), and only 3 of the 6 fetched stations carry any
  reference (HT.CHOS 15/14, HL.PRK 12/4, KO.DKL 7/6; the three KO stations
  BLCB, BODT, AYDB at 58 to 104 km have none). It stays a development case
  for diagnostics; #35 must not report a Samos recall as an endpoint result
  under these rules.

For the acceptance candidates, the acceptance criterion of #37 is met in
the direction it requires: the Noto swarm is `not_evaluable` with
`zero_overlap`, Hualien with `single_station`; neither can be scored, let
alone passed. Reykjanes and West Bohemia are `single_station` too.
Petrinja's "both" rests entirely on the `unknown_ok` tier counted as
reviewed (0 manual rows); La Palma likewise. Kahramanmaraş's 20
three-station events use the 101 explicitly manual P rows (NEIC phase data
and ISC manual rows); on the `reference_ok` tier including preliminary
unknown-mode rows it would be 113 events, which is the size of the
preliminary tier's contribution and the reason it is kept separate.

## Waveform coverage

Every expected file was present and readable. Coverage is the
three-component common coverage summed over fetched stations and windows;
fraction is that sum over the expected station-window seconds.

| Case | Files expected / present | 3-component coverage (s) | Fraction | Gaps | Gap s | Stations < 100 Hz | Refs per fetched station-hour | Refs per covered hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kaikoura_2016 | 6 / 6 | 64800.0 | 1.0000 | 0 | 0.0 | 0 | 45.3 | 45.3 |
| norcia_2016 | 6 / 6 | 43104.0 | 0.9978 | 1 | 96.0 | 0 | 113.1 | 113.3 |
| thessaly_2021 | 6 / 6 | 64800.0 | 1.0000 | 0 | 0.0 | 0 | 40.3 | 40.3 |
| samos_2020 | 6 / 6 | 64800.0 | 1.0000 | 0 | 0.0 | 3 | 3.2 | 3.2 |
| adriatic_2022 | 6 / 6 | 64800.0 | 1.0000 | 0 | 0.0 | 0 | 42.1 | 42.1 |
| etna_2022_2024 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 0 | 27.6 | 27.6 |
| corinth_thiva_2020 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 0 | 5.4 | 5.4 |
| kahramanmaras_2023 | 7 / 7 | 75131.0 | 0.9938 | 1 | 469.0 | 1 | 44.0 | 44.1 |
| noto_2024 | 2 / 2 | 21045.5 | 0.9743 | 1 | 554.5 | 1 | 30.2 | 31.0 |
| hualien_2024 | 1 / 1 | 10800.0 | 1.0000 | 0 | 0.0 | 1 | 49.7 | 49.7 |
| petrinja_2020 | 6 / 6 | 64799.4 | 1.0000 | 3 | 0.6 | 0 | 16.3 | 16.3 |
| reykjanes_2023 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 1 | 2.5 | 2.5 |
| fagradalsfjall_2021 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 1 | 0.7 | 0.7 |
| la_palma_2021 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 0 | 52.5 | 52.5 |
| santorini_2025 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 0 | 61.1 | 61.1 |
| west_bohemia_2018 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 0 | 8.2 | 8.2 |
| maurienne_2017 | 12 / 12 | 129570.0 | 0.9998 | 1 | 30.0 | 1 | 16.6 | 16.6 |
| noto_swarm_2023 | 2 / 2 | 21432.7 | 0.9923 | 2 | 167.3 | 1 | 0.0 | 0.0 |
| campi_flegrei_2023 | 12 / 12 | 129600.0 | 1.0000 | 0 | 0.0 | 0 | 1.6 | 1.6 |

Gaps are few: IV.CAMP in Norcia (coverage 0.987, 96 s), KO.GAZ in
Kahramanmaraş (0.957, 469 s), IU.MAJO in Noto 2024 (0.949, 554 s) and two
gaps on IU.MAJO in the Noto swarm (0.985, 167 s), RD.ORIF in Maurienne
(0.997, 30 s), and three gaps of 0.09 to 0.38 s on SL.BOJS, SL.CRES and
SL.KOGS in Petrinja. No station falls under the 0.5 coverage rule, so
"covered" equals "fetched" for every case. References falling inside a gap are excluded from the
counts above: Kahramanmaraş 652 to 650 P, none elsewhere. The stations
below 100 Hz are KO.DKL, KO.BLCB, KO.BODT (50 Hz, Samos), IU.ANTO (40 Hz),
JP.JSD (20 Hz, Noto 2024 and swarm), JP.YOJ (20 Hz, Hualien), II.BORG
(40 Hz, Reykjanes and Fagradalsfjall), RD.ORIF (50 Hz). They are counted,
not excluded; the resampling contract is #34's.

Reference density is per fetched station-hour: from 0.0 (Noto swarm) and
0.7 (Fagradalsfjall) to 113 (Norcia). Campi Flegrei (1.6), Reykjanes (2.5),
Samos (3.2) and Corinth-Thiva (5.4) are sparse.

## Station-code resolution and station-selection bias

Counts are picks inside the windows, all tiers.

| Case | Picks in file | In windows | Fetched | Unfetched (resolved) | Ambiguous | Unresolved | Unresolved code = fetched code | Top unresolved codes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| kaikoura_2016 | 7213 | 7134 | 956 | 6178 | 0 | 0 | 0 |  |
| norcia_2016 | 8986 | 8873 | 1357 | 7516 | 0 | 0 | 0 |  |
| thessaly_2021 | 3650 | 3570 | 726 | 2844 | 0 | 0 | 0 |  |
| samos_2020 | 560 | 560 | 58 | 502 | 0 | 0 | 0 |  |
| adriatic_2022 | 4837 | 4622 | 757 | 3865 | 0 | 0 | 0 |  |
| etna_2022_2024 | 4537 | 4420 | 992 | 3428 | 0 | 0 | 0 |  |
| corinth_thiva_2020 | 1072 | 1072 | 195 | 877 | 0 | 0 | 0 |  |
| kahramanmaras_2023 | 20252 | 19941 | 1126 | 10077 | 0 | 8738 | 22 | BRTR:173, KHMN:116, MMAI:93, PDYAR:90, GERES:88 |
| noto_2024 | 16012 | 15948 | 314 | 8465 | 0 | 7169 | 0 | JJH:255, JTT:154, JHG:135, JJN:134, JSZ:131 |
| hualien_2024 | 32461 | 31870 | 253 | 9767 | 0 | 21850 | 0 | ETM:262, EHP:257, WHF:256, ETL:254, NNS:246 |
| petrinja_2020 | 4042 | 3993 | 344 | 340 | 0 | 3309 | 0 | OBKA:114, SOKA:112, MORH:91, KHC:83, FRGS:83 |
| reykjanes_2023 | 11338 | 11298 | 173 | 4598 | 0 | 6527 | 0 | ILFE:195, IVOG:194, IVOS:192, IKRI:190, IGRV:188 |
| fagradalsfjall_2021 | 4476 | 4396 | 37 | 1405 | 0 | 2954 | 0 | IKRI:60, IVOS:59, IVOG:58, ISAN:58, ISAU:57 |
| la_palma_2021 | 3151 | 3107 | 1891 | 1076 | 0 | 140 | 0 | EXILP:136, CTFS:2, CNOR:1, CRAJ:1 |
| santorini_2025 | 9053 | 8914 | 2254 | 6660 | 0 | 0 | 0 |  |
| west_bohemia_2018 | 9769 | 3078 | 297 | 0 | 0 | 2781 | 0 | KRC:314, LBC:312, POC:310, VAC:305, STC:304 |
| maurienne_2017 | 1153 | 1153 | 598 | 555 | 0 | 0 | 0 |  |
| noto_swarm_2023 | 13461 | 12976 | 0 | 0 | 0 | 12976 | 0 | SUZU:1146, N.SUZH:1137, E.YUOS:1099, E.IDES:978, N.YGDH:712 |
| campi_flegrei_2023 | 95 | 95 | 56 | 39 | 0 | 0 | 0 |  |

No committed `station_map.csv` lists an ambiguous code, so the quarantine
branch is exercised only by the tests. The 22 Kahramanmaraş picks whose
bare code equals a fetched station's code stay unresolved; resolving them
is an inventory task, not a counting decision.

Station-selection bias: the build ranked candidates by reference-pick
count, then distance, and kept the first six that returned data (README).
The fetched stations therefore carry the densest references, and the
"unfetched" column shows how much reference sits on registered stations
that were not fetched: 6,178 of 7,134 in-window picks for Kaikōura, 7,516
of 8,873 for Norcia, 3,865 of 4,622 for Adriatic. The busiest unfetched
stations per certified case (`evaluability_stations.csv`, P/S in window):
Kaikōura NZ.BHW 44/25, NZ.GVZ 43/47; Norcia IV.OFFI 112/5, IV.TERO 108/92;
Thessaly HL.EVR 63/44, HL.NEO 60/37; Adriatic IV.CING 66/35, IV.ARVD 62/30;
Etna IV.EPOZ 82/9, IV.EMPL 81/29; Corinth-Thiva HL.MET2 18/11, HL.MET3
18/15; Samos none. Any station-level extension of a case in 37B draws from
these tables; the bias is recorded, not corrected, here.

## Provenance tiers

Raw counts (before duplicate collapse) of in-window picks on the fetched
stations, P / S. "Bulletin reviewed" is the `BULLETIN_REVIEWED` entry used
for the `unknown_ok` tier; blank means the case has no `unknown_ok` rows.

| Case | Sources | manual_ok | unknown_ok | automatic_rejected | unknown_rejected | Bulletin reviewed | reference_ok dedup P / S | RM dedup in coverage P / S |
|---|---|---:|---:|---:|---:|---|---:|---:|
| kaikoura_2016 | GEONET | 357 / 458 | 0 / 0 | 141 / 0 | 0 / 0 |  | 357 / 458 | 357 / 458 |
| norcia_2016 | INGV | 701 / 656 | 0 / 0 | 0 / 0 | 0 / 0 |  | 701 / 656 | 701 / 656 |
| thessaly_2021 | NOA | 395 / 331 | 0 / 0 | 0 / 0 | 0 / 0 |  | 395 / 331 | 395 / 331 |
| samos_2020 | NOA | 34 / 24 | 0 / 0 | 0 / 0 | 0 / 0 |  | 34 / 24 | 34 / 24 |
| adriatic_2022 | INGV | 400 / 357 | 0 / 0 | 0 / 0 | 0 / 0 |  | 400 / 357 | 400 / 357 |
| etna_2022_2024 | INGV | 597 / 395 | 0 / 0 | 0 / 0 | 0 / 0 |  | 597 / 395 | 597 / 395 |
| corinth_thiva_2020 | NOA | 108 / 87 | 0 / 0 | 0 / 0 | 0 / 0 |  | 108 / 87 | 108 / 87 |
| kahramanmaras_2023 | ISC, USGS phase-data | 102 / 37 | 692 / 290 | 5 / 0 | 0 / 0 | False | 652 / 271 | 101 / 37 |
| noto_2024 | ISC, USGS phase-data | 49 / 23 | 145 / 93 | 4 / 0 | 0 / 0 | False | 100 / 81 | 49 / 23 |
| hualien_2024 | ISC, USGS phase-data | 38 / 15 | 127 / 72 | 1 / 0 | 0 / 0 | False | 86 / 63 | 38 / 15 |
| petrinja_2020 | ISC | 0 / 0 | 189 / 155 | 0 / 0 | 0 / 0 | True | 152 / 142 | 152 / 142 |
| reykjanes_2023 | ISC, USGS phase-data | 25 / 18 | 93 / 32 | 3 / 2 | 0 / 0 | False | 60 / 30 | 25 / 18 |
| fagradalsfjall_2021 | ISC | 2 / 0 | 22 / 8 | 5 / 0 | 0 / 0 | True | 16 / 8 | 16 / 8 |
| la_palma_2021 | ISC | 0 / 0 | 940 / 951 | 0 / 0 | 0 / 0 | True | 940 / 951 | 940 / 951 |
| santorini_2025 | NOA | 1132 / 1067 | 0 / 0 | 0 / 0 | 12 / 43 |  | 1132 / 1067 | 1132 / 1067 |
| west_bohemia_2018 | Zenodo pha | 138 / 159 | 0 / 0 | 0 / 0 | 0 / 0 |  | 137 / 157 | 137 / 157 |
| maurienne_2017 | franceseisme | 290 / 308 | 0 / 0 | 0 / 0 | 0 / 0 |  | 290 / 308 | 290 / 308 |
| noto_swarm_2023 | JMA deck | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |  | 0 / 0 | 0 / 0 |
| campi_flegrei_2023 | INGV | 30 / 26 | 0 / 0 | 0 / 0 | 0 / 0 |  | 30 / 26 | 30 / 26 |

The same tiers on all stations in the windows, to size what acquisition
could unlock:

| Case | manual_ok | unknown_ok | automatic_rejected | unknown_rejected |
|---|---:|---:|---:|---:|
| kaikoura_2016 | 2741 / 2567 | 0 / 0 | 1826 / 0 | 0 / 0 |
| norcia_2016 | 5553 / 3320 | 0 / 0 | 0 / 0 | 0 / 0 |
| thessaly_2021 | 2533 / 1037 | 0 / 0 | 0 / 0 | 0 / 0 |
| samos_2020 | 395 / 165 | 0 / 0 | 0 / 0 | 0 / 0 |
| adriatic_2022 | 3704 / 918 | 0 / 0 | 0 / 0 | 0 / 0 |
| etna_2022_2024 | 2978 / 1442 | 0 / 0 | 0 / 0 | 0 / 0 |
| corinth_thiva_2020 | 610 / 462 | 0 / 0 | 0 / 0 | 0 / 0 |
| kahramanmaras_2023 | 3343 / 48 | 13301 / 1632 | 1615 / 2 | 0 / 0 |
| noto_2024 | 2649 / 41 | 10075 / 893 | 2290 / 0 | 0 / 0 |
| hualien_2024 | 2911 / 178 | 19095 / 8101 | 1585 / 0 | 0 / 0 |
| petrinja_2020 | 4 / 0 | 2445 / 1521 | 23 / 0 | 0 / 0 |
| reykjanes_2023 | 2149 / 28 | 6276 / 2318 | 525 / 2 | 0 / 0 |
| fagradalsfjall_2021 | 217 / 0 | 2773 / 845 | 561 / 0 | 0 / 0 |
| la_palma_2021 | 0 / 0 | 1530 / 1577 | 0 / 0 | 0 / 0 |
| santorini_2025 | 6942 / 1878 | 0 / 0 | 0 / 0 | 22 / 72 |
| west_bohemia_2018 | 1485 / 1593 | 0 / 0 | 0 / 0 | 0 / 0 |
| maurienne_2017 | 660 / 493 | 0 / 0 | 0 / 0 | 0 / 0 |
| noto_swarm_2023 | 4224 / 5181 | 0 / 0 | 1555 / 2016 | 0 / 0 |
| campi_flegrei_2023 | 58 / 37 | 0 / 0 | 0 / 0 | 0 / 0 |

Three tiers matter for what follows. The seven certified cases are
manual-only (GEONET, INGV, NOA; Kaikōura also carries 141 automatic P on
the fetched stations, rejected). The ISC cases with a preliminary bulletin
have almost no manual S anywhere: 48 for Kahramanmaraş, 41 for Noto 2024,
178 for Hualien, 28 for Reykjanes, on all stations. Their S pick scoring
cannot become eligible by fetching more waveforms; it needs the operator's
manual readings, or the reviewed bulletin. The NEIC phase-data product
supplies manual P for M >= 4.5 and little else.

## Event support

| Case | Catalogue events in windows | Mag min / max | Without magnitude | Events with reference_ok picks on fetched stations | 3-station P, reference_ok on fetched | 3-station P, RM in coverage |
|---|---:|---|---:|---:|---:|---:|
| kaikoura_2016 | 137 | 3.40 / 6.18 | 0 | 135 | 67 | 67 |
| norcia_2016 | 150 | 2.40 / 4.30 | 0 | 150 | 143 | 143 |
| thessaly_2021 | 74 | 2.11 / 5.05 | 0 | 74 | 72 | 72 |
| samos_2020 | 16 | 3.00 / 4.55 | 0 | 15 | 6 | 6 |
| adriatic_2022 | 70 | 1.50 / 3.80 | 0 | 71 | 71 | 71 |
| etna_2022_2024 | 108 | 1.40 / 3.50 | 0 | 108 | 108 | 108 |
| corinth_thiva_2020 | 21 | 1.16 / 2.82 | 0 | 22 | 20 | 20 |
| kahramanmaras_2023 | 132 | 2.60 / 6.17 | 16 | 132 | 113 | 20 |
| noto_2024 | 252 | 2.40 / 6.07 | 3 | 105 | 0 | 0 |
| hualien_2024 | 198 | 2.66 / 5.92 | 12 | 107 | 0 | 0 |
| petrinja_2020 | 58 | 1.70 / 4.35 | 33 | 38 | 19 | 19 |
| reykjanes_2023 | 107 | 3.29 / 5.05 | 62 | 70 | 0 | 0 |
| fagradalsfjall_2021 | 30 | 3.60 / 5.43 | 20 | 10 | 0 | 0 |
| la_palma_2021 | 163 | 1.30 / 3.30 | 0 | 163 | 163 | 163 |
| santorini_2025 | 201 | 0.35 / 4.87 | 0 | 201 | 199 | 199 |
| west_bohemia_2018 | 162 | 1.40 / 3.20 | 0 | 159 | 0 | 0 |
| maurienne_2017 | 63 | 0.43 / 3.63 | 0 | 63 | 61 | 61 |
| noto_swarm_2023 | 602 | 0.50 / 5.00 | 13 | 0 | 0 | 0 |
| campi_flegrei_2023 | 14 | 1.00 / 2.20 | 0 | 5 | 5 | 5 |

Events are the pick file's event identifiers; the "events with reference_ok
picks" column can exceed the catalogue count by one where the pick harvest
and the stage-1 catalogue differ (Adriatic 71 vs 70, Corinth-Thiva 22 vs
21). Kaikōura's 67 of 135 events with three-station support reflects
GeoNet's station geometry at 52 to 195 km from the fetched six.

## Open acquisition tasks

Owner is Marine for every external account or contact. Access state is
what the existing docs record (`data/heldout_testset/README.md` "Known
gaps", the registry notes, `docs/2026-09-08_heldout_test_cases.md` "Data
status"); nothing was re-checked against the providers here. "Today" is
from the tables above. "Would become eligible" is conditional on the counts
after the data arrive, and stays provisional until 37B.

| Task | Case | Owner | Access state as recorded | Today | Would become eligible |
|---|---|---|---|---|---|
| NIED Hi-net account: JMA unified picks (`HinetPy get_arrivaltime`) and Hi-net waveforms | noto_2024 | Marine | account needed; public JMA deck files end 2023-12 | 2 fetched stations (JP.JSD 20 Hz at 106 km, IU.MAJO at 135 km); 7,169 unresolved in-window picks on Hi-net/JMA codes (JJH, JTT, JHG, JJN, JSZ); manual S on all stations 41 | both endpoints, only if the JMA picks (manual) come with the waveforms; waveforms alone leave S under N |
| NIED Hi-net account: waveforms for the 54 JMA stations already located in `station_map.csv` (N.SUZH 3 km) | noto_swarm_2023 | Marine | account needed; picks already in the file (JMA deck, 4,224 / 5,181 manual P / S in the window) | zero overlap: 0 picks on the 2 fetched stations, 12,976 unresolved | both endpoints (602 catalogue events in the window) |
| CWA GDMS account: CWASN waveforms | hualien_2024 | Marine | account needed; ISC carries the CWB picks with station codes | 1 fetched station (JP.YOJ 20 Hz at 164 km); 21,850 unresolved in-window picks on CWASN codes (ETM, EHP, WHF, ETL, NNS); manual S on all stations 178, unknown_ok preliminary 19,095 / 8,101 | network/event scoring; S pick scoring needs the CWB picks reclassified or the reviewed bulletin |
| Refetch the TW (BATS) stations from EarthScope | hualien_2024 | pipeline rerun, no account | registry note says TW is on EarthScope; the build fetched none (build.log not committed) | TW.YHNB 131/127, TW.SSLB 126/114, TW.YULB 121/78, TW.TDCB 118/103 in-window picks, registered and unfetched | lifts `single_station`; endpoint counts depend on the manual tier on those stations |
| IMO: SIL station list and manual picks | reykjanes_2023 | Marine | "on request"; ISC preliminary for 2023-11 | only II.BORG carries picks (25 / 18 manual); Y7.DC01 to DC05 at 20 km carry none; 6,527 unresolved on SIL codes (ILFE, IVOG, IVOS, IKRI, IGRV) | both endpoints, once the codes resolve to fetched stations |
| IMO: SIL station list and manual picks | fagradalsfjall_2021 | Marine | ISC reviewed; SIL codes barely resolve against the EIDA inventory | II.BORG 15/8 and 4Q.MA2 1/0; OR.KRIST, VI.KAS, VI.SAN, VI.VOS fetched with no picks; 2,954 unresolved | both endpoints, once the codes resolve |
| AFAD manual readings | kahramanmaras_2023 | Marine | "to be added when bulk access is confirmed" | network/event eligible (20 events); manual S 37 in coverage, 48 on all stations; 8,738 unresolved, 22 with a fetched station's code | pick scoring (S), hence both |
| Croatian Seismological Survey (Zagreb): bulletin and CR waveforms | petrinja_2020 | Marine | not recorded beyond "Zagreb (ZAG) and neighbours; CR and SL on EIDA" | both endpoints on the reviewed `unknown_ok` tier only (0 manual); CR.MOSL 33/34, CR.OZLJ 34/32, CR.ZAG 34/15 registered at 44 to 65 km and unfetched; 3,309 unresolved | no endpoint change; a manual tier and the nearest stations |
| WEBNET Zenodo tarball `waveforms.tar.gz` (966 MB) | west_bohemia_2018 | Marine or the lab server (Zenodo answers 403 to the build host) | 403 at build time; `catalog_2018swarm_quality1.pha` and `station_coordinates.txt` refused the same way | `single_station`: CZ.NKC 137/157; BW stations at 20 to 34 km carry none; 2,781 unresolved on KRC, LBC, POC, VAC, STC | both endpoints (1,485 / 1,593 manual P / S in windows, 162 catalogue events) |
| INGV-OV bulletin | campi_flegrei_2023 | Marine | "the real reference, not on an FDSN service" | 30 / 26 manual, 5 events with three-station support, 14 catalogue events in the windows from the national service | both endpoints, subject to the OV counts |

## Left for 37B and beyond

- 37B: the immutable eligibility table for the sealed panel, per-endpoint
  minimum support and precision, predeclared fallbacks, without looking at
  candidate scores. The provisional rows above are its input.
- Reference validation beyond tiers: onset, uncertainty and time-weight
  columns exist in `picks.parquet` and are not used here.
- Extension of cases to early coda, later hours or days and ordinary
  conditions (issue #37 acceptance) is not started; the station tables
  give the starting point.
- The `BULLETIN_REVIEWED` table should be checked against ISC once, and
  the two 2024 cases re-harvested when the reviewed bulletin reaches them.

## Validation

```sh
python -m pytest tests -q          # 52 passed
python scripts/certify_evaluability.py --waveform-root <clone>/data/heldout_testset
```

`tests/test_certify_evaluability.py` builds synthetic case directories
with MiniSEED written by obspy and checks: the four resolution categories
and the quarantine of ambiguous codes; tier counting and the reviewed flag;
duplicate collapse; the three-station event rule and the three verdict
outcomes; `zero_overlap`, `single_station` and `no_waveforms`; gap and
component measurement (a 20 s gap, a two-component file, a missing file);
that an acceptance case gets `provisional = True` and no `certified_by`
while its `reference_qa` access is logged with the rule parameters; and
that an unknown key fails before any access.
