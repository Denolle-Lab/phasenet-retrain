### H2a. S supervision in the training corpus: what can be bounded without the manifest

| quantity | value | basis |
|---|---|---|
| cap budget, all sources | 1453400 | DATASET_CONFIGS caps (scripts/build_training_dataset.py:209-319) |
| cap budget, P-only sources (geofon, lendb, meier2019jgr) | 340000 | use_s=False |
| share of cap budget that can never carry S | 0.234 | upper bound; the manifest is drawn from these pools |
| teleseismic target fraction of the training split | 0.250 | TARGET_FRACTIONS (build_training_dataset.py:325-330); S nulled for every teleseismic window (P-only policy, ~line 570) |
| S fraction of manifests_v2/train.csv, all bins | 0.380 | data/README.md ('~527k train, ~38% S'); by-bin split needs the manifest (server) |
| S fraction by distance bin | not computed | data/manifests_v2/train.csv is on the lab server (unreachable 2026-09-07) |

The per-bin S fraction is the number H2 needs; it must be computed on the server with this script once data/manifests_v2/train.csv is reachable (verify with scripts/hash_manifests.py --check).
