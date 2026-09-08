### H3c. Which corpus datasets could hold each external sequence (documentation, not a join)

| sequence | corpus_datasets_covering_region_and_dates | status | basis |
|---|---|---|---|
| Norcia 2016 (INGV) | instancecounts (Italy, 2005-01 to 2020-01); stead (global, 2005-2018) | likely in v7's corpus | docs/2026-09-07_training_history_audit.md section 5 (date coverage only, no join yet) |
| Kaikoura 2016 (GeoNet) | stead (global, 2005-2018); crew (global regional) | possible | same |
| Thessaly 2021 (NOA) | crew (global regional) | possible; instancecounts ends 2020-01, stead 2018 | same |
| Ridgecrest 2019 (SCEDC/NCEDC) | scedc, ceed, ross2018gpd (S. California) | likely in v7's corpus | same |
| Monroe 2019 (UW) | pnw (Cascadia) | likely in v7's corpus | same |
| all five | manifests_v2 train/val + 10 full corpora | counts NOT computed: lab server unreachable 2026-09-07 | scripts/audit_heldout_sequences.py (task 1) must run on the server |
