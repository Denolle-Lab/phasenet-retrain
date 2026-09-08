### H3b. Benchmark traces inside a held-out window's RADIUS (space only; origin times are on the server)

| dataset | n_benchmark | n_with_location | months_known | norcia_2016_radius | kaikoura_2016_radius | thessaly_2021_radius | ridgecrest_2019_radius | monroe_2019_radius |
|---|---|---|---|---|---|---|---|---|
| aq2009gm | 1346 | 1346 | unknown | 1346 | 0 | 0 | 0 | 0 |
| ceed | 5435 | 5435 | unknown | 0 | 0 | 0 | 406 | 0 |
| cwa | 200 | 200 | unknown | 0 | 0 | 0 | 0 | 0 |
| ethz | 1458 | 1458 | unknown | 0 | 0 | 0 | 0 | 0 |
| instancecounts | 9776 | 9776 | unknown | 916 | 0 | 0 | 0 | 0 |
| mlaapde | 1350 | 1350 | 201307-201410 | 0 | 0 | 0 | 0 | 0 |
| obst2024 | 400 | 0 | unknown | 0 | 0 | 0 | 0 | 0 |
| pisdl | 765 | 765 | unknown | 0 | 0 | 0 | 0 | 0 |
| pnw | 2841 | 2841 | unknown | 0 | 0 | 0 | 0 | 987 |
| stead | 9360 | 9360 | unknown | 17 | 43 | 23 | 49 | 70 |
| txed | 2261 | 2261 | unknown | 0 | 0 | 0 | 0 | 0 |
| vcseis | 200 | 200 | unknown | 0 | 0 | 0 | 0 | 1 |

notebooks/benchmark_manifest.csv carries source lat/lon but an origin month only for mlaapde (2013-07 to 2014-10). Counts are an upper bound on benchmark traces that could sit in a window; the spatiotemporal join (scripts/audit_heldout_sequences.py, benchmark_manifest source) resolves them.
