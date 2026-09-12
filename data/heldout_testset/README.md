# Held-out test set

*Built by `scripts/build_heldout_testset.py` from the registry in
`scripts/heldout_testset_registry.py`, 2026-09-08 onward. One directory per
sequence of `docs/2026-09-08_heldout_test_cases.md`.*

Current suite roles and scoring permissions come from
[`configs/evaluation_suites.json`](../../configs/evaluation_suites.json),
not historical manifests. See the [suite policy](../../docs/2026-09-10_suite_policy.md)
for regression cases, protected acceptance candidates and QA access logging.

## What is here and what is pinned

| File | Committed | Content |
|---|---|---|
| `catalog.parquet` | yes | stage-1 catalogue used to choose the scoring windows: event, origin, lat, lon, depth_km, mag, source |
| `windows.csv` | yes | the scoring windows, with the rule that chose them |
| `picks.parquet` | yes | the reference picks, QuakeScope schema (`sequence, event, origin, mag, station, channel, phase, time, mode, status, method, agency, time_weight, onset, uncertainty`) plus `network`, `source`, `reference_ok` |
| `station_map.csv` | yes | bare station codes (ISC, JMA) resolved to `NET.STA` against the inventories, with ambiguities listed |
| `stations.csv` | yes | candidate stations within the radius, reference-pick counts inside the windows, distance, route, and whether waveforms were fetched |
| `manifest.json` | yes | the registry entry, git commit, Python/obspy/pandas versions, every query with its service URL and UTC time, counts, and the sha256 and size of every file including the waveforms |
| `evaluability_stations.csv` | yes | one row per registered station and scoring window: waveform coverage, gaps and reference counts inside the coverage, written by `scripts/certify_evaluability.py` (#37A); the case-level table is `../evaluability.csv` |
| `waveforms/*.mseed` | no | raw MiniSEED, one file per station, band and window; re-fetchable and verifiable against the manifest |
| `build.log`, `cache/` | no | the run log; downloaded JMA and Zenodo files |

The picks are stored exactly as the QuakeScope notebooks store their
harvest, so `tutorials/phasenet_global_sequences.ipynb` reads them
unchanged. Two columns are added. `source` names the service. `reference_ok`
is the row filter the scorer uses: `mode == "manual"`, or `mode` unknown for
ISC-sourced picks, because the ISC bulletin relays the agencies' reviewed
readings without an evaluation mode. The notebooks' own filter,
`mode == "manual"`, still works and is stricter.

## Reproducing

```bash
python scripts/build_heldout_testset.py --sequence petrinja_2020 all     # skips steps whose output exists
python scripts/build_heldout_testset.py --sequence petrinja_2020 all --force
python scripts/build_heldout_testset.py --verify petrinja_2020            # re-hash every file against manifest.json
python scripts/build_heldout_testset.py --index                           # rebuild index.csv
```

The event services serve revisable catalogues: a re-harvest months later
can return different picks. The committed `picks.parquet` is the reference;
`--force` makes a new one and the manifest records when. Waveforms are
deterministic for a fixed query as long as the archive does not change, and
the manifest's sha256 says whether it did.

## Scoring windows

Mainshock sequences use the notebooks' rule: the window opens 600 s after
the mainshock, past its coda, and lasts 120 or 180 min. Swarm and
volcano-tectonic sequences take the two most populated non-overlapping 3 h
windows of the stage-1 catalogue inside a fixed span, on a 15 min grid,
earliest first on ties; the count is written into `windows.csv`.

## Station selection

Candidates are the stations within the radius plus 0.5° that have an HH, EH
or BH band in the inventory of the sequence's waveform routes, ranked by
reference picks inside the windows, then distance, then name. The first six
that return at least half of a window are kept, plus the USGS GSN station
listed for the sequence. Bare station codes in ISC and JMA picks are matched
to the inventory by code; when two networks share a code the first in
alphabetical order is taken and both are recorded in `station_map.csv`.

## Known gaps (2026-09-08)

- West Bohemia: the classic WEBNET stations carrying the picks are not on
  EIDA (only NKC, as CZ.NKC); their waveforms are in the Zenodo tarball
  `waveforms.tar.gz` (966 MB), not fetched while Zenodo answers 403 to this
  host. The optional `catalog_2018swarm_quality1.pha` and
  `station_coordinates.txt` were refused the same way; the build proceeds
  on the cached main phase file and says so in `build.log`.
- Fagradalsfjall 2021 and Reykjanes 2023: ISC station codes barely resolve
  against the EIDA inventory of IMO's VI network; the reference on the
  fetched stations is thin until IMO's picks and station list are obtained.
- Campi Flegrei: the INGV national service holds few caldera events; the
  INGV-OV bulletin is the reference to obtain.
- `git_dirty` in `manifest.json` lists uncommitted changes to the pipeline
  at build time; manifests built before it was added (the first fourteen
  sequences) record the commit but not the working-tree state, which for
  them was the code committed in the following commit with robustness
  fixes only, except the JMA parser fixes, after which the Noto swarm was
  rebuilt.

- Noto 2024 and the Hualien 2024 CWASN stations need NIED Hi-net and CWA
  GDMS accounts for waveforms; the picks for Noto 2024 need Hi-net too
  (`HinetPy`), the public JMA deck files end in 2023-12.
- Kahramanmaraş 2023 relies on the ISC preliminary bulletin and the USGS
  phase-data product (M ≥ 4.5) until AFAD's manual readings are obtained.
- Reykjanes 2023 likewise until IMO's SIL picks are obtained.
