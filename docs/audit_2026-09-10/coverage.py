"""Inspect reference availability without scoring any models. Requires pandas/pyarrow."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import heldout_testset_registry as registry
OUT = Path(__file__).parent
coverage = []
for seq in registry.SEQUENCES:
    d = ROOT / "data/heldout_testset" / seq["key"]
    if not (d / "picks.parquet").exists():
        continue
    picks = pd.read_parquet(d / "picks.parquet")
    stations = pd.read_csv(d / "stations.csv")
    fetched = set(stations.loc[stations.fetched, "station"])
    total = {"P": 0, "S": 0}
    manual = {"P": 0, "S": 0}
    selected_events = set()
    files = list((d / "waveforms").glob("*.mseed"))
    for w in pd.read_csv(d / "windows.csv").itertuples():
        lo, hi = pd.to_datetime(w.t0, utc=True), pd.to_datetime(w.t1, utc=True)
        sel = picks[picks.reference_ok & picks.station.isin(fetched)
                    & (picks.time >= lo) & (picks.time <= hi)]
        for (_, phase), group in sel.groupby(["station", "phase"]):
            kept = []
            for t in sorted(group.time):
                if not kept or (t - kept[-1]).total_seconds() > registry.MATCH_TOL_S:
                    kept.append(t)
            total[phase] += len(kept)
        for phase in ("P", "S"):
            manual[phase] += int(((sel.phase == phase) & (sel["mode"] == "manual")).sum())
        selected_events.update(sel.event)
    coverage.append(dict(key=seq["key"], suite=seq["suite"], stations=len(fetched),
        local_waveform_files=len(files), total_reference_ok=int(picks.reference_ok.sum()),
        window_fetched_P=total["P"], window_fetched_S=total["S"],
        manual_P_before_dedup=manual["P"], manual_S_before_dedup=manual["S"],
        events_with_reference_on_fetched_stations=len(selected_events)))
pd.DataFrame(coverage).to_csv(OUT / "reference_coverage.csv", index=False)
print(pd.DataFrame(coverage).to_string(index=False))
