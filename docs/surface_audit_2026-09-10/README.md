# Surface-plan audit evidence

`probes.py` runs synthetic checks and reads optional local metadata. It does not
load pretrained weights, train a model, score any real waveform, or fetch data.
It writes `probe_results.json` in this directory. Run from the repository root:

Activate an environment with the versions in `probe_results.json`, then run:

```sh
python docs/surface_audit_2026-09-10/probes.py
python -m unittest discover -s docs/surface_audit_2026-09-10 -p 'test_*.py'
```

If a writable Matplotlib cache is needed, set `MPLCONFIGDIR` to a local writable
directory. For a worktree outside the sibling-repository directory, pass
`--external-root /path/to/checkouts` to the probe command. This path is used for
reading only and is not written to the evidence.

The results pin library versions and code/input SHA-256 hashes. External files
are read from sibling `surface_events`, `QuakeScope` and `thunderquakes` checkouts
when present; their raw contents are not copied. Evidence identifiers use `repo/`,
`package/seisbench/`, `package/obspy/` and `sources/<repository>/` prefixes, not
machine paths. Preserve these identifiers and hashes when comparing runs.
Missing/invalid catalogue timestamps are counted separately from valid timestamps.

The model round-trip failure is a **finding** captured in the results, not a
passing deployment test. Likewise, the convolution-support probe excludes
normalization and uses artificial positive weights: it measures numerical
structural support, not learned effective context. Tone results use long pure
sinusoids with boundary samples removed and do not establish event performance.
Random weights have a fixed seed. Exact floating-point values may vary by platform.

The review follow-up adds 20/40→100 Hz tone tests with the Hann-response prediction,
a normalized categorical-target check and illustrative exact binomial recall
intervals. Interval examples assume independent events; they do not measure the
actual acceptance panel or replace event-family/station-day uncertainty.

Original proposal: `98543f3`, SHA-256 of the plan at that commit:
`8930cae058cb48c8015fff1c1f0d8eeb9d6b40a8ac8ed3af4390eb7b629d99e1`.

Primary sources inspected on 2026-09-10:

- https://seismica.library.mcgill.ca/article/view/368/868 — PNW annotation,
  curation and archive history.
- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023499 —
  picker training/evaluation and random-window rule.
- https://publikationen.bibliothek.kit.edu/1000143103/146775569 — archived PDF
  containing the 4-second PhaseNet receptive-field statement quoted by the draft;
  not a measurement of the installed variable-length implementation.
- https://seismica.library.mcgill.ca/article/view/2068 — classifier study;
  specific local deployment/placement findings were checked against the
  QuakeScope plan, not inferred from the paper's abstract.
- https://arxiv.org/abs/2311.13971 — low-frequency earthquake precedent and title.

The audit does not certify the original proposal's remaining bibliography,
API service counts or global waveform availability. Those claims remain inputs
to the proposed source census, not prerequisites silently counted as complete.
