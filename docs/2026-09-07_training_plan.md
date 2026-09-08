# Plan: a general picker for a global run with local resolution

*2026-09-07, Marine Denolle with Claude, branch `audit/2026-09-07-generalization`.
Supersedes the conditional v21 proposal of the same date as the thing to
act on; that proposal survives as the recipe for the training arms in §5.
Nothing here has been run.*

## 1. What the audit fixes about the design

Twenty finetunes of `jma_wc` on 527,477 windows drawn from twenty sources
did not beat the parent, which was trained on 6.1 million waveforms with
one labelling convention (Naoi et al. 2024). The one gain that survived a
paired comparison was 19 ms of P timing on picks both models make, and it
did not show on continuous data. Recipe changes were tried one variable at
a time and the recall never came back. The conclusion is about data, not
optimisation: half a million heterogeneous windows can perturb a model of
this size but not re-teach it, and every version paid for the perturbation
in recall.

Four things about the parent itself did surface, and they are the targets.

| Weakness of `jma_wc` | Where it was measured | Size |
|---|---|--:|
| Regional S | benchmark, 150–1500 km | S recall 0.36 [0.35, 0.38] |
| Low-SNR P | benchmark, below 0 dB | P recall 0.66 [0.64, 0.67] |
| Precision per pick | Kaikōura, Norcia, Thessaly, matched budget | `instance` leads by 3–12 points of P along the whole budget overlap |
| False triggers on noise | noise pool, own best threshold | precision 0.83; the EQTransformer ensemble reaches 0.96 at similar recall |

Two more were built into the training pipeline and never measured. The
training windows were cut with one labelled pick each
(`scripts/manifest_dataset.py:227-247`); a second event in the same 30 s
was present in the waveform and absent from the label, which teaches the
model to ignore it. That is the aftershock regime. And every waveform was
resampled to 100 Hz once, so nothing saw the 20, 40 and 50 Hz instruments
the campaign upsamples.

The teleseismic objective fought the regional one every time it was
raised (v13, v16, v17: teleseismic oversampling cost regional timing and
recall). Those two jobs get two models.

## 2. The target, written so it can be scored

The campaign runs one picker over every network EarthScope holds. What it
has to do well, in order:

1. **Local and regional sensitivity**, 0 to 300 km, down to the operator's
   completeness in dense networks and below it in sparse ones. Scored as
   recall at matched pick budget against the operator's manual picks, per
   phase and per distance bin, and as the false-pick rate per station-day
   on quiet days.
2. **Aftershock sequences.** The first 48 hours after an M6+ mainshock,
   events seconds apart, coda everywhere. Scored at the event level: the
   fraction of the operator's located events the pipeline recovers after
   association, by magnitude and by hour after the mainshock, plus picks
   per station-hour and the analyst residual.
3. **Distant P for completeness where there are no stations.** Offshore
   events and sparse regions such as most of Africa are recorded at 3 to
   30° on whatever exists. Scored as events located against the ISC and
   NEIC bulletins and as the completeness magnitude versus distance to the
   nearest station. This is a separate model (§6), not a property demanded
   of the regional picker.

The acceptance suite is fixed before training and never read during
development. It is the five sequences already held out (Kaikōura, Norcia,
Thessaly, Ridgecrest, Monroe), the whole years 2016 and 2021, and six more
sequences chosen for coverage of the campaign, each with an operator that
publishes manual arrivals through an FDSN event service: candidates are
Kahramanmaraş 2023 (KOERI or AFAD), Illapel 2015 (CSN Chile), Anchorage
2018 (AEC through USGS), Puebla 2017 (SSN Mexico), the Reykjanes 2021
swarm (IMO) and Botswana 2017 (sparse; ISC picks). Which services actually
return arrivals has to be tested with the harvest code of
`~/GitHub/QuakeScope/tutorials/phasenet_global_sequences.ipynb`; the
three proven ones are GeoNet, INGV and NOA. A development suite of
different sequences and years serves every decision before the final one.

## 3. Phase 0, before any training (two to three weeks)

Most of what the campaign needs this quarter is available without a GPU.

1. **Run task 1 on the server** (`python scripts/audit_heldout_sequences.py`),
   commit `data/exclusions/heldout_sequences.csv`. Nothing is built
   without it.
2. **Rank the candidates that already exist at matched budget** on the
   external suite: `jma_wc`, `instance`, `jma_wc_ft_global_v11` (the only
   finetune above the parent on the noise pool, 0.804 [0.799, 0.808] against
   0.776 [0.771, 0.780]), and three ensembles that cost no training:
   `jma_wc` + `instance`, `jma_wc` + `eqt_original_nonconservative`, and
   v7 + v11. Probability curves averaged, one threshold per ensemble.
   Ensembles were the best entries on the noise pool (0.823 to 0.859) and
   no one has scored them on continuous data.
3. **Set thresholds per weight and per region** to a false-pick target on
   quiet station-days, not to 0.3. The notebooks showed the threshold, not
   the model, produced most of the apparent ranking.
4. **Build the event-level scorer**: PyOcto over the picks, matched to the
   operator catalogue, events recovered by magnitude and by hour.

Gate: whichever candidate leads at matched budget on at least five of the
six non-US pairs and lowers the quiet-day false-pick rate replaces
`jma_wc` in the campaign now. Training continues regardless, because none
of these fixes regional S or low-SNR P.

## 4. Phase 1, the corpus (six to eight weeks)

The parent was taught by six million analyst picks with one convention.
The nearest thing available outside Japan is the operators' own reviewed
bulletins, which is also what the campaign is judged against. The corpus
is built from them.

**Sources.** Arrivals harvested from every operator whose FDSN event
service returns manual picks, the same code path as the notebooks: GeoNet,
INGV, NOA, USGS ComCat for the US networks (NC, CI, UW, NN, AK, HV and the
rest it covers), and, to be tested one by one, NRCan, Geoscience
Australia, IMO, SED, KOERI, CSN, SSN, GFZ, RESIF. Only picks flagged
manual, only P and S, only stations whose waveforms are open. The
SeisBench sets with analyst P and S (ETHZ, PNW, CWA, SCEDC and CEED,
TXED, Iquique, INSTANCE where the pick status says manual) are added with
the same exclusions and the label-error filter. No P-only sets; nothing
beyond 2000 km.

**Windows.** Cut from continuous data at native sampling rate, 60 s long,
with every arrival of every catalogued event inside the window labelled.
This is the single most important change for aftershock sequences, and
the bulletin harvest gives it for free: all picks on a station in a time
range, not one pick per trace. Each window carries SNR, epicentral
distance, magnitude, sampling rate, instrument, operator, year and the
number of events it contains.

**Size.** A one-week census first: events per year times stations with
picks per event, per operator, over the years the waveforms are open.
INGV and GeoNet alone review tens of thousands of events a year at
roughly ten picked stations each, so 3 to 5 million windows over a decade
of bulletins is the target to verify, not a guess to build on.

**Composition targets**, enforced by stratified sampling: at least 35 % of
windows below 5 dB on the P window (the benchmark's share is 33 % and the
low-SNR loss is where v7 fell behind); at least 30 % of windows with more
than one event; an S label in at least 60 % of windows; a distance mix
that matches the campaign's station geometry, which at Kaikōura and
Thessaly is regional-heavy, 80 to 120 km for most reviewed picks; no
single operator above 30 % of the corpus.

**Exclusions and hygiene.** Held-out sequences, 2016, 2021, the
development suite, benchmark traces and events; `scripts/hash_manifests.py`
fingerprints; `scripts/audit_heldout_sequences.py --check-manifest` must
pass on train and val; the acceptance suite is never opened.

**Label checks.** The confident-learning filter as before, plus two
physical ones the audit found wanting: S minus P against distance for
every operator, and a per-operator recall cap test of the kind that
exposed Thessaly's S at 0.5 for every model, which points at the reference
rather than the pickers.

## 5. Phase 2, training experiments (eight to twelve weeks)

Every run uses the recipe of `docs/2026-09-07_v21_proposal.md` §2 and §3:
random window position, real-noise superposition from `data/noise_global`
at 0 to 25 dB, band-limiting and resampling across 20 to 100 Hz, channel
dropout, soft Gaussian targets, LR 5e-6, early stopping on a validation
loss and nothing else. Selection is on the development suite at matched
budget with paired intervals; the acceptance suite is run once, at the
end.

**E1, the data-scaling curve, before anything else.** Initialise from
`jma_wc`, distillation at T = 1.5 and α = 0.3, and train on nested subsets
of 0.25, 0.5, 1, 2 and 4 million windows. Plot matched-budget recall
against the parent on the development suite versus corpus size. This is
the experiment the twenty versions never ran, and it answers the question
that decides everything else: does more data of this kind move the parent
at all? If the curve is flat at or below the parent by 2 million, the
PhaseNet finetune line stops here and the campaign keeps the Phase 0
winner; the remaining effort goes to §6 and to the associator.

**E2, initialisation and anchor**, at the largest size that helped in E1:
`jma_wc` init against a from-scratch PhaseNetWC (from scratch lost badly at
527k; at millions it may not), and α = 0 against α = 0.3.

**E3, augmentation ablation**, one arm each without real-noise
superposition, without resampling, without multi-event windows. The
multi-event arm is scored on the aftershock metric only.

**E4, architecture.** EQTransformer finetuned on the same corpus, and the
ensemble of the best PhaseNet arm with it. On the noise pool the
EQTransformer ensembles were the best detectors by a margin no PhaseNet
variant approached; whether that holds at matched budget on continuous
data is unknown and cheap to learn once the corpus exists.

Compute: v7 reached epoch 44 on 527k windows in one server session; the
per-epoch time is in `results/finetune_jma_wc_global_v7_metrics.csv` on
the server and sets the budget. Assume a 4 million window run is eight
times v7 per epoch and plan the E1 subsets to fit the GPUs available; E2
to E4 are three to five runs at one size.

## 6. Phase 3, a distant-P model for offshore and sparse regions

A 30 s window at 100 Hz is the wrong instrument for P at 10 to 30°. Rather
than pull the regional model toward it again, train a second PhaseNet at
20 Hz on 120 s windows, P only, from GEOFON, MLAAPDE, CREW and ISC-labelled
P at 3 to 30° from M ≥ 4, initialised from the SeisBench `geofon` weights
(the only model in the pool with teleseismic recall, 0.78 on the
benchmark). It runs only where it earns its cost: ocean-bottom
deployments, oceanic islands, and station-sparse regions defined by
nearest-neighbour station distance above 300 km. Its picks enter the
associator with a global velocity model. It is scored by events located
against ISC and NEIC and by the completeness magnitude versus nearest
station distance, on offshore and African test regions held out by year.

## 7. Phase 4, deployment

Thresholds belong to the weight and the region, set to a false-pick target
and recorded with the campaign. Association settings for sequences are
tuned on the aftershock metric, not on picks. Every change of weights
re-runs the two QuakeScope notebooks and the event-level scorer on the
acceptance suite; that is the standing acceptance test, and until a
candidate beats `jma_wc` at matched budget on the non-US sequences the
campaign stays on `jma_wc` or the Phase 0 ensemble.

## 8. Gates and timeline

| When | Deliverable | Decision |
|---|---|---|
| Week 3 | Task 1 list; matched-budget ranking of existing weights and ensembles; per-region thresholds; event scorer | Campaign picker for this quarter |
| Week 4 | Harvest census per operator | Corpus size and operator list |
| Week 11 | Corpus built, checked, fingerprinted; development and acceptance suites frozen | Go to E1 |
| Week 15 | E1 scaling curve | Continue the finetune line, or stop it and keep the Phase 0 winner |
| Week 22 | E2 to E4; one candidate per line | Acceptance run, once |
| Week 24 | Distant-P model first version | Offshore and sparse-region deployment |

## 9. What this plan does not do

No more single-variable changes on the v7 corpus; the corpus was the
problem. No teleseismic rebalancing inside the regional model. No
selection on `notebooks/step3_metrics.csv`; the benchmark stays as a unit
test of timing on single-arrival windows, which is what it measures.
No claim of a better picker before the acceptance run.

## References

- Zhu, W., & Beroza, G. C. (2019). PhaseNet. *GJI* 216, 261–273.
- Mousavi, S. M., et al. (2020). EQTransformer. *Nat. Commun.* 11, 3952.
- Münchmeyer, J., et al. (2022). Which picker fits my data? *JGR Solid Earth* 127, e2021JB023499.
- Woollam, J., et al. (2022). SeisBench. *SRL* 93, 1695–1709.
- Naoi, M., et al. (2024). PhaseNet models trained on the JMA unified catalogue. *EPS* 76, doi:10.1186/s40623-024-02091-8.
- Münchmeyer, J. (2024). PyOcto. *Seismica* 3(1).
- Sun, H., et al. (2023). Phase neural operator for multi-station picking. *Nat. Commun.* 14, 8283.
