# Held-out test cases for three regimes

*2026-09-08, Marine Denolle with Claude, branch `audit/2026-09-07-generalization`.
Companion to `docs/2026-09-07_training_plan.md` (v2). The windows and places
below are enforced by `scripts/heldout_sequences.py` and applied at build
time by `scripts/build_training_dataset.py`; the counts of training traces
they remove are produced on the server by
`scripts/audit_heldout_sequences.py`, not yet run.*

## Why sequences, and why places

Akash's benchmark (`notebooks/benchmark_manifest.csv`) is 35,392
single-arrival windows from twelve SeisBench datasets, with one pick per
window and no sequence context. It measures timing on isolated arrivals.
None of the three regimes that decide whether a picker is useful in a
campaign (a mainshock's first two days, a volcano waking up, a swarm
migrating for months) is represented in it, and every one of its datasets
is in the fine-tune's training manifest. It stays as a unit test.

The cases below are scored on continuous data at the event level after
association, against picks an operator or a published study made by hand,
with the same matched-budget scoring the QuakeScope notebooks use.

Two kinds of hold-out. Mainshock-aftershock sequences are held out as
time-bounded windows around the epicentre, as Kaikōura, Norcia and
Thessaly already are. Volcano-tectonic sequences and fluid-driven swarms
recur at the same place, and the SeisBench sets contain those places
(INSTANCE holds Etna and Campi Flegrei from 2005 to 2020; VCSEIS holds
Alaska, Hawaii, northern California and the Cascades; CREW is global at
regional distance). A time window would leave the same volcano's earlier
years in training, which tests memory of a place, not generalisation. So
those are held out as places, at all times, and the cost in training data
is accepted. Tier 1 places were never in any training set once the
exclusion runs; tier 2 are known places at new times, listed for
development use, not for the generalisation claim.

## Regime 1: mainshock-aftershock sequences

| Sequence | Mainshock | Reference picks | Access | Status |
|---|---|---|---|---|
| **Kahramanmaraş 2023** (Türkiye) | 2023-02-06, Mw 7.8 and 7.6 | AFAD national catalogue: 566,329 P and 461,033 S manual readings for 50,085 events, Feb 6 to Oct 31 2023, used for the double-difference relocations in *Acta Geophysica* (2024); NLL-SSST-coherence catalogue of 17,248 events (Lomax, Zenodo 7727678); GFZ machine-learning catalogues for the first five days; 200-node SmartSolo deployment from May 2023 | AFAD and KOERI waveforms through their FDSN services; pick access via AFAD to confirm | acceptance, tier 1 |
| **Noto 2024** (Japan) | 2024-01-01, Mw 7.5 | JMA unified catalogue picks; high-precision aftershock catalogue from 30 temporary stations at ~5 km spacing (Takahashi et al. 2026, *GRL*); 3-D relocation with uncertainty (*EPS* 2025) | Hi-net and JMA data need NIED registration, not FDSN; picks from the JMA unified catalogue | acceptance, tier 1; after the parent's 2014–2021 training years, so held out for `jma_wc` too |
| **Hualien 2024** (Taiwan) | 2024-04-02, Mw 7.4 | CWA manual P and S for 2,011 relocated aftershocks (Xu et al. 2025, *GRL*, from the GDMS); a 118-station catalogue with 260,659 P and 231,159 S picks (*Tectonophysics* 2026); AutoQuake ML catalogue (Yang et al. 2025, *JGR*) | CWA GDMS registration; the SeisBench `cwa` set ends in 2021 | acceptance, tier 1 |
| **Petrinja 2020–21** (Croatia) | 2020-12-29, Mw 6.4 | 13,837 events over six months located from 255,729 hand-picked onsets, complete to ML 1.2 (*Tectonophysics* 2023; *J. Seismol.* 2024); EQTransformer+PyOcto catalogue (EGU 2024) | CR network on ORFEUS EIDA to confirm; picks from the Croatian Seismological Survey | acceptance, tier 1 |
| **Samos 2020** (Aegean) | 2020-10-30, Mw 7.0 | NOA, AUTH and KOERI relocated catalogues (Karakostas et al. 2021; Kiratzi et al. 2021; Lentas et al. 2021); NOA manual picks harvestable as at Thessaly | NOA FDSN event service, proven | development, tier 1 |
| **Adriatic 2022** (offshore Italy) | 2022-11-09, Mw 5.5 | INGV bulletin picks; ML-based catalogue (*Sensors* 2024) | INGV FDSN event service, proven; after INSTANCE's 2020 end | development, tier 1; the offshore case |

Ridgecrest 2019, Monroe 2019, Kaikōura 2016, Norcia 2016 and Thessaly 2021
stay in the acceptance suite as already held out.

## Regime 2: volcano-tectonic sequences

| Sequence | Period | Reference picks | Access | Status |
|---|---|---|---|---|
| **Reykjanes Peninsula** (Iceland): Fagradalsfjall 2021 dike; Sundhnúkur–Grindavík dikes from 2023-11-10 | 2021; 2023–2025 | IMO SIL manual catalogue; >80,000 microearthquakes by QuakeMigrate on a dense local network before the 2021 eruption; double-difference relocations (Fischer et al. 2022, *EPSL*); Sigmundsson et al. 2024 *Science* for the 2023 dike | IMO waveforms partly on EIDA; pick access via IMO to confirm | acceptance, tier 1 (place) |
| **La Palma 2021** (Canary Islands) | 2021-09-11 to 2021-12 | IGN catalogue (manual), relocations and tomography (D'Auria et al. 2022, *Sci. Rep.*); fully automated pre-eruptive series (*JVGR* 2023) | IGN FDSN services; 2021 is a held-out year anyway | acceptance, tier 1 (place) |
| **Santorini–Amorgos 2025** (Greece) | 2025-01 to 2025-03 | Manually picked catalogue from AUTH and NOA relocated with NLL-SSST-coherence, plus a 34,442-event ML catalogue (*Seismica* 2025; *Science* 2025 on the dike) | NOA FDSN, proven | acceptance, tier 1 (place); magmatic dike with triggered tectonic seismicity, the mixed case |
| **Etna** (Italy) | 2022–2024 | INGV-OE catalogue; multi-scale relocations 2014–2023 (*Annals of Geophysics* 2024) | INGV FDSN, proven; Etna removed from INSTANCE by the place hold-out | development, tier 1 (place) |
| **Mayotte 2018–19** (Comoros, offshore) | 2018-05 to 2019-05 | GFZ catalogue of 6,990 VT events (Cesca et al. 2019, doi:10.5880/GFZ.2.1.2019.004); BRGM catalogues; >3,000 events manually reviewed on OBS during MAYOBS (*GJI* 2021) | RESIF/EIDA for land stations; OBS data on request; must be checked against the OBST2024 deployment list | development, tier 1 (place); the offshore OBS case |
| Mauna Loa 2022, Kīlauea 2023–24 (Hawaii) | 2022-09 to 2024 | HVO catalogue (ComCat with arrivals); precursory seismicity 2013–2022 (*Bull. Volcanol.* 2025) | open (HV on EarthScope, ComCat) | development, **tier 2**: Hawaii is in VCSEIS; usable only if the VCSEIS traces end before these dates, to be read from its metadata on the server |
| Mount Spurr 2024–25 (Alaska) | 2024-04 to 2025-08 | AVO/AEC catalogue; no paper yet | open | development, **tier 2**: Alaska is in VCSEIS |

## Regime 3: fluid-driven swarms

| Sequence | Period | Reference picks | Access | Status |
|---|---|---|---|---|
| **West Bohemia 2018** (Czechia) | 2018-05 to 2018-06 | WEBNET: >4,000 events, >1,500 with ML ≥ 1.3 processed manually; Zenodo dataset with waveforms, station XML and a relocated catalogue (Eulenfeld, doi:10.5281/zenodo.3741465); *GJI* 2021 | Zenodo, open | acceptance, tier 1 (place); CO2-driven, the type example |
| **Maurienne 2017–19** (French Alps) | 2017-08 to 2019-03 | SISmalp/RESIF; 71,064-event template-matched and relocated catalogue, Mc 0.7 (Minetto et al. 2022, *ESS*) | RESIF FDSN; manual picks via BCSF-RENASS to confirm | acceptance, tier 1 (place); fluid and slip interplay |
| **Noto swarm 2020–23** (Japan) | 2020-12 to 2023-12 | JMA unified catalogue; >20,000 relocated events, updip migration (Yoshida et al. 2023, *GRL*; Shelly 2024, *GRL*; *EPS* 2024) | as Noto 2024 | acceptance, tier 1 (time window); 2022–23 also after the parent's training years |
| **Campi Flegrei 2023–24** (Italy) | 2023-01 to 2024-12 | INGV-OV manual picks; ML catalogue of ~8,400 events (PhaseNet+GaMMA) and a located catalogue 2000–2023 (*Nat. Commun.* 2025; *Commun. Earth Environ.* 2025) | INGV FDSN, proven; removed from INSTANCE by the place hold-out | acceptance, tier 1 (place); hydrothermal-magmatic |
| **Corinth–Thiva 2020–22** (Greece): Perachora and Thiva | 2020-01 to 2022-06 | NOA/AUTH relocated catalogues; template-matching to 24,000 events at Perachora (*Sensors* 2023); Thiva clustering (*Entropy* 2025) | NOA FDSN, proven | development, tier 1 (place) |
| Mayotte 2018–19 | see regime 2 | | | also a deep magmatic swarm |

## What the exclusion does

`scripts/heldout_sequences.py` now carries 24 windows: the original eight,
seven new time-bounded ones (Kahramanmaraş, Noto 2024, Hualien, Petrinja,
Samos, Adriatic, the Noto swarm) and nine places with no time bound
(Reykjanes Peninsula, La Palma, Santorini–Amorgos, Etna, Mayotte, Campi
Flegrei, West Bohemia, Maurienne, Corinth–Thiva). Each carries a `regime`
and a `tier`. A training trace is dropped if its source origin falls in
any of them, and the 2016 and 2021 whole-year hold-out stands. The same
windows must be applied to the noise pool (§5 of the plan): noise windows
from a held-out place or time teach the station character.

Radii are deliberately generous (0.15° for Campi Flegrei, which stops short of Vesuvius, up to 2.5° for
the 350 km Kahramanmaraş rupture). Over-excluding a few hundred unrelated
events is cheap; leaving one aftershock in is not.

## What has to be verified on the server

- The counts: `python scripts/audit_heldout_sequences.py`. The new places
  will remove Etna and Campi Flegrei from INSTANCE and whatever CREW holds
  around the other places; the numbers decide whether any of them is too
  costly.
- `vcseis` and `crew` metadata date ranges, and the `obst2024` deployment
  list, to settle the tier of Hawaii, Alaska and Mayotte.
- Waveform and pick access for Noto (NIED), Hualien (CWA GDMS), Petrinja
  (Croatian Seismological Survey), Reykjanes (IMO) and Maurienne
  (BCSF-RENASS); the rest are on FDSN services already used.

## Sources

- Kahramanmaraş: [Acta Geophysica 2024, long-term aftershock properties](https://link.springer.com/article/10.1007/s11600-024-01419-y); [Zenodo 7727678, NLL-SSST-coherence catalogue](https://zenodo.org/records/7727678); [GFZ ML catalogues](https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=8ea6a16e-b61e-11ee-967a-4ffbfe06208e); [Solid Earth 2024](https://se.copernicus.org/articles/15/197/2024/)
- Noto 2024: [Takahashi et al. 2026, GRL](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL118413); [EPS 2025 aftershock distribution](https://link.springer.com/article/10.1186/s40623-025-02227-4); [Yoshida et al. 2024, GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL110993)
- Hualien 2024: [Xu et al. 2025, GRL](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL115218); [Yang et al. 2025, JGR](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JB032792); [Tectonophysics 2026](https://www.sciencedirect.com/science/article/abs/pii/S0040195126000752)
- Petrinja: [Tectonophysics 2023](https://www.sciencedirect.com/science/article/abs/pii/S004019512300183X); [J. Seismol. 2024](https://link.springer.com/article/10.1007/s10950-024-10228-1)
- Samos: [Acta Geophysica 2021](https://link.springer.com/article/10.1007/s11600-021-00580-y); [Bull. Earthq. Eng. 2021](https://link.springer.com/article/10.1007/s10518-021-01260-4)
- Reykjanes: [Fischer et al. 2022, EPSL](https://www.sciencedirect.com/science/article/pii/S0012821X22003211); [Sigmundsson et al. 2024, Science](https://www.science.org/doi/10.1126/science.adn2838); [IMO overview](https://en.vedur.is/volcanoes/fagradalsfjall-eruption/)
- La Palma: [JVGR 2023, automated pre-eruptive series](https://www.sciencedirect.com/science/article/pii/S0377027323002032); [Sci. Rep. 2023 plumbing](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9870893/)
- Santorini–Amorgos: [Seismica 2025](https://seismica.library.mcgill.ca/article/view/1663); [Science 2025 dike](https://www.science.org/doi/10.1126/science.adz8538)
- Etna: [Annals of Geophysics 2024 relocations](https://www.annalsofgeophysics.eu/index.php/annals/article/view/9147)
- Mayotte: [GFZ catalogue, Cesca et al. 2019](https://dataservices.gfz-potsdam.de/panmetaworks/showshort.php?id=escidoc:4507894); [GJI 2021 OBS](https://academic.oup.com/gji/article/228/2/1281/6374867)
- Mauna Loa: [Bull. Volcanol. 2025](https://link.springer.com/article/10.1007/s00445-025-01860-x); [Nat. Commun. 2024](https://www.nature.com/articles/s41467-024-52881-7)
- Spurr: [AVO](https://avo.alaska.edu/eruption/spurr-unrest-2024)
- West Bohemia: [Zenodo dataset](https://zenodo.org/records/5016845); [GJI 2021](https://academic.oup.com/gji/article/224/3/1835/5955445)
- Maurienne: [Minetto et al. 2022, ESS](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021EA002097)
- Noto swarm: [Yoshida et al. 2023, GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106023); [Shelly 2024, GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL107897)
- Campi Flegrei: [Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-59821-z); [Commun. Earth Environ. 2025](https://www.nature.com/articles/s43247-025-02045-2)
- Corinth–Thiva: [Sensors 2023 Perachora](https://pmc.ncbi.nlm.nih.gov/articles/PMC10056727/); [Acta Geophysica 2022](https://link.springer.com/article/10.1007/s11600-022-00864-x); [Entropy 2025 Thiva](https://pmc.ncbi.nlm.nih.gov/articles/PMC12468699/)
- Training-set coverage: [VCSEIS / volpick](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL108438); [PNW 2002–2022](https://seismica.library.mcgill.ca/article/view/368); [CWA 2011–2021](https://pubs.geoscienceworld.org/ssa/srl/article/96/3/2079/650394/The-CWA-Benchmark-A-Seismic-Dataset-from-Taiwan); [CREW](https://seismica.library.mcgill.ca/article/view/1049); [OBST2024 / PickBlue](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023EA003332)
