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
| Mayotte 2018–19 (Comoros, offshore) | 2018-05 to 2019-05 | GFZ catalogue of 6,990 VT events (Cesca et al. 2019, doi:10.5880/GFZ.2.1.2019.004); BRGM catalogues; >3,000 events manually reviewed on OBS during MAYOBS (*GJI* 2021) | RESIF/EIDA for land stations; OBS data on request | **out of scope this round** (2026-09-08: no ocean-bottom observations); the reference rests on OBS picks |
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
| Mayotte 2018–19 | see regime 2 | | | also a deep magmatic swarm; out of scope this round |

## What the exclusion does

`scripts/heldout_sequences.py` now carries 23 windows: the original eight,
seven new time-bounded ones (Kahramanmaraş, Noto 2024, Hualien, Petrinja,
Samos, Adriatic, the Noto swarm) and eight places with no time bound
(Reykjanes Peninsula, La Palma, Santorini–Amorgos, Etna, Campi Flegrei,
West Bohemia, Maurienne, Corinth–Thiva). Mayotte was removed on
2026-09-08 with the ocean-bottom scope. Each carries a `regime`
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
- `vcseis` and `crew` metadata date ranges, to settle the tier of Hawaii
  and Alaska.
- Waveform and pick access for Noto (NIED), Hualien (CWA GDMS), Petrinja
  (Croatian Seismological Survey), Reykjanes (IMO) and Maurienne
  (BCSF-RENASS); the rest are on FDSN services already used.

## Data status (2026-09-08, evening)

The test data are built by `scripts/build_heldout_testset.py` from
`scripts/heldout_testset_registry.py` and live under `data/heldout_testset/`
(one directory per sequence; `index.csv` is the summary, `README.md` the
layout and the reproducibility rules). Picks are stored in the QuakeScope
schema; the Thessaly harvest reproduces the QuakeScope cache row for row
(3,650 picks, identical times), which is the pipeline's calibration.

What the builds showed, sequence by sequence, beyond the counts in
`index.csv`:

- **Reference picks are rich where the operator's own service serves them**
  (INGV, NOA, GeoNet, franceseisme, the JMA deck file, the Zenodo phase
  file) and where ISC relays a reviewed bulletin (Petrinja, La Palma,
  Fagradalsfjall). For 2023–2025 events ISC is preliminary and the picks
  carry no evaluation mode; they are flagged `reference_ok` on that basis
  and the USGS phase-data product adds NEIC's manual picks for M ≥ 4.5.
- **Station resolution is the weak point of ISC-sourced references.** ISC
  gives station codes without networks. They resolve against the EIDA or
  EarthScope inventories for La Palma (17 of 21 codes), Petrinja (18 of the
  328 codes lie within the radius) and Kahramanmaraş (24 of 962), but
  barely for Fagradalsfjall (2 of 558) and Reykjanes 2023 (1 of 378),
  because IMO's SIL stations are not in the EIDA inventory for those dates.
  The waveforms fetched there are the few VI stations on EIDA and II.BORG;
  the reference on them is thin until IMO's picks and station list arrive.
- **Noto 2024 and Hualien 2024** have large references (16,012 and 32,461
  arrivals) and almost no waveforms: IU.MAJO and JP.JSD for Noto, IU.TATO
  for Hualien. Hi-net and CWA GDMS accounts are the missing piece. The
  **Noto swarm** (JMA deck, 13,461 arrivals, 54 JMA stations with
  coordinates in `station_map.csv`, N.SUZH 3 km from the swarm) is in the
  same position.
- **West Bohemia**: the Zenodo phase file gives 9,769 picks on the classic
  WEBNET stations (KRC, STC, POC, LBC, VAC, NKC, SKC, KVC …), of which only
  NKC is served on EIDA (as CZ.NKC); the others' waveforms are in the
  Zenodo tarball (966 MB), to be fetched when Zenodo stops answering 403 to
  this host. The Bavarian BW stations fetched instead sit 40 to 90 km away.
- **Campi Flegrei**: the INGV national event service holds few of the
  caldera's small events (71 in nine days, 5 with arrivals in the two
  windows); the INGV-OV bulletin is the real reference and is not on an
  FDSN service.
- **Kaikōura, Norcia, Thessaly, Adriatic, Samos, Maurienne, Santorini,
  Etna** built cleanly with six stations per window.

## Where the picks are, and what USGS stations sit nearby

*Added 2026-09-08 on Marine's request. "Server" is the service that holds
the manual picks; "waveforms" the open archive; "GSN" the nearest
USGS-operated Global Seismographic Network station, whose picks the NEIC
publishes for the events it locates.*

Two servers hold picks for most of these sequences regardless of the
operator, and are the first place to look:

- **ISC Bulletin** (`https://www.isc.ac.uk/fdsnws/event/1/` with
  `includearrivals=true`, and the dedicated arrivals service
  `https://www.isc.ac.uk/iscbulletin/search/arrivals/`): the archive of
  phase readings the operators report, including JMA, AFAD and KOERI,
  CWA, IMO (agency REY), Zagreb (ZAG) and Strasbourg (STR). The reviewed
  bulletin runs about two years behind real time, so 2023 sequences are
  reviewed and 2024 ones may still be preliminary.
- **USGS ComCat** (`https://earthquake.usgs.gov/fdsnws/event/1/` with
  `includearrivals=true`; the `phase-data` product): picks for every
  event the NEIC locates, which outside the United States means roughly
  M ≥ 4.5, on GSN stations and on stations contributed to the NEIC. This
  covers the larger aftershocks of every sequence below, not the small
  ones a picker is judged on, but it is open and needs no account.

| Sequence | Picks (server) | Waveforms | Nearest USGS GSN station |
|---|---|---|---|
| **Noto 2024** and the **Noto swarm** | JMA publishes the unified catalogue's arrival-time files ("検測値データ") itself: `https://www.data.jma.go.jp/eqev/data/bulletin/deck_e.html`, monthly files with a documented format (`.../data/format/datfmt_e.html`), no account. The same files come through Hi-net with a free NIED account, and **HinetPy** (`github.com/seisman/HinetPy`, `Client.get_arrivaltime(startdate, span)`) downloads them in Python; `Client.get_waveform` fetches Hi-net continuous data. | Hi-net and F-net through NIED (account; HinetPy), IU.MAJO and the JP-contributed stations on EarthScope | IU.MAJO, 1.2° (134 km) from the Noto epicentre: JMA and NEIC both pick it |
| **Kahramanmaraş 2023** | AFAD event catalogue (`https://deprem.afad.gov.tr/event-catalog`, web service `apiv2/event/filter`; phase readings on the event pages, bulk access to confirm with AFAD); KOERI bulletin; ISC (ISK, AFAD). The ESSD 2025 data set (Colavitti et al., Zenodo 10.5281/zenodo.13838992) has 9,442 events and 270,704 phases 2019–2024 with 271 stations, but the picks are from an automatic processor with quality control, not analysts, so they are a check, not the reference. | KOERI EIDA node (KO network, open); AFAD TDVMS (TK, request); the 2023 SmartSolo nodal deployment on EarthScope (embargo to check) | IU.ANTO, 4.2° (470 km): NEIC picks for M ≥ 4.5 aftershocks only |
| **Hualien 2024** | CWA GDMS (`https://gdms.cwa.gov.tw`, registration; event catalogue and phase files since 1991; bulk membership via `opendata.cwa.gov.tw`); ISC (CWB) | CWASN via GDMS; BATS (TW network, Academia Sinica) open on EarthScope | IU.TATO, 1.2° (130 km): NEIC picks for M ≥ 4.5 |
| **Petrinja 2020–21** | Croatian Seismological Survey, Zagreb (`pmf.unizg.hr/geof`): the 255,729 hand-picked onsets of the *Tectonophysics* 2023 catalogue, on request to the authors; ISC (ZAG) | CR network on ORFEUS EIDA (open per the EIDA network list); SL (Slovenia), HU, OE and IV neighbours open on EIDA | none within 500 km; ComCat has the mainshock and the few M ≥ 4.5 aftershocks |
| **Samos 2020** | NOA FDSN event service with arrivals (proven at Thessaly); KOERI; AUTH relocated catalogues on request | NOA EIDA node (HL), KOERI EIDA (KO), open | IU.ANTO, 5.1° (560 km): M ≥ 4.5 only |
| **Reykjanes 2021, 2023–25** | IMO: SIL manual picks; a "Quakes API" is replacing the SIL web pages as IMO moves to SeisComP; bulk picks on request to IMO; ISC (REY) | IMO permanent stations: only project periods on EIDA until an Icelandic node exists; the 2021 dense local networks on EarthScope with embargo to check | IU.BORG, 1.0° (110 km): NEIC picks for the many M ≥ 4.5 of the November 2023 dike |
| **Maurienne 2017–19** | BCSF-RENASS: `api.franceseisme.fr` fdsnws-event (arrival support to test); the Minetto et al. 2022 catalogue on request; ISC (STR) | RESIF FDSN (FR, RA), open | none nearby |
| **West Bohemia 2018** | Zenodo 10.5281/zenodo.3741465: waveforms, StationXML and the relocated catalogue with picks, open | in the Zenodo record; WEBNET (WB) on EIDA | none nearby; GR (BGR) open |
| **Santorini–Amorgos 2025**, **Corinth–Thiva** | NOA FDSN event service with arrivals; AUTH manual picks on request | NOA EIDA node, open | none nearby |
| **Campi Flegrei 2023–24**, **Etna**, **Adriatic 2022** | INGV FDSN event service with arrivals (proven at Norcia); INGV-OV and INGV-OE bulletins | INGV EIDA node (IV), open | none nearby; MN (MedNet) is INGV, not USGS |
| **La Palma 2021** | IGN catalogue (`https://www.ign.es/web/ign/portal/sis-catalogo-terremotos`; FDSN event service to test) | IGN (ES network) open | none nearby |

GSN distances computed from the FDSN station book coordinates:

| Station | Sequence | Distance | |
|---|---|---|---|
| IU.BORG (Borgarnes, Iceland) | Reykjanes (Sundhnúkur) | 1.0° | 110 km |
| IU.MAJO (Matsushiro, Japan) | Noto 2024 | 1.2° | 134 km |
| IU.TATO (Taipei, Taiwan) | Hualien 2024 | 1.2° | 129 km |
| IU.ANTO (Ankara, Türkiye) | Kahramanmaraş 2023 (Mw7.8 epicentre) | 4.2° | 470 km |
| IU.ANTO (Ankara, Türkiye) | Samos 2020 | 5.1° | 564 km |

What follows from this. The Japanese cases need no negotiation: the JMA
arrival-time files are public, and HinetPy turns them and the Hi-net
waveforms into a script. Kahramanmaraş has three independent pick sets
(AFAD manual, KOERI, and the automatic ESSD set), which is more than any
other case; the ISC copy of AFAD's readings is the bulk route. Hualien and
Petrinja need an account or an email. Reykjanes is the one where the
operator's own archive is in transition, and where the GSN station is
close enough that the NEIC's picks on BORG give an independent reference
for the M ≥ 4.5 events of the 2023 dike, which were dozens in eight hours.
The USGS stations never replace the operator's picks below M 4.5, which is
where a picker is judged; they add an open, consistently picked reference
for the largest events of each sequence and a station whose waveform
character is the same across all of them.

## Published work behind each sequence

*DOIs verified against Crossref or DataCite by `scripts/heldout_references.py`; the list is `data/heldout_testset/references.csv` and appears in each panel of the atlas. One entry (Karakostas et al. 2021, Bull. Geol. Soc. Greece) has no DOI on record.*

| Sequence | Kind | Reference | DOI |
|---|---|---|---|
| Kaikoura 2016 | report | Kaiser et al. (2017), The 2016 Kaikōura, New Zealand, earthquake: preliminary seismological report, SRL 88 | [10.1785/0220170018](https://doi.org/10.1785/0220170018) |
| Kaikoura 2016 | catalogue | Lanza et al. (2019), Crustal fault connectivity of the Mw 7.8 2016 Kaikōura earthquake constrained by aftershock relocations, GRL 46 | [10.1029/2019GL082780](https://doi.org/10.1029/2019GL082780) |
| Norcia 2016 | report | Chiaraluce et al. (2017), The 2016 central Italy seismic sequence: a first look at the mainshocks, aftershocks, and source models, SRL 88 | [10.1785/0220160221](https://doi.org/10.1785/0220160221) |
| Norcia 2016 | catalogue | Michele et al. (2020), Fine-scale structure of the 2016–2017 central Italy seismic sequence from data recorded at the Italian National Network, JGR Solid Earth 125 | [10.1029/2019JB018440](https://doi.org/10.1029/2019JB018440) |
| Norcia 2016 | catalogue | Michele et al. (2020), Multi-segment rupture of the 2016 Amatrice-Visso-Norcia seismic sequence (central Italy) constrained by the first high-quality catalog of early aftershocks, Sci. Rep. 10 | [10.1038/s41598-019-43393-2](https://doi.org/10.1038/s41598-019-43393-2) |
| Thessaly 2021 | report | Karakostas et al. (2021), The March 2021 Tyrnavos, central Greece, doublet (Mw 6.3 and 6.0): aftershock relocation, faulting details, Coulomb stress evolution and seismic hazard implications, Bull. Geol. Soc. Greece 58 | — |
| Thessaly 2021 | catalogue | Kassaras et al. (2022), Seismotectonic analysis of the 2021 Damasi-Tyrnavos (Thessaly, central Greece) earthquake sequence and implications on the stress field rotations, J. Geodyn. 150 | [10.1016/j.jog.2022.101898](https://doi.org/10.1016/j.jog.2022.101898) |
| Thessaly 2021 | catalogue | Kassaras et al. (2021), The March 2021 Damasi earthquake sequence, central Greece: reactivation evidence across the westward propagating Tyrnavos graben, Geosciences 11 | [10.3390/geosciences11080328](https://doi.org/10.3390/geosciences11080328) |
| Kahramanmaras 2023 | catalogue | Long-term aftershock properties of the catastrophic 6 February 2023 Kahramanmaraş (Türkiye) earthquake sequence, Acta Geophysica (2024) | [10.1007/s11600-024-01419-y](https://doi.org/10.1007/s11600-024-01419-y) |
| Kahramanmaras 2023 | dataset | Lomax (2023), Precise NLL-SSST-coherence hypocenter catalog for the 2023 Mw 7.8 and Mw 7.6 SE Turkey earthquake sequence, Zenodo | [10.5281/zenodo.7727678](https://doi.org/10.5281/zenodo.7727678) |
| Kahramanmaras 2023 | dataset | Colavitti et al. (2025), A high-quality data set for seismological studies in the East Anatolian Fault Zone, Türkiye, ESSD 17 | [10.5194/essd-17-3089-2025](https://doi.org/10.5194/essd-17-3089-2025) |
| Kahramanmaras 2023 | catalogue | Earthquake monitoring using deep learning with a case study of the Kahramanmaras Turkey earthquake aftershock sequence, Solid Earth 15 (2024) | [10.5194/se-15-197-2024](https://doi.org/10.5194/se-15-197-2024) |
| Kahramanmaras 2023 | catalogue | High-resolution seismicity imaging and early aftershock migration of the 2023 Kahramanmaraş (SE Türkiye) MW7.9 & 7.8 earthquake doublet, Earthq. Sci. (2023) | [10.1016/j.eqs.2023.06.002](https://doi.org/10.1016/j.eqs.2023.06.002) |
| Noto 2024 | catalogue | Yoshida et al. (2024), Role of a hidden fault in the early process of the 2024 Mw7.5 Noto Peninsula earthquake, GRL 51 | [10.1029/2024GL110993](https://doi.org/10.1029/2024GL110993) |
| Noto 2024 | catalogue | Aftershock distribution of the 2024 Noto Peninsula earthquake, Japan, determined using a 3D velocity structure and uncertainty quantification, EPS 77 (2025) | [10.1186/s40623-025-02227-4](https://doi.org/10.1186/s40623-025-02227-4) |
| Noto 2024 | catalogue | Takahashi et al. (2026), High-precision aftershock distribution highlights the complex fault geometry of the 2024 Mw 7.5 Noto Peninsula earthquake, GRL | [10.1029/2025GL118413](https://doi.org/10.1029/2025GL118413) |
| Hualien 2024 | catalogue | Xu et al. (2025), Unzipping of the conjugate fault system during the 2024 Mw7.4 Hualien earthquake, GRL 52 | [10.1029/2025GL115218](https://doi.org/10.1029/2025GL115218) |
| Hualien 2024 | catalogue | Yang et al. (2025), An ML-enhanced earthquake catalog for the 2024 MW 7.4 Hualien earthquake sequence, JGR Solid Earth | [10.1029/2025JB032792](https://doi.org/10.1029/2025JB032792) |
| Hualien 2024 | catalogue | Complex rupture of the 2 April 2024 MW 7.4 Hualien earthquake inferred from seismic and geodetic observations, Tectonophysics (2026) | [10.1016/j.tecto.2026.231141](https://doi.org/10.1016/j.tecto.2026.231141) |
| Petrinja 2020 | catalogue | Properties of the Petrinja (Croatia) earthquake sequence of 2020–2021: results of seismological research for the first six months of activity, Tectonophysics (2023) | [10.1016/j.tecto.2023.229885](https://doi.org/10.1016/j.tecto.2023.229885) |
| Petrinja 2020 | catalogue | Spatiotemporal properties of the 2020–2021 Petrinja (Croatia) earthquake sequence, J. Seismol. (2024) | [10.1007/s10950-024-10228-1](https://doi.org/10.1007/s10950-024-10228-1) |
| Petrinja 2020 | report | Environmental effects and seismogenic source characterization of the December 2020 earthquake sequence near Petrinja, Croatia, GJI (2022) | [10.1093/gji/ggac123](https://doi.org/10.1093/gji/ggac123) |
| Samos 2020 | catalogue | Seismotectonic implications of the 2020 Samos, Greece, Mw 7.0 mainshock based on high-resolution aftershock relocation and source slip model, Acta Geophysica (2021) | [10.1007/s11600-021-00580-y](https://doi.org/10.1007/s11600-021-00580-y) |
| Samos 2020 | catalogue | The 30 October 2020, MW = 7.0, Samos earthquake: aftershock relocation, slip model, Coulomb stress evolution and estimation of shaking, Bull. Earthq. Eng. (2021) | [10.1007/s10518-021-01260-4](https://doi.org/10.1007/s10518-021-01260-4) |
| Adriatic 2022 | catalogue | A new catalogue and insights into the 2022 Adriatic offshore seismic sequence using a machine learning-based procedure, Sensors (2024/2025) | [10.3390/s25010082](https://doi.org/10.3390/s25010082) |
| Reykjanes 2023 dike | report | Sigmundsson et al. (2024), Fracturing and tectonic stress drive ultrarapid magma flow into dikes, Science 383 | [10.1126/science.adn2838](https://doi.org/10.1126/science.adn2838) |
| Reykjanes 2023 dike | report | 2023–2024 inflation-deflation cycles at Svartsengi and repeated dike injections and eruptions at the Sundhnúkur crater row, Reykjanes Peninsula, Iceland, EPSL (2025) | [10.1016/j.epsl.2025.119324](https://doi.org/10.1016/j.epsl.2025.119324) |
| Reykjanes 2023 dike | report | Pascale et al. (2024), On the move: 2023 observations on real time graben formation, Grindavík, Iceland, GRL 51 | [10.1029/2024GL110150](https://doi.org/10.1029/2024GL110150) |
| Fagradalsfjall 2021 | catalogue | Fischer et al. (2022), Swarm seismicity illuminates stress transfer prior to the 2021 Fagradalsfjall eruption in Iceland, EPSL 594 | [10.1016/j.epsl.2022.117685](https://doi.org/10.1016/j.epsl.2022.117685) |
| Fagradalsfjall 2021 | catalogue | Pre-existing structures control the orientation of strike-slip faulting during the 2021 dike intrusion at Fagradalsfjall, Iceland, JGR Solid Earth (2024) | [10.1029/2024JB030162](https://doi.org/10.1029/2024JB030162) |
| Fagradalsfjall 2021 | report | Deformation, seismicity, and monitoring response preceding and during the 2022 Fagradalsfjall eruption, Iceland, Bull. Volcanol. 85 (2023) | [10.1007/s00445-023-01671-y](https://doi.org/10.1007/s00445-023-01671-y) |
| La Palma 2021 | catalogue | D'Auria et al. (2022), Rapid magma ascent beneath La Palma revealed by seismic tomography, Sci. Rep. 12 | [10.1038/s41598-022-21818-9](https://doi.org/10.1038/s41598-022-21818-9) |
| La Palma 2021 | catalogue | Unveiling the pre-eruptive seismic series of the La Palma 2021 eruption: insights through a fully automated analysis, JVGR (2023) | [10.1016/j.jvolgeores.2023.107946](https://doi.org/10.1016/j.jvolgeores.2023.107946) |
| Santorini-Amorgos 2025 | catalogue | The 2024–2025 seismic sequence in the Santorini-Amorgos region: insights into volcano-tectonic activity through high-resolution seismic monitoring, Seismica (2025) | [10.26443/seismica.v4i1.1663](https://doi.org/10.26443/seismica.v4i1.1663) |
| Santorini-Amorgos 2025 | report | The 2025 Santorini unrest unveiled: rebounding magmatic dike intrusion with triggered seismicity, Science (2025) | [10.1126/science.adz8538](https://doi.org/10.1126/science.adz8538) |
| Etna 2022-2024 | catalogue | A new view of seismicity under Mt. Etna volcano, Italy, 2014-2023 from multi-scale high-precision earthquake relocations, Annals of Geophysics (2024) | [10.4401/ag-9147](https://doi.org/10.4401/ag-9147) |
| Etna 2022-2024 | catalogue | Moment magnitude for earthquakes in the Etna volcano area, GJI 234 (2023) | [10.1093/gji/ggad257](https://doi.org/10.1093/gji/ggad257) |
| West Bohemia 2018 | dataset | Eulenfeld (2020), Seismological dataset for 2018 West Bohemia earthquake swarm, Zenodo | [10.5281/zenodo.5016845](https://doi.org/10.5281/zenodo.5016845) |
| West Bohemia 2018 | catalogue | From earthquake swarm to a main shock–aftershocks: the 2018 activity in West Bohemia/Vogtland, GJI 224 (2021) | [10.1093/gji/ggaa523](https://doi.org/10.1093/gji/ggaa523) |
| West Bohemia 2018 | dataset | Moment tensor catalogue of earthquakes in West Bohemia from 2008 to 2018, ESSD 14 (2022) | [10.5194/essd-14-2179-2022](https://doi.org/10.5194/essd-14-2179-2022) |
| Maurienne 2017-2019 | catalogue | Minetto et al. (2022), Analysis of the spatiotemporal evolution of the Maurienne swarm (French Alps) based on earthquake clustering, Earth and Space Science 9 | [10.1029/2021EA002097](https://doi.org/10.1029/2021EA002097) |
| Maurienne 2017-2019 | report | Unprecedented seismic swarm in the Maurienne valley (2017–2019) observed by the SISmalp Alpine seismic network: operational monitoring and management, Comptes Rendus Géoscience (2021) | [10.5802/crgeos.70](https://doi.org/10.5802/crgeos.70) |
| Noto swarm 2023 | catalogue | Yoshida et al. (2023), Updip fluid flow in the crust of the northeastern Noto Peninsula, Japan, triggered the 2023 Mw 6.2 Suzu earthquake during swarm activity, GRL 50 | [10.1029/2023GL106023](https://doi.org/10.1029/2023GL106023) |
| Noto swarm 2023 | report | Shelly (2024), Examining the connections between earthquake swarms, crustal fluids, and large earthquakes in the context of the 2020–2024 Noto Peninsula, Japan, earthquake sequence, GRL 51 | [10.1029/2023GL107897](https://doi.org/10.1029/2023GL107897) |
| Noto swarm 2023 | catalogue | The role of fluids in earthquake swarms in northeastern Noto Peninsula, central Japan: insights from source mechanisms, EPS 76 (2024) | [10.1186/s40623-024-02099-0](https://doi.org/10.1186/s40623-024-02099-0) |
| Noto swarm 2023 | report | Rupture of solidified ancient magma that impeded preceding swarm migrations led to the 2024 Noto earthquake, Sci. Adv. (2025) | [10.1126/sciadv.adv5938](https://doi.org/10.1126/sciadv.adv5938) |
| Campi Flegrei 2023 | catalogue | 3D structure and dynamics of Campi Flegrei enhance multi-hazard assessment, Nat. Commun. 16 (2025) | [10.1038/s41467-025-59821-z](https://doi.org/10.1038/s41467-025-59821-z) |
| Campi Flegrei 2023 | catalogue | Causal processes of shallow and deep seismicity at Campi Flegrei caldera, Commun. Earth Environ. 6 (2025) | [10.1038/s43247-025-02045-2](https://doi.org/10.1038/s43247-025-02045-2) |
| Campi Flegrei 2023 | report | Seismic risk mitigation at Campi Flegrei in volcanic unrest, Nat. Commun. 15 (2024) | [10.1038/s41467-024-55023-1](https://doi.org/10.1038/s41467-024-55023-1) |
| Corinth-Thiva 2020-2021 | catalogue | The 2020 Perachora peninsula earthquake sequence (East Corinth Rift, Greece): spatiotemporal evolution and implications for the triggering mechanism, Acta Geophysica (2022) | [10.1007/s11600-022-00864-x](https://doi.org/10.1007/s11600-022-00864-x) |
| Corinth-Thiva 2020-2021 | catalogue | Cluster analysis of seismicity in the eastern Gulf of Corinth based on a waveform template matching catalog, Sensors 23 (2023) | [10.3390/s23062923](https://doi.org/10.3390/s23062923) |
| Corinth-Thiva 2020-2021 | catalogue | Physical and statistical pattern of the Thiva (Greece) 2020–2022 seismic swarm, Entropy 27 (2025) | [10.3390/e27090979](https://doi.org/10.3390/e27090979) |

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
- Training-set coverage: [VCSEIS / volpick](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024GL108438); [PNW 2002–2022](https://seismica.library.mcgill.ca/article/view/368); [CWA 2011–2021](https://pubs.geoscienceworld.org/ssa/srl/article/96/3/2079/650394/The-CWA-Benchmark-A-Seismic-Dataset-from-Taiwan); [CREW](https://seismica.library.mcgill.ca/article/view/1049)
