#!/usr/bin/env python3
"""
heldout_references.py

The published work behind each held-out test sequence: the seismological
reports, relocated catalogues and data sets whose picks or events the
reference rests on. Each entry carries the citation as written and a DOI;
`resolve()` checks every DOI against Crossref or DataCite (title match) and
looks up the ones left blank, so the committed CSV holds only verified
identifiers. Entries that cannot be verified are kept with doi="" and
status="unverified" rather than dropped.

    python scripts/heldout_references.py          # writes data/heldout_testset/references.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "heldout_testset" / "references.csv"
MAILTO = "mdenolle@uw.edu"

# key -> list of (kind, citation, doi-or-None, title-for-lookup)
REFS = {
    "kaikoura_2016": [
        ("report", "Kaiser et al. (2017), The 2016 Kaikōura, New Zealand, earthquake: preliminary seismological report, SRL 88", "10.1785/0220170018", "The 2016 Kaikoura New Zealand earthquake preliminary seismological report"),
        ("catalogue", "Lanza et al. (2019), Crustal fault connectivity of the Mw 7.8 2016 Kaikōura earthquake constrained by aftershock relocations, GRL 46", "10.1029/2019GL082780", "Crustal fault connectivity of the Mw 7.8 2016 Kaikoura earthquake constrained by aftershock relocations"),
    ],
    "norcia_2016": [
        ("report", "Chiaraluce et al. (2017), The 2016 central Italy seismic sequence: a first look at the mainshocks, aftershocks, and source models, SRL 88", "10.1785/0220160221", "The 2016 Central Italy seismic sequence a first look at the mainshocks aftershocks and source models"),
        ("catalogue", "Michele et al. (2020), Fine-scale structure of the 2016–2017 central Italy seismic sequence from data recorded at the Italian National Network, JGR Solid Earth 125", "10.1029/2019JB018440", "Fine-scale structure of the 2016-2017 Central Italy seismic sequence from data recorded at the Italian National Network"),
        ("catalogue", "Michele et al. (2020), Multi-segment rupture of the 2016 Amatrice-Visso-Norcia seismic sequence (central Italy) constrained by the first high-quality catalog of early aftershocks, Sci. Rep. 10", "10.1038/s41598-019-43393-2", "Multi-segment rupture of the 2016 Amatrice-Visso-Norcia seismic sequence central Italy constrained by the first high-quality catalog of Early Aftershocks"),
    ],
    "thessaly_2021": [
        ("report", "Karakostas et al. (2021), The March 2021 Tyrnavos, central Greece, doublet (Mw 6.3 and 6.0): aftershock relocation, faulting details, Coulomb stress evolution and seismic hazard implications, Bull. Geol. Soc. Greece 58", None, "The March 2021 Tyrnavos central Greece doublet aftershock relocation faulting details Coulomb stress evolution and seismic hazard implications"),
        ("catalogue", "Kassaras et al. (2022), Seismotectonic analysis of the 2021 Damasi-Tyrnavos (Thessaly, central Greece) earthquake sequence and implications on the stress field rotations, J. Geodyn. 150", "10.1016/j.jog.2022.101898", "Seismotectonic analysis of the 2021 Damasi-Tyrnavos Thessaly Central Greece earthquake sequence and implications on the stress field rotations"),
        ("catalogue", "Kassaras et al. (2021), The March 2021 Damasi earthquake sequence, central Greece: reactivation evidence across the westward propagating Tyrnavos graben, Geosciences 11", "10.3390/geosciences11080328", "The March 2021 Damasi earthquake sequence central Greece reactivation evidence across the westward propagating Tyrnavos Graben"),
    ],
    "kahramanmaras_2023": [
        ("catalogue", "Long-term aftershock properties of the catastrophic 6 February 2023 Kahramanmaraş (Türkiye) earthquake sequence, Acta Geophysica (2024)", "10.1007/s11600-024-01419-y", "Long-term aftershock properties of the catastrophic 6 February 2023 Kahramanmaras Turkiye earthquake sequence"),
        ("dataset", "Lomax (2023), Precise NLL-SSST-coherence hypocenter catalog for the 2023 Mw 7.8 and Mw 7.6 SE Turkey earthquake sequence, Zenodo", "10.5281/zenodo.7727678", None),
        ("dataset", "Colavitti et al. (2025), A high-quality data set for seismological studies in the East Anatolian Fault Zone, Türkiye, ESSD 17", "10.5194/essd-17-3089-2025", "A high-quality data set for seismological studies in the East Anatolian Fault Zone Turkiye"),
        ("catalogue", "Earthquake monitoring using deep learning with a case study of the Kahramanmaras Turkey earthquake aftershock sequence, Solid Earth 15 (2024)", "10.5194/se-15-197-2024", "Earthquake monitoring using deep learning with a case study of the Kahramanmaras Turkey earthquake aftershock sequence"),
        ("catalogue", "High-resolution seismicity imaging and early aftershock migration of the 2023 Kahramanmaraş (SE Türkiye) MW7.9 & 7.8 earthquake doublet, Earthq. Sci. (2023)", "10.1016/j.eqs.2023.06.002", "High-resolution seismicity imaging and early aftershock migration of the 2023 Kahramanmaras SE Turkiye earthquake doublet"),
    ],
    "noto_2024": [
        ("catalogue", "Yoshida et al. (2024), Role of a hidden fault in the early process of the 2024 Mw7.5 Noto Peninsula earthquake, GRL 51", "10.1029/2024GL110993", "Role of a hidden fault in the early process of the 2024 Mw7.5 Noto Peninsula earthquake"),
        ("catalogue", "Aftershock distribution of the 2024 Noto Peninsula earthquake, Japan, determined using a 3D velocity structure and uncertainty quantification, EPS 77 (2025)", "10.1186/s40623-025-02227-4", "Aftershock distribution of the 2024 Noto Peninsula Earthquake Japan determined using a 3D velocity structure and uncertainty quantification"),
        ("catalogue", "Takahashi et al. (2026), High-precision aftershock distribution highlights the complex fault geometry of the 2024 Mw 7.5 Noto Peninsula earthquake, GRL", "10.1029/2025GL118413", "High-precision aftershock distribution highlights the complex fault geometry of the 2024 Mw 7.5 Noto Peninsula earthquake"),
    ],
    "hualien_2024": [
        ("catalogue", "Xu et al. (2025), Unzipping of the conjugate fault system during the 2024 Mw7.4 Hualien earthquake, GRL 52", "10.1029/2025GL115218", "Unzipping of the conjugate fault system during the 2024 Mw7.4 Hualien earthquake"),
        ("catalogue", "Yang et al. (2025), An ML-enhanced earthquake catalog for the 2024 MW 7.4 Hualien earthquake sequence, JGR Solid Earth", "10.1029/2025JB032792", "An ML-enhanced earthquake catalog for the 2024 MW 7.4 Hualien earthquake sequence insights into structural transition from collision to subduction in eastern Taiwan"),
        ("catalogue", "Complex rupture of the 2 April 2024 MW 7.4 Hualien earthquake inferred from seismic and geodetic observations, Tectonophysics (2026)", None, "Complex rupture of the 2 April 2024 MW 7.4 Hualien earthquake inferred from seismic and geodetic observations"),
    ],
    "petrinja_2020": [
        ("catalogue", "Properties of the Petrinja (Croatia) earthquake sequence of 2020–2021: results of seismological research for the first six months of activity, Tectonophysics (2023)", None, "Properties of the Petrinja Croatia earthquake sequence of 2020-2021 Results of seismological research for the first six months of activity"),
        ("catalogue", "Spatiotemporal properties of the 2020–2021 Petrinja (Croatia) earthquake sequence, J. Seismol. (2024)", "10.1007/s10950-024-10228-1", "Spatiotemporal properties of the 2020-2021 Petrinja Croatia earthquake sequence"),
        ("report", "Environmental effects and seismogenic source characterization of the December 2020 earthquake sequence near Petrinja, Croatia, GJI (2022)", "10.1093/gji/ggac123", "Environmental effects and seismogenic source characterization of the December 2020 earthquake sequence near Petrinja Croatia"),
    ],
    "samos_2020": [
        ("catalogue", "Seismotectonic implications of the 2020 Samos, Greece, Mw 7.0 mainshock based on high-resolution aftershock relocation and source slip model, Acta Geophysica (2021)", "10.1007/s11600-021-00580-y", "Seismotectonic implications of the 2020 Samos Greece Mw 7.0 mainshock based on high-resolution aftershock relocation and source slip model"),
        ("catalogue", "The 30 October 2020, MW = 7.0, Samos earthquake: aftershock relocation, slip model, Coulomb stress evolution and estimation of shaking, Bull. Earthq. Eng. (2021)", "10.1007/s10518-021-01260-4", "The 30 October 2020 MW 7.0 Samos earthquake aftershock relocation slip model Coulomb stress evolution and estimation of shaking"),
    ],
    "adriatic_2022": [
        ("catalogue", "A new catalogue and insights into the 2022 Adriatic offshore seismic sequence using a machine learning-based procedure, Sensors (2024/2025)", None, "A New Catalogue and Insights into the 2022 Adriatic Offshore Seismic Sequence Using a Machine Learning-Based Procedure"),
    ],
    "reykjanes_2023": [
        ("report", "Sigmundsson et al. (2024), Fracturing and tectonic stress drive ultrarapid magma flow into dikes, Science 383", "10.1126/science.adn2838", "Fracturing and tectonic stress drive ultrarapid magma flow into dikes"),
        ("report", "2023–2024 inflation-deflation cycles at Svartsengi and repeated dike injections and eruptions at the Sundhnúkur crater row, Reykjanes Peninsula, Iceland, EPSL (2025)", None, "2023-2024 inflation-deflation cycles at Svartsengi and repeated dike injections and eruptions at the Sundhnukur crater row Reykjanes Peninsula Iceland"),
        ("report", "Pascale et al. (2024), On the move: 2023 observations on real time graben formation, Grindavík, Iceland, GRL 51", "10.1029/2024GL110150", "On the Move 2023 observations on real time graben formation Grindavik Iceland"),
    ],
    "fagradalsfjall_2021": [
        ("catalogue", "Fischer et al. (2022), Swarm seismicity illuminates stress transfer prior to the 2021 Fagradalsfjall eruption in Iceland, EPSL 594", None, "Swarm seismicity illuminates stress transfer prior to the 2021 Fagradalsfjall eruption in Iceland"),
        ("catalogue", "Pre-existing structures control the orientation of strike-slip faulting during the 2021 dike intrusion at Fagradalsfjall, Iceland, JGR Solid Earth (2024)", "10.1029/2024JB030162", "Pre-existing structures control the orientation of strike-slip faulting during the 2021 dike intrusion at Fagradalsfjall Iceland"),
        ("report", "Deformation, seismicity, and monitoring response preceding and during the 2022 Fagradalsfjall eruption, Iceland, Bull. Volcanol. 85 (2023)", "10.1007/s00445-023-01671-y", "Deformation seismicity and monitoring response preceding and during the 2022 Fagradalsfjall eruption Iceland"),
    ],
    "la_palma_2021": [
        ("catalogue", "D'Auria et al. (2022), Rapid magma ascent beneath La Palma revealed by seismic tomography, Sci. Rep. 12", None, "Rapid magma ascent beneath La Palma revealed by seismic tomography"),
        ("catalogue", "Unveiling the pre-eruptive seismic series of the La Palma 2021 eruption: insights through a fully automated analysis, JVGR (2023)", None, "Unveiling the pre-eruptive seismic series of the La Palma 2021 eruption Insights through a fully automated analysis"),
    ],
    "santorini_2025": [
        ("catalogue", "The 2024–2025 seismic sequence in the Santorini-Amorgos region: insights into volcano-tectonic activity through high-resolution seismic monitoring, Seismica (2025)", None, "The 2024-2025 seismic sequence in the Santorini-Amorgos region insights into volcano-tectonic activity through high-resolution seismic monitoring"),
        ("report", "The 2025 Santorini unrest unveiled: rebounding magmatic dike intrusion with triggered seismicity, Science (2025)", "10.1126/science.adz8538", "The 2025 Santorini unrest unveiled rebounding magmatic dike intrusion with triggered seismicity"),
    ],
    "etna_2022_2024": [
        ("catalogue", "A new view of seismicity under Mt. Etna volcano, Italy, 2014-2023 from multi-scale high-precision earthquake relocations, Annals of Geophysics (2024)", None, "A new view of seismicity under Mt Etna volcano Italy 2014-2023 from multi-scale high-precision earthquake relocations"),
        ("catalogue", "Moment magnitude for earthquakes in the Etna volcano area, GJI 234 (2023)", "10.1093/gji/ggad257", "Moment magnitude for earthquakes in the Etna volcano area"),
    ],
    "west_bohemia_2018": [
        ("dataset", "Eulenfeld (2020), Seismological dataset for 2018 West Bohemia earthquake swarm, Zenodo", "10.5281/zenodo.5016845", None),
        ("catalogue", "From earthquake swarm to a main shock–aftershocks: the 2018 activity in West Bohemia/Vogtland, GJI 224 (2021)", None, "From earthquake swarm to a main shock-aftershocks the 2018 activity in West Bohemia Vogtland"),
        ("dataset", "Moment tensor catalogue of earthquakes in West Bohemia from 2008 to 2018, ESSD 14 (2022)", "10.5194/essd-14-2179-2022", "Moment tensor catalogue of earthquakes in West Bohemia from 2008 to 2018"),
    ],
    "maurienne_2017": [
        ("catalogue", "Minetto et al. (2022), Analysis of the spatiotemporal evolution of the Maurienne swarm (French Alps) based on earthquake clustering, Earth and Space Science 9", "10.1029/2021EA002097", "Analysis of the spatiotemporal evolution of the Maurienne swarm French Alps based on earthquake clustering"),
        ("report", "Unprecedented seismic swarm in the Maurienne valley (2017–2019) observed by the SISmalp Alpine seismic network: operational monitoring and management, Comptes Rendus Géoscience (2021)", None, "Unprecedented seismic swarm in the Maurienne valley 2017-2019 observed by the SISmalp Alpine seismic network operational monitoring and management"),
    ],
    "noto_swarm_2023": [
        ("catalogue", "Yoshida et al. (2023), Updip fluid flow in the crust of the northeastern Noto Peninsula, Japan, triggered the 2023 Mw 6.2 Suzu earthquake during swarm activity, GRL 50", "10.1029/2023GL106023", "Updip fluid flow in the crust of the northeastern Noto Peninsula Japan triggered the 2023 Mw 6.2 Suzu earthquake during swarm activity"),
        ("report", "Shelly (2024), Examining the connections between earthquake swarms, crustal fluids, and large earthquakes in the context of the 2020–2024 Noto Peninsula, Japan, earthquake sequence, GRL 51", "10.1029/2023GL107897", "Examining the connections between earthquake swarms crustal fluids and large earthquakes in the context of the 2020-2024 Noto Peninsula Japan earthquake sequence"),
        ("catalogue", "The role of fluids in earthquake swarms in northeastern Noto Peninsula, central Japan: insights from source mechanisms, EPS 76 (2024)", "10.1186/s40623-024-02099-0", "The role of fluids in earthquake swarms in northeastern Noto Peninsula central Japan insights from source mechanisms"),
        ("report", "Rupture of solidified ancient magma that impeded preceding swarm migrations led to the 2024 Noto earthquake, Sci. Adv. (2025)", "10.1126/sciadv.adv5938", "Rupture of solidified ancient magma that impeded preceding swarm migrations led to the 2024 Noto earthquake"),
    ],
    "campi_flegrei_2023": [
        ("catalogue", "3D structure and dynamics of Campi Flegrei enhance multi-hazard assessment, Nat. Commun. 16 (2025)", "10.1038/s41467-025-59821-z", "3D structure and dynamics of Campi Flegrei enhance multi-hazard assessment"),
        ("catalogue", "Causal processes of shallow and deep seismicity at Campi Flegrei caldera, Commun. Earth Environ. 6 (2025)", "10.1038/s43247-025-02045-2", "Causal processes of shallow and deep seismicity at Campi Flegrei caldera"),
        ("report", "Seismic risk mitigation at Campi Flegrei in volcanic unrest, Nat. Commun. 15 (2024)", "10.1038/s41467-024-55023-1", "Seismic risk mitigation at Campi Flegrei in volcanic unrest"),
    ],
    "corinth_thiva_2020": [
        ("catalogue", "The 2020 Perachora peninsula earthquake sequence (East Corinth Rift, Greece): spatiotemporal evolution and implications for the triggering mechanism, Acta Geophysica (2022)", "10.1007/s11600-022-00864-x", "The 2020 Perachora peninsula earthquake sequence East Corinth Rift Greece spatiotemporal evolution and implications for the triggering mechanism"),
        ("catalogue", "Cluster analysis of seismicity in the eastern Gulf of Corinth based on a waveform template matching catalog, Sensors 23 (2023)", None, "Cluster analysis of seismicity in the eastern Gulf of Corinth based on a waveform template matching catalog"),
        ("catalogue", "Physical and statistical pattern of the Thiva (Greece) 2020–2022 seismic swarm, Entropy 27 (2025)", "10.3390/e27090979", "Physical and statistical pattern of the Thiva Greece 2020-2022 seismic swarm"),
    ],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def _similar(a: str, b: str) -> float:
    A, B = set(_norm(a)), set(_norm(b))
    return len(A & B) / max(1, len(A | B))


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": f"phasenet-retrain/heldout_references (mailto:{MAILTO})"})
    return json.load(urllib.request.urlopen(req, timeout=60))


PREPRINT_PREFIXES = ("10.5194/egusphere", "10.21203/", "10.1002/essoar", "10.31223/", "10.48550/arxiv")


def crossref_lookup(title: str, min_sim: float = 0.6):
    """First journal article (not an abstract or preprint) whose title matches."""
    q = urllib.parse.quote(title)
    d = _get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=8&mailto={MAILTO}")
    for it in d["message"]["items"]:
        doi = it["DOI"].lower()
        if it.get("type") not in ("journal-article", "dataset") or doi.startswith(PREPRINT_PREFIXES):
            continue
        t = " ".join(it.get("title", []))
        if _similar(t, title) >= min_sim:
            return it["DOI"], t
    return None, None


def crossref_title(doi: str):
    try:
        d = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}")
        return " ".join(d["message"].get("title", []))
    except Exception:  # noqa: BLE001
        return None


def datacite_title(doi: str):
    try:
        d = _get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
        return " ".join(t.get("title", "") for t in d["data"]["attributes"].get("titles", []))
    except Exception:  # noqa: BLE001
        return None


def resolve() -> list:
    rows = []
    for key, refs in REFS.items():
        for kind, citation, doi, title in refs:
            status, resolved_title = "unverified", ""
            if doi:
                t = crossref_title(doi) or datacite_title(doi)
                if t and (title is None or _similar(t, title) >= 0.5):
                    resolved_title, status = t, "verified"
                else:
                    print(f"    {key}: DOI {doi} resolves to {t!r}; looking the title up instead")
                    doi = None
            if not doi and title:
                found, t = crossref_lookup(title)
                if found:
                    doi, resolved_title, status = found, t, "verified"
            rows.append(dict(key=key, kind=kind, citation=citation, doi=doi or "", url=(f"https://doi.org/{doi}" if doi else ""),
                             resolved_title=resolved_title, status=status))
            print(f"{key:20s} {status:18s} {doi or '-':32s} {citation[:60]}", flush=True)
            time.sleep(0.3)
    return rows


def main():
    rows = resolve()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "kind", "citation", "doi", "url", "resolved_title", "status"], lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    n_ok = sum(r["status"] == "verified" for r in rows)
    print(f"\n{len(rows)} references, {n_ok} verified -> {OUT}")


if __name__ == "__main__":
    main()
