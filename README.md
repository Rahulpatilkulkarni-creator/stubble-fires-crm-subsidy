# 🔥 Stubble Fires vs. the Happy Seeder Subsidy

**Did India's crop-residue-management equipment subsidy actually reduce farm fires in Punjab & Haryana — and can we predict where fires will spike next season?**

Every October–November, farmers across Punjab and Haryana burn paddy stubble to clear fields for the next crop. The smoke drives Delhi's annual air-quality emergency. Governments responded with subsidies for machines — most famously the **Happy Seeder** — that let farmers sow without burning. This project asks two questions with real data:

1. **Causal — did the subsidy work?** Districts took up the Crop Residue Management (CRM) subsidy at different times and intensities. That staggered rollout is a natural quasi-experiment. Using a **difference-in-differences** design, did higher-uptake districts see fire counts fall *relative to* lower-uptake districts, after controlling for weather and crop timing?

2. **Predictive — where next?** Using historical fire locations, weather (wind, humidity), and crop-calendar timing, build a **spatiotemporal early-warning model** for next season's hotspots — useful for targeting enforcement and machine allocation.

## Why this is different

- **Raw satellite data, not a pre-cleaned CSV.** The fire signal comes from **NASA FIRMS** thermal-anomaly detections (MODIS + VIIRS) — the same feeds that drive real air-quality reporting.
- **A real policy-evaluation question** environmental economists actually study — not a toy correlation.
- **It connects to something millions live through:** Delhi's November smog.

## Data sources

| Layer | Source | Access |
|---|---|---|
| Fire detections (outcome) | NASA FIRMS — MODIS C6.1 + VIIRS (SNPP, NOAA-20) | Keyless regional feeds (current) + Area API for history (free `MAP_KEY`) |
| Weather (controls/features) | Open-Meteo — ERA5 archive + forecast | Free, keyless |
| Subsidy uptake (treatment) | CRM scheme allocations / state agriculture depts | Being sourced (see status) |
| Boundaries & crop calendar | District admin boundaries; kharif paddy calendar | Public |

## Repository layout

```
src/fire_policy/     # reusable package
  config.py          # geography, time windows, data-source templates, paths
  firms.py           # NASA FIRMS fetchers (keyless feeds + keyed Area API + archive loader)
  weather.py         # Open-Meteo ERA5 archive + forecast
data/                # raw / interim / processed  (raw is gitignored)
notebooks/           # 01_data → 02_eda → 03_causal_did → 04_prediction
reports/             # findings + figures
```

## Getting started

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Pull **real, current** fire detections over Punjab & Haryana (no key needed):

```bash
.venv/Scripts/python -m fire_policy.firms
```

For **historical** pulls (needed for the DiD + model training), get a free FIRMS key
(https://firms.modaps.eosdis.nasa.gov/api/area/), copy `.env.example` → `.env`, and paste it in.

## Status & roadmap

- [x] **Phase 0 — Scaffold**: repo, package, config, data-source contracts verified against live endpoints.
- [ ] **Phase 1 — Data**: real fire pulls (current feeds now; full history on key), weather archive.
- [ ] **Phase 2 — Geo & EDA**: attach district boundaries, build the district × season fire panel, hotspot maps.
- [ ] **Phase 3 — Causal**: assemble subsidy treatment, test parallel trends, run difference-in-differences.
- [ ] **Phase 4 — Prediction**: spatiotemporal hotspot model + evaluation + early-warning maps.
- [ ] **Phase 5 — Report**: findings writeup (+ optional dashboard).

## Honest caveats

- **Historical FIRMS** needs a free `MAP_KEY` (emailed on signup) — the only external credential.
- **Subsidy uptake** at district×year granularity is the hardest input; if only state-level data
  exists, the DiD uses a clearly-documented proxy (e.g., CRM machines sanctioned per district).
- Satellite fire counts undercount on cloudy days and around overpass times — handled as a known
  measurement caveat, not silently ignored.
