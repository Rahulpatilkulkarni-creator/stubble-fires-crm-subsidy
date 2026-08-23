# 🔥 Stubble Fires vs. the Crop-Residue Subsidy

**Can we predict where farm fires will spike next season — and did India's crop-residue
equipment subsidy actually put them out, in Punjab & Haryana?**

Every October–November, farmers across Punjab and Haryana burn paddy stubble to clear
fields for wheat. The smoke drives Delhi's annual air-quality emergency. From 2018-19 the
government scaled up the **Crop-Residue-Management (CRM) scheme**, subsidising machines —
the **Happy Seeder**, Super Seeder, balers — that let farmers sow without burning. This
project answers two questions with raw data, end to end:

1. **Predictive — where next?** A **district × week** early-warning model for hotspots.
   → **R² = 0.90, Spearman = 0.96, ROC-AUC = 0.99** on a 2017–18 forward test.
2. **Causal — did the subsidy work?** A **dose-response difference-in-differences** on the
   staggered-intensity CRM rollout, controlling for weather and crop timing.

**→ Read [`reports/FINDINGS.md`](reports/FINDINGS.md) for the full write-up, results and
figures.**

## Headline findings

- **Fires are extremely predictable — but *not* from weather.** Last week's burning +
  the crop calendar + district identity carry ~97% of the model's signal; an 8-variable
  ERA5 weather block moves R² by **+0.004** (and *hurts* a no-persistence model by
  **−0.09**). Weather gates whether burning is *possible*, not *where* it concentrates.
- **The subsidy was targeted at the worst-burning districts** (within-state rank
  correlation **+0.86 Punjab / +0.69 Haryana**) — good policy, but a confound that makes a
  within-district DiD essential. Its **parallel-trends assumption holds** (joint pre-trend
  test p = 0.18) and **placebo DiDs return null**.
- **The clean causal estimate is one dataset away**: the keyless fire record ends in 2018,
  the subsidy's scale-up year. The estimator and all validity checks pass on the
  pre-period; a post-2018 fire panel (one free NASA FIRMS key) yields the effect directly.

## Why this is different

- **Raw satellite grids + primary government records, not a pre-cleaned CSV.** Fire mass
  comes from the **SAGE-IGP** daily 0.25° inventory; treatment from **PPCB MIS** and
  **Haryana CRM** action-plan PDFs.
- **A real policy-evaluation question** — with the identification problem (targeting)
  diagnosed and handled, not hidden.
- **It connects to something millions live through:** Delhi's November smog.

## Data sources

| Layer | Source | Coverage | Access |
|---|---|---|---|
| Fire (burned dry matter) | **SAGE-IGP** (Liu et al. 2020, Harvard Dataverse, CC0) | 2003–2018, 0.25° daily | keyless |
| Weather | **Open-Meteo ERA5 archive** | 2012–2018, daily→weekly | keyless |
| Treatment (CRM machines) | **PPCB MIS** (Punjab, actual) + **Haryana CRM plan** (targets) | 2018–2022, district | primary PDFs/portal |
| Geography | **geoBoundaries** ADM2 (CC-BY) | 43 districts | keyless |
| *Post-2018 fire (to finish DiD)* | *NASA FIRMS Area API / archive* | *2019+* | *free `MAP_KEY` (emailed) — see caveats* |

## Repository layout

```
src/fire_policy/
  config.py      # geography, time windows, paths, FIRMS templates
  geo.py         # geoBoundaries → Punjab/Haryana district layer, point tagging, centroids
  sage.py        # SAGE-IGP grids → district×year and district×week burned-mass panels
  weather.py     # Open-Meteo ERA5 → district×week weather (multi-location batched)
  treatment.py   # PPCB + Haryana CRM records → harmonised district dose panel
  predict.py     # district×week LightGBM early-warning model + weather ablation
  causal.py      # dose-response DiD: targeting, parallel-trends, placebos, estimator
  eda.py         # descriptive figures + interactive hotspot map
  firms.py       # NASA FIRMS fetchers (wired for the post-2018 extension)
data/            # raw / interim / processed   (raw is gitignored, re-fetchable)
reports/         # FINDINGS.md + figures/ (01–11) + hotspot_map.html
notebooks/       # 01_data → 02_eda → 03_prediction → 04_causal (narrative walk-throughs)
```

## Notebooks

Four executed notebooks narrate the project end to end — thin wrappers over the package
that read the processed panels, call the analysis functions, and embed the figures. They
ship **pre-run** (tables and plots baked in), so they render on GitHub without a kernel:

| Notebook | What it covers |
|---|---|
| [`01_data`](notebooks/01_data.ipynb) | The three panels, the harmonised CRM dose, the 2018 wall |
| [`02_eda`](notebooks/02_eda.ipynb) | Concentration of burning, timing, the weak fire–weather link |
| [`03_prediction`](notebooks/03_prediction.ipynb) | Week-ahead model, climatology baseline, weather ablation |
| [`04_causal`](notebooks/04_causal.ipynb) | Dose-response DiD: targeting, parallel trends, placebos |

Rebuild them from source with `PYTHONPATH=src .venv/Scripts/python notebooks/_build_notebooks.py`.

## Getting started

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Run the pipeline in order (each stage writes to `data/processed/` and is independently
re-runnable):

```bash
PYTHONPATH=src .venv/Scripts/python -m fire_policy.geo        # district layer
PYTHONPATH=src .venv/Scripts/python -m fire_policy.sage       # fire panels
PYTHONPATH=src .venv/Scripts/python -m fire_policy.weather    # ERA5 weekly weather
PYTHONPATH=src .venv/Scripts/python -m fire_policy.treatment  # CRM dose panel
PYTHONPATH=src .venv/Scripts/python -m fire_policy.eda        # figures 01–04
PYTHONPATH=src .venv/Scripts/python -m fire_policy.predict    # figures 05–08 + ablation
PYTHONPATH=src .venv/Scripts/python -m fire_policy.causal     # figures 09–11 + DiD checks
```

## Status

- [x] **Data** — SAGE-IGP fire panels (301 district-years, 2,838 district-weeks); ERA5
  weekly weather (4,171 district-weeks); harmonised CRM treatment (43 districts).
- [x] **Geo & EDA** — district layer, area-weighted fire allocation, hotspot map, timing.
- [x] **Prediction** — district×week model, climatology baseline, weather ablation,
  early-warning evaluation (figs 05–08).
- [x] **Causal (design + pre-period)** — treatment doses, targeting analysis,
  parallel-trends event study, placebo DiDs, ready-to-run estimator (figs 09–11).
- [ ] **Causal (effect)** — awaits a post-2018 district fire outcome (free FIRMS
  `MAP_KEY`); `estimate_did()` produces β + event study on drop-in.
- [x] **Notebooks** — four executed, figure-embedded walk-throughs (01_data → 04_causal).

## Honest caveats

- **The causal leg is a *validated design*, not yet a delivered effect estimate.** SAGE-IGP
  stops in 2018; the only external credential needed to finish it is a free FIRMS `MAP_KEY`
  (emailed on signup). Stated plainly rather than replaced with a weak proxy.
- **Treatment comparability**: Punjab reports actual cumulative machines, Haryana one-year
  targets (near-uniform across districts) — harmonised to a within-state z-score; the real
  dose variation is Punjab's.
- **Fire resolution**: 0.25° (~667 km²) is coarse for small districts; totals are
  approximate and area-weighted.
- **Weather adds little to prediction** — a finding, not a bug: the burn decision tracks
  the harvest calendar and economics, not the weather.
