# 🔥 Stubble Fires vs. the Crop-Residue Subsidy

[![Live demo](https://img.shields.io/badge/live_demo-streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://stubble-fires-crm.streamlit.app/) &nbsp;**→ [stubble-fires-crm.streamlit.app](https://stubble-fires-crm.streamlit.app/)**

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
   continuous-intensity CRM rollout. → **β = +0.04 (p = 0.58)** — no dose-response reduction;
   burning fell in aggregate, but not in proportion to the machines a district received.

**→ Read [`reports/FINDINGS.md`](reports/FINDINGS.md) for the full write-up, results and
figures.**

## Results at a glance

| Q1 — Can we see the hotspots coming? | Q2 — Did the subsidy reduce burning? |
|:---:|:---:|
| [![Predicted vs actual weekly district burning on the held-out 2017–18 seasons](reports/figures/05_pred_vs_actual.png)](reports/figures/05_pred_vs_actual.png) | [![Aggregate burning-season fire trend for Punjab and Haryana, 2012–2023](reports/figures/13_aggregate_trend.png)](reports/figures/13_aggregate_trend.png) |
| The week-ahead model tracks the held-out **2017–18** seasons closely — **R² = 0.90, ROC-AUC = 0.99**. | Burning fell after 2018 (**Haryana −44%, Punjab −18%**), but the dose-response DiD is a **null (β = +0.04, p = 0.58)** — higher-dose districts didn't fall *more*, so the design can't credit the subsidy. |

## Headline findings

- **Fires are extremely predictable — but *not* from weather.** Last week's burning +
  the crop calendar + district identity carry ~97% of the model's signal; an 8-variable
  ERA5 weather block moves R² by **+0.004** (and *hurts* a no-persistence model by
  **−0.09**). Weather gates whether burning is *possible*, not *where* it concentrates.
- **The subsidy was targeted at the worst-burning districts** (within-state rank
  correlation **+0.86 Punjab / +0.69 Haryana**, +0.88 / +0.83 on FIRMS) — sound policy, but
  a confound severe enough that even a within-district DiD can't fully clean it: pre-trends
  are parallel when pooled, yet **fail within Punjab** (joint p = 0.005), where the real dose
  variation lives.
- **No evidence the subsidy *drove* the decline.** On a consistent NASA FIRMS outcome built
  across **2012–2023**, the dose-response DiD is a precise null — **β = +0.04 (p = 0.58)**,
  with every robustness specification zero-to-positive. Aggregate burning *did* fall
  post-2018 (**Haryana −44%, Punjab −18%**), but a year-fixed-effect design can't credit the
  subsidy for a *common* trend, and the *differential* test — did more-subsidised districts
  fall *more*? — comes back empty. The honest read: *"no detectable dose-response effect, and
  the quasi-experiment can't cleanly isolate one."*

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
| Fire — active-fire counts *(causal outcome)* | **NASA FIRMS VIIRS-SNPP** archive | 2012–2023, 375 m → district×year | free `MAP_KEY` (emailed) |

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
  effect.py      # DiD finisher: consistent FIRMS 2012–23 outcome → β + event study + aggregate trend
  eda.py         # descriptive figures + interactive hotspot map
  firms.py       # NASA FIRMS fetchers (VIIRS-SNPP archive → causal outcome)
  panel.py       # standardized FIRMS points → balanced district×period fire panel
app/
  streamlit_app.py  # interactive dashboard: early-warning hotspots + causal design
data/            # raw / interim / processed   (raw is gitignored, re-fetchable)
reports/         # FINDINGS.md + figures/ (01–13) + hotspot_map.html
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
| [`04_causal`](notebooks/04_causal.ipynb) | Dose-response DiD: targeting, parallel trends, placebos, the FIRMS effect |

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

**Reproduce the causal effect.** The first run needs a free NASA FIRMS `MAP_KEY` (emailed on
signup at <https://firms.modaps.eosdis.nasa.gov/api/area/>), put in `.env` as
`FIRMS_MAP_KEY=...`; it caches the FIRMS panel, so later runs rebuild the DiD offline:

```bash
PYTHONPATH=src .venv/Scripts/python -m fire_policy.effect     # β + event study + aggregate (figs 12–13)
```

**Explore interactively** — try the **[live dashboard](https://stubble-fires-crm.streamlit.app/)**,
or run it locally (it reads the processed panels and figures):

```bash
PYTHONPATH=src .venv/Scripts/python -m streamlit run app/streamlit_app.py
```

## Status

- [x] **Data** — SAGE-IGP fire panels (301 district-years, 2,838 district-weeks); ERA5
  weekly weather (4,171 district-weeks); harmonised CRM treatment (43 districts).
- [x] **Geo & EDA** — district layer, area-weighted fire allocation, hotspot map, timing.
- [x] **Prediction** — district×week model, climatology baseline, weather ablation,
  early-warning evaluation (figs 05–08).
- [x] **Causal (design + pre-period)** — treatment doses, targeting analysis,
  parallel-trends event study, placebo DiDs (figs 09–11).
- [x] **Causal (effect)** — consistent NASA FIRMS VIIRS outcome, 2012–2023 (516
  district-years); dose-response DiD **β = +0.04 (p = 0.58)**, robustness table, dynamic
  event study, aggregate trend (figs 12–13). Result: no dose-response reduction.
- [x] **Notebooks** — four executed, figure-embedded walk-throughs (01_data → 04_causal).
- [x] **Dashboard** — Streamlit app: interactive early-warning hotspots + causal-design
  walk-through (`app/streamlit_app.py`).

## Honest caveats

- **The causal result is a *null*, and an honestly confounded one.** The dose-response DiD
  finds no reduction (β = +0.04, p = 0.58), but near-perfect targeting (ρ ≈ 0.88) and a
  Punjab pre-trend that fails (p = 0.005) mean it can't be read as a clean causal zero — only
  as "no evidence that machines-per-district cut burning." A year-fixed-effect design is also
  blind to any uniform, state-wide effect of the scheme.
- **Treatment comparability**: Punjab reports actual cumulative machines, Haryana one-year
  targets (near-uniform across districts) — harmonised to a within-state z-score; the real
  dose variation is Punjab's.
- **Fire resolution**: 0.25° (~667 km²) is coarse for small districts; totals are
  approximate and area-weighted.
- **Weather adds little to prediction** — a finding, not a bug: the burn decision tracks
  the harvest calendar and economics, not the weather.
