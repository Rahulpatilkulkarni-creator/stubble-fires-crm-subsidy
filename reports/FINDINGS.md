# Findings — Stubble Burning vs. Crop-Residue Subsidy in Punjab & Haryana

*Two questions on the fires behind Delhi's November smog: **can we predict** next
season's hotspots, and **did the equipment subsidy** actually put them out?*

Built end-to-end from raw satellite grids and primary government subsidy records —
no pre-cleaned analysis CSV. All figures referenced below live in
[`reports/figures/`](figures/).

---

## TL;DR

1. **Where and when fires happen is extremely predictable — but almost none of that
   predictability comes from weather.** A week-ahead model reaches **R² = 0.90,
   Spearman = 0.96, ROC-AUC = 0.99** for catching the worst district-weeks. It runs on
   fire *persistence* + crop-calendar + district identity; adding an 8-variable ERA5
   weather block changes R² by **+0.004**. In the structural (no-fire-lag) model weather
   actually *hurts* (**−0.09 R²**). Weather sets whether burning is *possible*; it does
   not decide *where* it concentrates.

2. **The subsidy was deliberately aimed at the worst-burning districts** — within-state
   rank correlation between CRM machines received and prior burning is **+0.86 in Punjab,
   +0.69 in Haryana**. That is good policy targeting but a textbook confound: any naive
   "treated vs untreated" comparison is biased toward finding the subsidy made things
   *worse*. A within-district difference-in-differences is required, and its
   parallel-trends assumption **holds** in the data (joint pre-trend test p = 0.18).

3. **The clean causal estimate is one dataset away.** The keyless fire record
   (SAGE-IGP) stops in 2018 — the exact year the subsidy scaled up. The estimator, the
   treatment doses, the controls and every validity check are built and passing on the
   pre-period; dropping in a post-2018 district fire panel (one free NASA FIRMS key)
   produces the effect estimate immediately.

---

## The problem

Every autumn, farmers across Punjab and Haryana have ~2–3 weeks to clear paddy residue
before sowing wheat. The cheapest way is to burn it. The resulting smoke is a major
driver of Delhi's hazardous November air. Since **2018-19** the government scaled up the
**Crop-Residue-Management (CRM) scheme**, subsidising Happy Seeders, Super Seeders,
balers and other equipment that let farmers manage residue *without* burning.

This project asks the two questions a policy analyst actually cares about:

- **Predictive** — *can we see the hotspots coming*, early enough to pre-position
  enforcement and machinery?
- **Causal** — *did the subsidy work*, and where did the money go?

## Data (and why it's honest)

| Layer | Source | Resolution | Coverage | Access |
|---|---|---|---|---|
| Fire (burned mass) | **SAGE-IGP** (Liu et al. 2020, Harvard Dataverse, CC0) | 0.25° daily → district×week | **2003–2018** | keyless |
| Weather | **Open-Meteo ERA5 archive** | point daily → district×week | 2012–2018 | keyless |
| Treatment | **PPCB MIS** (Punjab, actual machines) + **Haryana CRM action plan** (2018-19 targets) | district | 2018–2022 | primary PDFs/portal |
| Geography | **geoBoundaries** ADM2 (CC-BY) | 43 districts | — | keyless |

Fire mass is allocated from the coarse 0.25° grid to districts by **area-weighted polygon
overlay** (equal-area CRS), not centroid assignment, so small districts aren't dropped.

**The 2018 wall.** SAGE-IGP ends in 2018. That is fine for prediction (train 2012–16,
test 2017–18) and for every *pre-period* causal check, but it means the post-treatment
fire outcome for the DiD is not available through any keyless source I could reach
(the near-real-time FIRMS feeds are 7-day-only; the PPCB action-plan PDF contains only
blank reporting *templates* and state-level totals). See the last section.

Descriptively, the burning is heavily concentrated: **Punjab is 86%** of the two-state
total (8.8 vs 1.5 Mt/yr), peaking sharply in **ISO weeks 44–45** (early November), led by
**Sangrur, Bathinda, Moga, Ludhiana and Muktsar**
([`01_state_trend`](figures/01_state_trend.png),
[`02_top_districts`](figures/02_top_districts.png),
[`04_seasonal_timing`](figures/04_seasonal_timing.png),
[`hotspot_map.html`](hotspot_map.html)).

---

## Q1 — Predicting the hotspots

**Setup.** Model at **district × ISO-week** resolution over the Sep–Nov window. Target is
`log1p(dry-matter burned, tonnes)`. Honest forward split: **train 2012–2016, test
2017–2018**, early-stopping on 2016. LightGBM regressor. Everything is compared to a
**climatology baseline** — each district-week's historical mean — because in a highly
seasonal problem, "it burns like it usually does" is a strong and fair yardstick.

Two model tiers, each run **with and without** the weather block (the ablation):

- **Structural (B)** — weather + calendar + geography + district/week climatology. *No
  within-season fire information.* "How well can drivers known before the season predict
  the pattern?"
- **Operational (A)** — B **plus autoregressive fire lags** (last week, season-to-date,
  same week last year). The real week-ahead nowcast.

### Results (test 2017–2018)

| Model | RMSE (log) | R² (log) | Spearman | ROC-AUC (top-10%) | Precision@k |
|---|---|---|---|---|---|
| Climatology baseline | 1.46 | 0.82 | 0.92 | 0.986 | 0.83 |
| Structural B — no weather | 1.50 | 0.81 | 0.91 | 0.986 | 0.84 |
| Structural B — **+weather** | 1.82 | **0.72** | 0.87 | 0.959 | 0.68 |
| Operational A — no weather | 1.09 | 0.90 | 0.955 | 0.988 | 0.83 |
| Operational A — **+weather** | **1.07** | **0.90** | **0.957** | 0.985 | 0.83 |

**Marginal value of the weather block:** Structural **ΔR² = −0.090**; Operational
**ΔR² = +0.004**.

### What this means

- **Persistence is almost everything.** Model A's skill is dominated by `dm_lag1`
  (last week's burning, **54%** of gain), district identity (**12%**), same-week-last-year
  (**7%**) and the crop calendar (`iso_week`, `week_clim`, **~12%** combined). The eight
  weather variables together contribute **≈3%**
  ([`06_feature_importance`](figures/06_feature_importance.png)).
- **Weather is close to useless *for prediction* here — and can mislead.** Given climatology
  + persistence, ERA5 adds nothing operationally and degrades the structural model by
  chasing year-to-year noise that doesn't generalise. This matches the EDA, where
  district-season burning barely correlated with rain/ET₀
  ([`03_fire_vs_weather`](figures/03_fire_vs_weather.png)). The decision to burn is driven
  by the harvest calendar and economics, not the weather — weather only gates whether a
  fire *can* be lit.
- **Operationally it works.** The week-ahead model catches ~83% of the worst-decile
  district-weeks in its own top-k, and its predicted 2018 map mirrors the actual one
  ([`05_pred_vs_actual`](figures/05_pred_vs_actual.png),
  [`07_earlywarning_heatmap`](figures/07_earlywarning_heatmap.png),
  [`08_model_skill`](figures/08_model_skill.png)). A dashboard needs last week's fire
  counts and the calendar — not a weather feed.

---

## Q2 — Did the subsidy reduce burning?

**Design.** A **dose-response difference-in-differences**. Treatment is a *continuous
intensity* (CRM machines per district), not an on/off switch. Because Punjab reports
*actual machines delivered* (cumulative 2018–22) while Haryana reports *2018-19 targets*,
raw counts aren't comparable across states — so intensity is harmonised to a
**within-state z-score** (`dose_z`). The estimand:

```
log_dm(i,t) = α_i + γ_t + β · (dose_i × post_t) + X(i,t)·θ + ε(i,t)
```

with district (`α_i`) and year (`γ_t`) fixed effects and district-clustered SEs. `β < 0`
⇒ higher-intensity districts cut burning more after 2018.

### Finding 1 — The subsidy was targeted at the worst offenders

Machines flowed to the districts that were already burning most: within-state
Spearman(machines, pre-period burning) = **+0.86 (Punjab)**, **+0.69 (Haryana)**;
regressing `dose_z` on pre-period log-burning (with a state dummy) gives
**β = +0.29, p < 0.001, R² = 0.45** ([`09_targeting`](figures/09_targeting.png)).

This is *sensible policy* but a *serious confound*: treated districts are structurally
high-burning, so a cross-sectional "more machines → more fire" comparison would
spuriously blame the subsidy. It is the reason a within-district DiD — which nets out
each district's fixed severity — is the only credible design.

### Finding 2 — Parallel pre-trends hold

The DiD's key assumption is that high- and low-dose districts were trending *together*
before the policy. An event study over the clean pre-period (2012–2017, reference 2017)
shows every `dose × year` coefficient's 95% CI **spanning zero**, with a joint
significance test of **p = 0.18** — no detectable differential pre-trend
([`10_pretrends_eventstudy`](figures/10_pretrends_eventstudy.png),
[`11_highlow_trajectories`](figures/11_highlow_trajectories.png)). The design is credible;
power is modest with 43 districts.

### Finding 3 — Placebos return null, as they must

Assigning a **fake policy in 2015** (all pre-treatment) yields β = −0.16 (p = 0.13); the
**2018 onset** — when essentially no machines were yet deployed by the Oct–Nov burning
season — yields β = +0.08 (p = 0.62), unchanged by weather controls (β = +0.09, p = 0.59).
The estimator does **not** manufacture an effect where none should exist.

### Finding 4 — The real estimate awaits one dataset

Everything needed for the causal answer is built and validated on the pre-period. The
missing ingredient is a **post-2018 district-year fire outcome**. `estimate_did()` is
written to consume exactly that:

```python
from fire_policy.causal import estimate_did, CONTROLS
panel = pd.concat([sage_2012_2018, firms_2019_plus])   # [state, district, year, dm_tonnes]
res = estimate_did(panel, policy_year=2018, controls=CONTROLS)   # → β, SE, p, event study
```

That post-2018 panel needs a **free NASA FIRMS `MAP_KEY`** (self-issued, but delivered by
email) or an Earthdata login — the single external credential this otherwise fully-keyless
pipeline cannot self-serve. With it, the same code that passed every placebo produces the
treatment effect and its dynamics.

---

## Honest limitations

- **Fire resolution.** SAGE-IGP is 0.25° (~667 km²); district totals are approximate, and
  area-weighting a coarse grid smooths small districts.
- **Treatment comparability.** Punjab = actual cumulative machines; Haryana = one-year
  targets that are near-uniform across districts (≈403 each), giving Haryana little usable
  dose variation. The within-state z-score is the most defensible harmonisation, but the
  identifying variation is really Punjab's.
- **No post-period outcome (yet).** The causal leg is a *validated design*, not a delivered
  effect estimate. This is stated plainly rather than papered over with a proxy.
- **Weather is ERA5 at one point per district**, aggregated to weekly — adequate for
  gating variables, not for microclimate.

## Reproduce

```bash
python -m fire_policy.geo         # build district layer
python -m fire_policy.sage        # fire panels (district×year, district×week)
python -m fire_policy.weather     # ERA5 weekly weather (multi-location, ~7 calls)
python -m fire_policy.treatment   # harmonised CRM dose panel
python -m fire_policy.eda         # figures 01–04 + hotspot map
python -m fire_policy.predict     # figures 05–08 + prediction ablation
python -m fire_policy.causal      # figures 09–11 + targeting/pre-trends/placebos
```

All intermediate panels are written to `data/processed/`; the models and DiD read from
there, so each stage is independently runnable and cheap to re-run.
