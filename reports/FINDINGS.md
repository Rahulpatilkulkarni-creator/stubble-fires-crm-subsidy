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
   parallel-trends assumption is **credible when pooled** (joint pre-trend test p = 0.18 on
   the keyless SAGE pre-panel), though it **weakens on the FIRMS causal outcome and fails
   within Punjab** — the honest caveat spelled out in Finding 3.

3. **Aggregate burning fell after the subsidy scaled up — but the subsidy can't be
   *credited* for it.** With a consistent NASA FIRMS fire outcome now built across the whole
   **2012–2023** horizon, the dose-response DiD returns **β = +0.04 (p = 0.58)** — no
   reduction, faintly *wrong*-signed, and every robustness specification lands zero-to-positive.
   Total Oct–Nov detections did drop post-2018 (**Haryana −44%, Punjab −18%**), but a
   dose-response design can't attribute a *common* trend to the subsidy, and the
   *differential* test — did more-subsidised districts fall *more*? — is null. Identification
   is genuinely hard here (near-perfect targeting, ρ ≈ 0.88; a pre-trend that *fails* within
   Punjab; effectively Punjab-only variation), so the honest verdict is **"no detectable
   dose-response effect, and the quasi-experiment cannot cleanly isolate one."**

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
| Fire — burned mass *(prediction + pre-period)* | **SAGE-IGP** (Liu et al. 2020, Harvard Dataverse, CC0) | 0.25° daily → district×week | **2012–2018** | keyless |
| Fire — active-fire counts *(causal outcome)* | **NASA FIRMS VIIRS-SNPP** (standard-processing archive) | 375 m detections → district×year | **2012–2023** | free key |
| Weather | **Open-Meteo ERA5 archive** | point daily → district×week | 2012–2018 | keyless |
| Treatment | **PPCB MIS** (Punjab, actual machines) + **Haryana CRM action plan** (2018-19 targets) | district | 2018–2022 | primary PDFs/portal |
| Geography | **geoBoundaries** ADM2 (CC-BY) | 43 districts | — | keyless |

Fire mass is allocated from the coarse 0.25° grid to districts by **area-weighted polygon
overlay** (equal-area CRS), not centroid assignment, so small districts aren't dropped.

**Two fire products, each used where it stays clean.** SAGE-IGP ends in 2018, the
subsidy's scale-up year — fine for prediction (train 2012–16, test 2017–18) and every
*pre-period* causal check, but with no post-treatment observations of its own. The wrong
fix would be to splice SAGE (pre) onto FIRMS (post): two different products meeting exactly
at the treatment boundary confound an inter-product level shift with the policy effect. So
the causal outcome is built entirely from a **single, consistent FIRMS VIIRS series across
2012–2023** (same sensor, same processing every year), while SAGE dry-matter mass carries
the prediction leg. The one external credential the otherwise-keyless pipeline needs is a
free FIRMS `MAP_KEY` (self-issued, emailed) — used once to pull the archive.

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

with district (`α_i`) and year (`γ_t`) fixed effects and district-clustered SEs. The
outcome is `log(1 + FIRMS VIIRS burning-season detections)` — the consistent 2012–2023
series described above — and `post` switches on in 2018. `β < 0` ⇒ higher-intensity
districts cut burning more after the subsidy.

### Finding 1 — The subsidy was targeted at the worst offenders

Machines flowed to the districts that were already burning most: within-state
Spearman(machines, prior burning) = **+0.88 (Punjab), +0.83 (Haryana)** on the FIRMS
pre-period (**+0.86 / +0.69** on SAGE — the same story either way); regressing `dose_z` on
pre-period log-burning (with a state dummy) gives **β = +0.29, p < 0.001, R² = 0.45**
([`09_targeting`](figures/09_targeting.png)).

This is *sensible policy* but a *serious confound*: treated districts are structurally
high-burning, so a cross-sectional "more machines → more fire" comparison would
spuriously blame the subsidy. It is the reason a within-district DiD — which nets out
each district's fixed severity — is the only credible design, and (as Finding 3 shows) even
that isn't enough to fully clean it.

### Finding 2 — No dose-response reduction shows up

Running the DiD on the consistent 2012–2023 FIRMS outcome gives the headline estimate:

> **β = +0.044**  (SE 0.078, p = 0.58, N = 516 district-years)

That is a **precise zero, faintly the wrong way**. The 95% CI is **[−0.11, +0.20]** — for a
one-SD increase in within-state subsidy intensity, the change in burning-season fire count
lands between **−10% and +22%**. So the data *rule out* a reduction larger than ~10% per SD,
but can't distinguish a small effect from none. And the result is not fragile — it holds
across every specification I tried, none of which turns negative:

| Specification | β | p |
|---|---|---|
| Headline (both states, 2012–2023) | **+0.044** | 0.58 |
| Punjab only (where the dose actually varies) | +0.212 | **0.002** |
| Drop 2020–21 (COVID + farm-law protests) | +0.024 | 0.77 |
| Outcome = log(1 + FRP sum), i.e. fire *intensity* | +0.099 | 0.25 |
| Binary treatment (dose above own-state mean) | +0.053 | 0.74 |
| Dose = district share of state machine total | +1.40 | 0.66 |

The **event study** ([`12_did_effect`](figures/12_did_effect.png)) tells the same story
dynamically: the post-2018 `dose × year` coefficients bounce around zero
(+0.13, +0.01, −0.03, +0.15, +0.03, −0.18) with **no downward drift as machines accumulate**
— the opposite of what a working, gradually-deploying subsidy should produce.

### Finding 3 — …and the design cannot cleanly identify one anyway

The one specification that *is* significant (Punjab-only, β = +0.21) points the *wrong* way,
which is the tell: this is **residual confounding, not a perverse causal effect of machines**.
Three things break clean identification, and I'd rather name them than bury them:

- **Targeting is near-perfect** (ρ ≈ 0.88). District fixed effects remove each district's
  *fixed* severity, but not a *differential trend* among the worst burners — and there is one.
- **Pre-trends fail within Punjab.** The pre-period event study is broadly clean when
  pooled (on the FIRMS causal outcome, joint p = 0.08 — though the 2016 coefficient already
  excludes zero; the keyless SAGE pre-panel in [`10_pretrends_eventstudy`](figures/10_pretrends_eventstudy.png)
  gives p = 0.18), but **within Punjab the FIRMS joint pre-trend test rejects (p = 0.005)**:
  high-dose Punjab districts were already on a *rising* differential trajectory *before* the
  policy (dose × year climbs from −0.17 in 2012 toward the 2017 reference) that then
  continues post-2018. That upward pre-trend is exactly what manufactures the spurious
  positive Punjab-only "effect."
- **The variation is effectively Punjab-only.** Haryana's dose is near-uniform (one-year
  targets ≈403/district), so it adds almost no cross-district contrast — this is really a
  single-state, ~22-district design, with the modest power that implies.
- **The post-period has its own shocks.** 2020–21 saw anomalous Punjab burning
  (COVID-disrupted labour and the farm-law protests: +11% and +15% above the pre-mean),
  orthogonal to machine density. Dropping those years leaves β ≈ 0 (p = 0.77) — the shocks
  aren't driving the null, but they underline how noisy the post-window is.

What *does* behave: the **placebos**. A fake 2015 policy yields β = −0.16 (p = 0.13) and the
2018 onset (before machines were really in the field) β = +0.08 (p = 0.62) — the estimator
isn't inventing effects; there simply isn't a dose-response signal to find.

### Finding 4 — Burning did fall after 2018 — just not in a way we can pin on the subsidy

Aggregate Oct–Nov detections dropped meaningfully after the scale-up
([`13_aggregate_trend`](figures/13_aggregate_trend.png)): **Haryana −44%, Punjab −18%**
(post-2018 mean vs 2012–17 mean), with the sharpest falls in **2022 (−32%)** and
**2023 (−54%)** in Punjab. It is tempting to read that as the subsidy working.

But a **dose-response DiD with year fixed effects is structurally blind to a common,
state-wide trend** — the year effects absorb it. All the DiD can test is whether
*more-subsidised* districts fell *more*, and Finding 2 says they didn't. So this decline is
real but **unattributable to machine intensity**: it is equally consistent with intensified
enforcement and penalties, ex-situ straw markets, weather, awareness campaigns, or the
subsidy acting *uniformly* rather than in proportion to machines-per-district. What the data
specifically **rule out** is the mechanism *"more CRM machines in a district → proportionally
less burning there."* That reads consistently with the field evidence on low machine
utilisation — custom-hiring-centre gaps, fuel and time costs, and Happy-Seeder adoption
friction — where owning more machines didn't translate into proportionally less fire.

---

## Honest limitations

- **The causal design can only test a *dose-response* effect.** With year fixed effects it
  is blind to any uniform, state-wide impact of the scheme — a real but evenly-spread
  reduction would be invisible to β. The null is specifically about *machines-per-district*,
  not about the CRM programme as a whole (see Finding 4).
- **Identification is confounded, not textbook-clean.** Near-perfect targeting (ρ ≈ 0.88)
  plus a differential pre-trend within Punjab (joint p = 0.005) mean the within-district DiD
  removes fixed severity but not everything. The estimate is best read as *"no evidence of a
  dose-response reduction,"* not a sharp causal zero.
- **Two fire products.** Prediction uses SAGE-IGP burned *mass* (2012–18); the causal
  outcome uses FIRMS VIIRS *detection counts* (2012–23). Different physical quantities, each
  chosen where it stays internally consistent — not interchangeable.
- **Treatment comparability.** Punjab = actual cumulative machines; Haryana = one-year
  targets that are near-uniform across districts (≈403 each), so the usable dose variation
  is really Punjab's — effectively a single-state design.
- **Fire resolution.** SAGE-IGP is 0.25° (~667 km²); district totals are approximate, and
  area-weighting a coarse grid smooths small districts. (FIRMS points are 375 m, so the
  causal outcome is spatially sharper than the prediction target.)
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
FIRMS_MAP_KEY=<key> python -m fire_policy.effect   # figures 12–13 + dose-response DiD (FIRMS 2012–23)
```

The `effect` step is the only one needing a credential — a free NASA FIRMS `MAP_KEY`
(emailed on signup) — and only for the *first* run: it caches the FIRMS panel to
`data/processed/`, after which the DiD and figures rebuild offline. All intermediate panels
are written to `data/processed/`; the models and DiD read from there, so each stage is
independently runnable and cheap to re-run.
