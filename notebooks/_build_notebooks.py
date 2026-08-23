"""Build & execute the four narrative notebooks from the package + processed panels.

Run:  PYTHONPATH=src .venv/Scripts/python notebooks/_build_notebooks.py

Each notebook is a thin, narrative wrapper over `fire_policy` — it reads the processed
panels, calls the analysis functions, and embeds the figures the modules save. Executing
here bakes tables/figures into the .ipynb so they render on GitHub without a kernel.
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
KERNEL = "fire_policy"

BOOTSTRAP = (
    "import sys, os, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "ROOT = os.path.dirname(os.getcwd()) if os.path.basename(os.getcwd()) == 'notebooks' "
    "else os.getcwd()\n"
    "os.chdir(ROOT)\n"
    "if os.path.join(ROOT, 'src') not in sys.path: sys.path.insert(0, os.path.join(ROOT, 'src'))\n"
    "import pandas as pd, numpy as np\n"
    "from IPython.display import Image, display\n"
    "pd.set_option('display.width', 140); pd.set_option('display.max_columns', 40)\n"
    "print('project root:', ROOT)"
)


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


def img(*paths: str) -> str:
    body = ", ".join(f"'reports/figures/{p}'" for p in paths)
    return f"for _p in [{body}]:\n    display(Image(filename=_p))"


# --------------------------------------------------------------------------- #
NOTEBOOKS: dict[str, list] = {}

NOTEBOOKS["01_data.ipynb"] = [
    md("""
# 01 · Building the data

Stubble-burning fires vs. the crop-residue subsidy in **Punjab & Haryana**. This project
is built from **raw satellite grids and primary government records**, not a pre-cleaned
CSV. Three processed panels drive everything downstream:

| Panel | Grain | Source |
|---|---|---|
| `fire_sage_district_{year,week}` | district × year / week | SAGE-IGP 0.25° daily burned mass (CC0) |
| `weather_district_week` | district × week | Open-Meteo ERA5 archive (keyless) |
| `treatment_district` | district | PPCB MIS + Haryana CRM plan |

**The 2018 wall:** the keyless fire record (SAGE-IGP) ends in 2018 — the CRM scale-up
year — which shapes the whole causal strategy (see notebook 04).
"""),
    code(BOOTSTRAP),
    md("## Fire — SAGE-IGP burned dry matter\nArea-weighted from the 0.25° grid to districts (equal-area overlay), so small districts aren't dropped."),
    code(
        "fire = pd.read_csv('data/processed/fire_sage_district_year.csv')\n"
        "print(f'{len(fire)} district-years | {fire.district.nunique()} districts | "
        "{fire.year.min()}-{fire.year.max()}')\n"
        "display(fire.head())\n"
        "piv = (fire.groupby(['year','state'])['dm_tonnes'].sum()/1e6).unstack('state').round(2)\n"
        "print('\\nState-year dry matter burned (million tonnes):'); display(piv)"
    ),
    md("Weekly grain (the prediction target). Burning peaks sharply in **ISO weeks 44–45** (early November)."),
    code(
        "fw = pd.read_csv('data/processed/fire_sage_district_week.csv')\n"
        "print(f'{len(fw)} district-weeks'); \n"
        "peak = fw.groupby('iso_week')['dm_tonnes'].sum().sort_values(ascending=False).head(5)\n"
        "print('Top ISO weeks by total burning:'); display((peak/1e3).round(1).rename('DM (kt)'))"
    ),
    md("## Weather — Open-Meteo ERA5 (keyless)\nFetched for all 43 districts in one multi-location call per year (~7 requests total)."),
    code(
        "wx = pd.read_csv('data/processed/weather_district_week.csv')\n"
        "print(f'{len(wx)} district-weeks | {wx.year.min()}-{wx.year.max()}')\n"
        "display(wx.head())"
    ),
    md("""## Treatment — harmonised CRM dose

Punjab reports **actual machines delivered** (cumulative 2018-22); Haryana reports
**2018-19 targets**. Not level-comparable, so intensity is harmonised to a **within-state
z-score** (`dose_z`) and share. District names are crosswalked to the geography layer
(1:1, validated)."""),
    code(
        "tr = pd.read_csv('data/processed/treatment_district.csv')\n"
        "for st in ['Punjab','Haryana']:\n"
        "    s = tr[tr.state==st]\n"
        "    print(f\"{st}: {int(s.crm_machines.sum()):,} machines ({s.source_kind.iloc[0]})\")\n"
        "display(tr.sort_values('crm_machines', ascending=False)"
        "[['state','district','crm_machines','crm_share','dose_z']].head(8))"
    ),
    md("**Takeaway:** three clean, keyless panels on a common 43-district grid — ready for EDA, prediction, and the causal design."),
]

NOTEBOOKS["02_eda.ipynb"] = [
    md("""
# 02 · What the fires look like

Descriptive groundwork before modelling. Two facts shape both later questions:
**burning is spatially concentrated** (Punjab ≫ Haryana; a handful of districts dominate)
and **weakly related to weather** at the seasonal scale.
"""),
    code(BOOTSTRAP),
    code(
        "import fire_policy.eda as E\n"
        "fire, weather = E._load()\n"
        "means = (fire.groupby('state')['dm_tonnes'].sum()/1e6/fire.year.nunique()).round(2)\n"
        "print('Mean annual DM burned (Mt):', means.to_dict())\n"
        "print('Punjab share of two-state total: "
        "%.0f%%' % (100*fire[fire.state=='Punjab'].dm_tonnes.sum()/fire.dm_tonnes.sum()))"
    ),
    md("### State trend & the worst districts\nPunjab burns ~6× Haryana; Sangrur, Bathinda, Moga, Ludhiana and Muktsar lead."),
    code(img("01_state_trend.png", "02_top_districts.png")),
    md("### When does it peak?\nA tight early-November spike — the ~2–3 week paddy-to-wheat turnaround."),
    code(img("04_seasonal_timing.png")),
    md("### Fire vs weather — weak at the seasonal scale\nThis is the clue that weather *gates* burning but doesn't decide *where* it concentrates (confirmed by the prediction ablation in notebook 03)."),
    code(
        "corrs = E.fig_fire_weather(fire, weather)\n"
        "print('District-season correlations (DM vs driver):')\n"
        "for k,v in corrs.items(): print(f'  {k:12s} r = {v:+.2f}')"
    ),
    code(img("03_fire_vs_weather.png")),
    md("An interactive district choropleth is written to [`reports/hotspot_map.html`](../reports/hotspot_map.html)."),
]

NOTEBOOKS["03_prediction.ipynb"] = [
    md("""
# 03 · Predicting the hotspots

**Question:** can we see next season's hotspots coming? Model at **district × ISO-week**,
target `log1p(DM tonnes)`, honest forward split **train 2012–16 / test 2017–18**, LightGBM.

Everything is benchmarked against a **climatology baseline** (each district-week's
historical mean), and each model is run **with and without** the ERA5 weather block —
an explicit ablation of how much weather actually helps.
"""),
    code(BOOTSTRAP),
    code(
        "import fire_policy.predict as P\n"
        "df = P.add_climatology(P.build_frame())\n"
        "print(f'{len(df)} district-weeks | weather merged: {df.tmax_mean.notna().any()}')"
    ),
    md("### Baseline + four models (with / without weather)"),
    code(
        "base = P.eval_baseline(df)\n"
        "_, _, m_b0, _ = P.train_eval(df, P.BASE_NO_WX, 'structural  (B, no weather)')\n"
        "mb, te_b, m_b, _ = P.train_eval(df, P.BASE_FEATURES, 'structural  (B, +weather)')\n"
        "_, _, m_a0, _ = P.train_eval(df, P.LAG_NO_WX,  'operational (A, no weather)')\n"
        "ma, te_a, m_a, fa = P.train_eval(df, P.LAG_FEATURES, 'operational (A, +weather)')\n"
        "cols=['model','RMSE_log','R2_log','Spearman','ROC_AUC_top10pct','Precision@k']\n"
        "display(pd.DataFrame([base,m_b0,m_b,m_a0,m_a])[cols].round(3))"
    ),
    md("### Marginal value of the weather block"),
    code(
        "for lbl, mw, m0 in [('Structural (B)', m_b, m_b0), ('Operational (A)', m_a, m_a0)]:\n"
        "    print(f\"{lbl:16s} dR2={mw['R2_log']-m0['R2_log']:+.3f}  \"\n"
        "          f\"dSpearman={mw['Spearman']-m0['Spearman']:+.3f}  \"\n"
        "          f\"dRMSE={mw['RMSE_log']-m0['RMSE_log']:+.3f}\")"
    ),
    md("""**Weather barely moves the operational model (+0.004 R²) and *hurts* the structural
one (−0.09 R²).** The signal is persistence + calendar, not weather."""),
    md("### Predicted vs actual, and what drives the forecast"),
    code(img("05_pred_vs_actual.png", "06_feature_importance.png")),
    md("`dm_lag1` (last week's burning) alone is ~54% of the model's gain; the 8 weather features together are ~3%."),
    md("### Early-warning check — does it light up the right district-weeks?"),
    code(img("07_earlywarning_heatmap.png", "08_model_skill.png")),
    md("""**Takeaway:** hotspots are highly predictable a week ahead (R²≈0.90, ROC-AUC≈0.99),
and an operational dashboard needs last week's fire counts + the calendar — *not* a
weather feed."""),
]

NOTEBOOKS["04_causal.ipynb"] = [
    md("""
# 04 · Did the subsidy work?

**Design — dose-response difference-in-differences.** Treatment is a *continuous
intensity* (CRM machines, harmonised to `dose_z`), switched on from 2018-19:

$$\\log\\_dm_{it} = \\alpha_i + \\gamma_t + \\beta\\,(dose_i \\times post_t) + X_{it}\\theta + \\varepsilon_{it}$$

with district ($\\alpha_i$) and year ($\\gamma_t$) fixed effects, district-clustered SEs.
$\\beta<0$ ⇒ higher-intensity districts cut burning more after the subsidy.
"""),
    code(BOOTSTRAP),
    code(
        "import fire_policy.causal as CA\n"
        "df = CA.load_causal_frame()\n"
        "print(f'{len(df)} district-years, {df.year.min()}-{df.year.max()}')\n"
        "display(df.head())"
    ),
    md("""## 1 · Targeting — the identification threat
Was the subsidy aimed at the worst-burning districts? If so, a naive comparison is
confounded and a within-district DiD is required."""),
    code(
        "m, out = CA.targeting_analysis(df)\n"
        "print('Within-state Spearman(machines, pre-burning): "
        "Punjab %+.2f, Haryana %+.2f' % (out['spearman_machines_vs_preburn_Punjab'], "
        "out['spearman_machines_vs_preburn_Haryana']))\n"
        "print('dose_z ~ pre-burning (+state FE): beta=%+.3f, p=%.3f, R2=%.2f' % "
        "(out['dose_on_preburn_beta'], out['dose_on_preburn_p'], out['dose_on_preburn_r2']))"
    ),
    code(img("09_targeting.png")),
    md("""## 2 · Parallel trends — the key DiD assumption
Were high- and low-dose districts trending *together* before the policy? An event study
over the clean pre-period (2012–17); flat, zero-spanning coefficients ⇒ the assumption is
credible **when pooled**."""),
    code(
        "res_ev, ev = CA.pretrends_event_study(df, ref_year=2017)\n"
        "display(ev.round(3))\n"
        "print('Joint significance of dose x year terms: p=%.3f' % res_ev.f_statistic.pval)"
    ),
    code(img("10_pretrends_eventstudy.png", "11_highlow_trajectories.png")),
    md("""> **Caveat carried into §4.** The pooled pre-trend above (SAGE, keyless) does not
reject — joint **p = 0.18**. On the *FIRMS* causal outcome used in §4 the pooled test is
similar (**p = 0.08**, with 2016 just excluding zero), **but within Punjab it fails
(p = 0.005)**: the highest-dose districts were on a rising differential trajectory before the
policy that continues afterward. That is the main reason the §4 effect is read as *"no
evidence of a dose-response reduction,"* not a clean zero."""),
    md("## 3 · Placebo DiDs — do they return null?\nA fake pre-treatment policy (2015) and the 2018 onset (≈0 machines deployed by that burning season) should both give β≈0."),
    code(
        "res_pl = CA.placebo_did(df)\n"
        "res_on = CA.estimate_did(df, policy_year=2018)\n"
        "for name, r in [('fake-2015 policy', res_pl), ('2018 onset', res_on)]:\n"
        "    print('%-18s beta=%+.4f (SE %.4f, p=%.3f)' % "
        "(name, r.params['dose_post'], r.std_errors['dose_post'], r.pvalues['dose_post']))"
    ),
    md("""## 4 · The real effect — on a consistent FIRMS outcome

SAGE-IGP stops at 2018, so it holds no post-treatment fire outcome. The clean fix is
**not** to splice SAGE (pre) onto FIRMS (post) — two different products meeting at the
treatment boundary would confound an inter-product level shift with the policy effect.
Instead [`fire_policy/effect.py`](../src/fire_policy/effect.py) rebuilds a **consistent NASA
FIRMS VIIRS outcome across the whole 2012–2023 horizon** (same sensor every year) and runs
this same `estimate_did()` on it. The first run needs a free NASA FIRMS `MAP_KEY`
(self-issued, emailed); it caches the panel, so the DiD rebuilds offline afterwards:

```bash
PYTHONPATH=src python -m fire_policy.effect
```
"""),
    code(
        "eff = pd.read_csv('data/processed/causal_effect_firms.csv')\n"
        "firms = pd.read_csv('data/processed/fire_firms_district_year.csv')\n"
        "print(f'FIRMS outcome panel: {len(firms)} district-years, "
        "{firms.year.min()}-{firms.year.max()} ({firms.district.nunique()} districts)')\n"
        "row = eff.iloc[0]\n"
        "print('Headline DiD  dose x post beta = %+.4f (SE %.4f, p=%.3f, N=%d)' % "
        "(row['beta'], row['se'], row['p'], int(row['N'])))\n"
        "display(eff.round(4))"
    ),
    code(img("12_did_effect.png", "13_aggregate_trend.png")),
    md("""**Conclusion — a precise null, honestly confounded.** dose × post **β = +0.04
(p = 0.58, N = 516)**: no dose-response reduction, and every robustness spec lands
zero-to-positive. Aggregate Oct–Nov burning *did* fall after 2018 (**Haryana −44%,
Punjab −18%**), but a year-fixed-effect design absorbs any common trend — it can only test
whether *higher-dose* districts fell *more*, and they didn't. Identification is genuinely
hard here (near-perfect targeting ρ ≈ 0.88; the pre-trend *fails* within Punjab, p = 0.005;
the usable dose variation is **effectively Punjab-only**, Haryana ≈ 403 machines/district),
so the honest verdict is **"no detectable dose-response effect, and the quasi-experiment
can't cleanly isolate one"** — not a clean causal zero. See
[`reports/FINDINGS.md`](../reports/FINDINGS.md) Q2 for the full treatment."""),
]


def build() -> None:
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {
            "name": KERNEL, "display_name": "Python (fire_policy)", "language": "python"}
        path = NB_DIR / name
        print(f"executing {name} ...", flush=True)
        client = NotebookClient(nb, timeout=900, kernel_name=KERNEL,
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbf.write(nb, path)
        print(f"  wrote {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    build()
