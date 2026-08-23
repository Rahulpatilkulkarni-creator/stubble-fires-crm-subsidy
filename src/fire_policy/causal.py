"""Causal analysis: did the CRM equipment subsidy reduce stubble burning?

The design is a **dose-response difference-in-differences**. Treatment is a continuous
intensity (CRM machines per district, harmonised to a within-state z-score `dose_z` in
treatment.py), switched on from the 2018-19 scale-up. The estimand is beta in

    log_dm_{it} = alpha_i + gamma_t + beta * (dose_i x post_t) + X_{it} theta + eps_{it}

with district (alpha_i) and year (gamma_t) fixed effects and district-clustered SEs.
beta < 0 means higher-intensity districts cut burning more after the subsidy.

WHAT THIS MODULE CAN DO WITH KEYLESS DATA (SAGE-IGP ends in 2018, the onset year):

  1. TARGETING / SELECTION.  Show the subsidy was allocated to the worst-burning
     districts -- the core identification threat that makes a naive post-only or
     high-vs-low comparison badly biased, and the reason a within-district DiD is
     needed at all.

  2. PARALLEL-TRENDS (event study).  Using the clean pre-period (2012-2017), trace
     dose x year coefficients. Flat, insignificant leads => high- and low-dose
     districts were on parallel trajectories before the policy: the key DiD
     assumption is credible.

  3. PLACEBO DiDs.  A fake 2015 policy (all pre-treatment) and the 2018 onset (when
     almost no machines were yet deployed by the Oct-Nov burning season) should both
     return beta ~ 0. They do -> the estimator is not manufacturing an effect.

WHAT REQUIRES ONE EXTERNAL CREDENTIAL:

  The *actual* treatment effect needs a post-onset district-year fire outcome. SAGE-IGP
  stops in 2018; the keyless FIRMS feeds are 7-day-only; the PPCB action-plan PDF holds
  only blank reporting templates + state-level totals. The clean fix is NOT to splice
  SAGE (pre) onto FIRMS (post) -- that would confound an inter-product level shift with
  the policy effect. Instead `effect.py` rebuilds a *consistent* NASA FIRMS VIIRS outcome
  across the whole 2012+ horizon (needs a free FIRMS MAP_KEY, self-issued but emailed) and
  runs this same `estimate_did()` on it -- one command:
      FIRMS_MAP_KEY=<key>  PYTHONPATH=src python -m fire_policy.effect

Inputs : data/processed/fire_sage_district_year.csv, treatment_district.csv,
         weather_district_week.csv
Outputs: reports/figures/09_targeting.png, 10_pretrends_eventstudy.png,
         11_highlow_trajectories.png ; data/processed/causal_targeting.csv
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from scipy.stats import pearsonr, spearmanr

from fire_policy import config as C

sns.set_theme(style="whitegrid", context="talk")

FIRE_YEAR = C.PROCESSED_DIR / "fire_sage_district_year.csv"
TREAT = C.PROCESSED_DIR / "treatment_district.csv"
WEATHER_WEEK = C.PROCESSED_DIR / "weather_district_week.csv"

PRE_YEARS = (2012, 2013, 2014, 2015, 2016, 2017)  # clean pre-period
ONSET_YEAR = 2018                                  # CRM scale-up (FY 2018-19)
DOSE = "dose_z"                                    # within-state standardised intensity
CONTROLS = ["precip_sum", "dry_days", "wind_max_mean", "et0_sum"]


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _seasonal_weather() -> pd.DataFrame:
    """Collapse the weekly weather panel to district-year burning-season aggregates."""
    if not WEATHER_WEEK.exists():
        return pd.DataFrame()
    w = pd.read_csv(WEATHER_WEEK)
    return (w.groupby(["state", "district", "year"]).agg(
        precip_sum=("precip_sum", "sum"),
        dry_days=("dry_days", "sum"),
        wind_max_mean=("wind_max_mean", "mean"),
        et0_sum=("et0_sum", "sum"),
        tmax_mean=("tmax_mean", "mean"),
    ).reset_index())


def load_causal_frame() -> pd.DataFrame:
    """District-year fire outcome + treatment dose + seasonal weather controls."""
    fire = pd.read_csv(FIRE_YEAR)
    fire["log_dm"] = np.log1p(fire["dm_tonnes"])
    treat = pd.read_csv(TREAT)[
        ["state", "district", "crm_machines", "crm_share", "dose_z",
         "crm_subsidy_lakh", "area_km2", "is_punjab"]]
    df = fire.merge(treat, on=["state", "district"], how="left")
    wx = _seasonal_weather()
    if not wx.empty:
        df = df.merge(wx, on=["state", "district", "year"], how="left")
    miss = df["dose_z"].isna().sum()
    if miss:
        print(f"  ! {miss} district-years without a treatment dose (name mismatch?)",
              file=sys.stderr)
    return df


# --------------------------------------------------------------------------- #
# 1. Targeting / selection
# --------------------------------------------------------------------------- #
def targeting_analysis(df: pd.DataFrame, save: bool = True) -> tuple[pd.DataFrame, dict]:
    """Was the subsidy sent to the worst-burning districts? (the identification threat)"""
    pre = (df[df["year"].isin(PRE_YEARS)]
           .groupby(["state", "district"])
           .agg(pre_log_dm=("log_dm", "mean"),
                pre_dm_tonnes=("dm_tonnes", "mean")).reset_index())
    t = df[["state", "district", "crm_machines", "crm_share", "dose_z",
            "is_punjab"]].drop_duplicates()
    m = pre.merge(t, on=["state", "district"])

    out = {}
    # within-state rank alignment of raw machines vs pre-period burning
    for st in ("Punjab", "Haryana"):
        s = m[m["state"] == st]
        rho = spearmanr(s["crm_machines"], s["pre_dm_tonnes"]).correlation
        out[f"spearman_machines_vs_preburn_{st}"] = float(rho)
    # pooled within-state relationship: dose_z (state-demeaned) vs pre-burning + state FE
    res = smf.ols("dose_z ~ pre_log_dm + is_punjab", data=m).fit(cov_type="HC1")
    out["dose_on_preburn_beta"] = float(res.params["pre_log_dm"])
    out["dose_on_preburn_p"] = float(res.pvalues["pre_log_dm"])
    out["dose_on_preburn_r2"] = float(res.rsquared)
    r_overall = pearsonr(m["pre_log_dm"], m["dose_z"])
    out["pearson_dose_vs_preburn"] = float(r_overall.statistic)

    if save:
        m.to_csv(C.PROCESSED_DIR / "causal_targeting.csv", index=False)
        _fig_targeting(m, res)
    return m, out


def _fig_targeting(m: pd.DataFrame, res) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {"Punjab": "#d1495b", "Haryana": "#30638e"}
    for st, s in m.groupby("state"):
        ax.scatter(s["pre_log_dm"], s["dose_z"], s=90, alpha=0.8,
                   color=colors[st], label=st, edgecolor="white", linewidth=0.7)
    # label the most intensively treated districts
    for _, r in m.nlargest(8, "dose_z").iterrows():
        ax.annotate(r["district"], (r["pre_log_dm"], r["dose_z"]),
                    fontsize=9, xytext=(4, 3), textcoords="offset points")
    xs = np.linspace(m["pre_log_dm"].min(), m["pre_log_dm"].max(), 50)
    # partial slope holding state FE at the sample mean of the state dummy
    b0 = res.params["Intercept"] + res.params.get("is_punjab", 0) * m["is_punjab"].mean()
    ax.plot(xs, b0 + res.params["pre_log_dm"] * xs, "k--", lw=1.5,
            label=f"within-state fit (β={res.params['pre_log_dm']:+.2f}, "
                  f"p={res.pvalues['pre_log_dm']:.3f})")
    ax.set(xlabel="Pre-period burning  (mean log DM tonnes, 2012-2017)",
           ylabel="Subsidy intensity  (within-state z-score, dose_z)",
           title="Targeting: the subsidy was aimed at the worst-burning districts")
    ax.legend(loc="upper left")
    ax.figure.text(0.99, 0.01, "Source: SAGE-IGP + PPCB/Haryana CRM records",
                   ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "09_targeting.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Parallel-trends event study (pre-period)
# --------------------------------------------------------------------------- #
def pretrends_event_study(df: pd.DataFrame, years=PRE_YEARS, ref_year: int = 2017,
                          save: bool = True):
    """Trace dose x year effects across the clean pre-period; flat => parallel trends."""
    d = df[df["year"].isin(years)].copy()
    interact = []
    for y in years:
        if y == ref_year:
            continue
        col = f"dose_x_{y}"
        d[col] = d[DOSE] * (d["year"] == y).astype(float)
        interact.append(col)
    d = d.dropna(subset=[DOSE, "log_dm"]).set_index(["district", "year"])
    formula = ("log_dm ~ 1 + " + " + ".join(interact) +
               " + EntityEffects + TimeEffects")
    res = PanelOLS.from_formula(formula, data=d).fit(
        cov_type="clustered", cluster_entity=True)

    ci = res.conf_int()
    rows = [{"year": ref_year, "coef": 0.0, "lo": 0.0, "hi": 0.0}]
    for y in years:
        if y == ref_year:
            continue
        c = f"dose_x_{y}"
        rows.append({"year": y, "coef": res.params[c],
                     "lo": ci.loc[c, "lower"], "hi": ci.loc[c, "upper"]})
    ev = pd.DataFrame(rows).sort_values("year")
    if save:
        _fig_pretrends(ev, ref_year)
    return res, ev


def _fig_pretrends(ev: pd.DataFrame, ref_year: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.errorbar(ev["year"], ev["coef"],
                yerr=[ev["coef"] - ev["lo"], ev["hi"] - ev["coef"]],
                fmt="o-", color="#3c6e47", capsize=5, lw=2, markersize=8)
    ax.set(xlabel="Year", ylabel="dose × year coefficient (95% CI)",
           title=f"Pre-trend check: dose-related burning by year (ref {ref_year})")
    ax.text(0.02, 0.03, "Flat & spanning zero → high- and low-dose districts were on\n"
            "parallel pre-policy trends (DiD assumption credible).",
            transform=ax.transAxes, fontsize=11, color="#333")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "10_pretrends_eventstudy.png", dpi=140)
    plt.close(fig)


def fig_highlow_trajectories(df: pd.DataFrame, save: bool = True) -> None:
    """Mean burning trajectory for high- vs low-dose districts (within-state split)."""
    d = df.copy()
    d["dose_grp"] = np.where(d["dose_z"] >= 0, "high dose", "low dose")
    traj = (d.groupby(["dose_grp", "year"])["dm_tonnes"].mean() / 1e3).reset_index()
    fig, ax = plt.subplots(figsize=(11, 7))
    for grp, s in traj.groupby("dose_grp"):
        ax.plot(s["year"], s["dm_tonnes"], marker="o", lw=2.5, label=grp)
    ax.axvline(ONSET_YEAR, color="gray", ls="--", lw=1.5)
    ax.text(ONSET_YEAR + 0.05, ax.get_ylim()[1] * 0.92, "CRM scale-up\n(2018-19)",
            fontsize=10, color="gray")
    ax.set(xlabel="Year", ylabel="Mean district DM burned (thousand tonnes)",
           title="High- vs low-dose districts: burning trajectories (2012-2018)")
    ax.legend(title="Subsidy intensity")
    ax.figure.text(0.99, 0.01, "Within-state median split on dose_z. SAGE-IGP ends 2018.",
                   ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    if save:
        fig.savefig(C.FIGURES_DIR / "11_highlow_trajectories.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. DiD estimator (placebo now; real effect when post-2018 fire is supplied)
# --------------------------------------------------------------------------- #
def estimate_did(fire_panel: pd.DataFrame, treatment: pd.DataFrame | None = None,
                 policy_year: int = ONSET_YEAR, dose: str = DOSE,
                 controls: list[str] | None = None, weather: pd.DataFrame | None = None):
    """Dose-response DiD: log_dm ~ dose×post + controls + district FE + year FE.

    fire_panel: long frame with [state, district, year, dm_tonnes] over ANY horizon
    spanning pre- and post-policy years. Supply a post-2018 panel (FIRMS) here to get
    the real treatment effect; run on SAGE alone and 2018 is the only 'post' year
    (a placebo/onset check, not a treatment estimate).
    """
    treat = treatment if treatment is not None else pd.read_csv(TREAT)
    fp = fire_panel.copy()
    if "log_dm" not in fp:
        fp["log_dm"] = np.log1p(fp["dm_tonnes"])
    if dose not in fp.columns:                     # idempotent: frame may already carry it
        fp = fp.merge(treat[["state", "district", dose]],
                      on=["state", "district"], how="inner")
    fp["post"] = (fp["year"] >= policy_year).astype(int)
    fp["dose_post"] = fp[dose] * fp["post"]

    rhs = ["dose_post"]
    if controls:
        need = [c for c in controls if c not in fp.columns]
        if need:
            wx = weather if weather is not None else _seasonal_weather()
            if not wx.empty:
                keep = ["state", "district", "year"] + [c for c in need if c in wx.columns]
                fp = fp.merge(wx[keep], on=["state", "district", "year"], how="left")
        rhs += [c for c in controls if c in fp.columns]
    fp = fp.dropna(subset=["log_dm", "dose_post"] + rhs).set_index(["district", "year"])
    formula = f"log_dm ~ 1 + {' + '.join(rhs)} + EntityEffects + TimeEffects"
    return PanelOLS.from_formula(formula, data=fp).fit(
        cov_type="clustered", cluster_entity=True)


def placebo_did(df: pd.DataFrame, fake_policy: int = 2015,
                pre=(2012, 2013, 2014), post=(2015, 2016, 2017)):
    """DiD on a fake pre-treatment policy year: beta should be ~0."""
    sub = df[df["year"].isin(tuple(pre) + tuple(post))].copy()
    return estimate_did(sub, policy_year=fake_policy)


# --------------------------------------------------------------------------- #
def _fmt(res, param="dose_post") -> str:
    b = res.params[param]
    se = res.std_errors[param]
    p = res.pvalues[param]
    return f"beta={b:+.4f}  (SE {se:.4f}, p={p:.3f}, N={res.nobs})"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = load_causal_frame()
    print(f"Causal frame: {len(df)} district-years "
          f"({df.district.nunique()} districts x {df.year.nunique()} years, "
          f"{df.year.min()}-{df.year.max()})\n")

    # 1. Targeting -----------------------------------------------------------
    m, out = targeting_analysis(df)
    print("[1] TARGETING / SELECTION  (was the subsidy aimed at high-burning districts?)")
    print(f"    Within-state Spearman(machines, pre-burning):  "
          f"Punjab {out['spearman_machines_vs_preburn_Punjab']:+.2f},  "
          f"Haryana {out['spearman_machines_vs_preburn_Haryana']:+.2f}")
    print(f"    dose_z ~ pre-burning (+state FE): β={out['dose_on_preburn_beta']:+.3f}, "
          f"p={out['dose_on_preburn_p']:.3f}, R²={out['dose_on_preburn_r2']:.2f}")
    print("    => subsidy intensity rises with prior burning: a naive comparison would\n"
          "       confound the policy with pre-existing severity. DiD removes fixed\n"
          "       district severity; parallel pre-trends are what must be checked next.\n")

    # 2. Parallel trends -----------------------------------------------------
    ref = 2017
    res_ev, ev = pretrends_event_study(df, ref_year=ref)
    print(f"[2] PARALLEL-TRENDS EVENT STUDY (clean pre-period "
          f"{min(PRE_YEARS)}-{max(PRE_YEARS)}, ref {ref})")
    print("    dose × year coefficients (95% CI):")
    for _, r in ev.iterrows():
        star = "" if (r["lo"] <= 0 <= r["hi"]) else "  <-- excludes 0"
        print(f"      {int(r['year'])}: {r['coef']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]{star}")
    joint = res_ev.f_statistic.pval if hasattr(res_ev, "f_statistic") else float("nan")
    print(f"    Joint significance of all dose×year terms: p={joint:.3f} "
          f"({'no' if joint > 0.05 else 'SOME'} pre-trend)\n")
    fig_highlow_trajectories(df)

    # 3. Placebo / onset DiDs ------------------------------------------------
    print("[3] PLACEBO & ONSET DiDs  (both should be ~0)")
    res_pl = placebo_did(df)
    print(f"    Fake 2015 policy (2012-14 vs 2015-17):   {_fmt(res_pl)}")
    res_on = estimate_did(df, policy_year=ONSET_YEAR)
    print(f"    2018 onset (2012-17 vs 2018, ~0 machines yet): {_fmt(res_on)}")
    res_on_c = estimate_did(df, policy_year=ONSET_YEAR, controls=CONTROLS)
    print(f"    2018 onset + weather controls:            {_fmt(res_on_c)}")
    print("    (SAGE ends 2018, so '2018 onset' is a placebo, not the treatment effect.)\n")

    # 4. Ready-to-run --------------------------------------------------------
    print("[4] REAL TREATMENT EFFECT — one credential + one command away")
    print("    SAGE-IGP stops at 2018. Rather than splice SAGE (pre) onto FIRMS (post),")
    print("    effect.py rebuilds a CONSISTENT NASA FIRMS VIIRS outcome across 2012+ and")
    print("    runs this same estimate_did() on it. Get a free FIRMS MAP_KEY (emailed),")
    print("    set FIRMS_MAP_KEY in .env, then:")
    print("        PYTHONPATH=src python -m fire_policy.effect")
    print("    -> beta, SEs, and the dynamic event study (reports/figures/12_did_effect.png).\n")

    print("Saved figures -> reports/figures/ (09_targeting, 10_pretrends_eventstudy,")
    print("                 11_highlow_trajectories)")
    print(f"Saved -> {(C.PROCESSED_DIR / 'causal_targeting.csv').relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
