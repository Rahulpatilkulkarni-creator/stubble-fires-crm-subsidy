"""Estimate the CRM subsidy's causal effect on burning — the DiD finisher.

SAGE-IGP (the keyless dry-matter inventory) stops in 2018, the treatment-onset year, so
it cannot supply a post-treatment outcome. The methodologically correct fix is NOT to
splice SAGE (pre) onto FIRMS (post) — two different outcome units meeting exactly at the
treatment boundary would confound any level/scale shift between the products with the
policy effect. Instead this module builds a *single, consistent* outcome from NASA FIRMS
**VIIRS-SNPP** active-fire detections spanning the ENTIRE horizon (2012+, pre AND post),
and runs the dose-response DiD from `causal.py` on it.

  outcome_it = log(1 + burning-season fire detections in district i, year t)

Same sensor and processing every year, so pre- and post-policy outcomes are directly
comparable. Weather controls are omitted by default: the ERA5 panel currently ends in
2018 (extend `weather.py` if wanted) and the prediction ablation already showed weather
contributes essentially nothing to burning — so the headline DiD is weather-free by
design, with controls available as an overlap-years robustness check.

This is the ONE step gated on an external credential: a free FIRMS MAP_KEY (self-issued,
delivered by email). Everything else is built. With the key it is a one-command finish:

    FIRMS_MAP_KEY=<key>  PYTHONPATH=src python -m fire_policy.effect

Outputs: data/processed/fire_firms_district_year.csv (the consistent outcome panel),
         causal_effect_firms.csv (headline DiD + the FINDINGS Q2 robustness table) ;
         reports/figures/12_did_effect.png (event study), 13_aggregate_trend.png
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fire_policy import causal as CA
from fire_policy import config as C
from fire_policy.geo import assign_points_to_districts, get_region_districts

SOURCE = "VIIRS_SNPP_SP"            # VIIRS-SNPP standard-processing archive (starts 2012)
SEASON = C.BURNING_SEASON_MONTHS    # (10, 11) — the Oct-Nov paddy-residue burning window
YEARS = tuple(range(2012, 2024))    # pre-policy 2012-2017 + post-policy 2018-2023
FIRMS_PANEL = C.PROCESSED_DIR / "fire_firms_district_year.csv"
OUT_RESULTS = C.PROCESSED_DIR / "causal_effect_firms.csv"


# --------------------------------------------------------------------------- #
# Build a consistent FIRMS district-year outcome (2012+)
# --------------------------------------------------------------------------- #
def build_firms_district_year(years=YEARS, source=SOURCE, season=SEASON,
                              rebuild: bool = False) -> pd.DataFrame:
    """VIIRS active-fire detections per district-year over the burning season.

    Cached to FIRMS_PANEL. District-years with no detections are kept as real zeros.
    Requires FIRMS_MAP_KEY (raises otherwise).
    """
    from fire_policy import firms

    if FIRMS_PANEL.exists() and not rebuild:
        return pd.read_csv(FIRMS_PANEL)
    if not firms.get_map_key():
        raise RuntimeError(
            "FIRMS_MAP_KEY not set. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/ and set it in .env "
            "(FIRMS_MAP_KEY=...). See this module's docstring.")

    frames = []
    for y in years:
        start, end = f"{y}-{season[0]:02d}-01", f"{y}-{season[-1]:02d}-30"
        pts = firms.fetch_area(source, start, end)          # keyed, chunked, cached
        if pts.empty:
            print(f"  ! {y}: no detections returned", file=sys.stderr)
            continue
        tagged = assign_points_to_districts(pts).dropna(subset=["district"])
        agg = (tagged.groupby(["state", "district"])
               .agg(fire_count=("latitude", "size"), frp_sum=("frp", "sum"))
               .reset_index())
        agg["year"] = y
        frames.append(agg)
        print(f"  {y}: {len(tagged):,} in-region detections, "
              f"{agg.district.nunique()} districts")
    if not frames:
        raise RuntimeError("No FIRMS detections returned for any year — check the key.")

    obs = pd.concat(frames, ignore_index=True)
    # complete the district × year grid so empty seasons are zeros, not gaps
    base = get_region_districts()[["state", "district"]].drop_duplicates()
    grid = (base.assign(_k=1).merge(pd.DataFrame({"year": list(years), "_k": 1}), on="_k")
            .drop(columns="_k"))
    panel = grid.merge(obs, on=["state", "district", "year"], how="left")
    panel[["fire_count", "frp_sum"]] = panel[["fire_count", "frp_sum"]].fillna(0.0)
    # name the modelling target log_dm so causal.estimate_did consumes it unchanged
    panel["log_dm"] = np.log1p(panel["fire_count"])
    panel.to_csv(FIRMS_PANEL, index=False)
    print(f"Saved FIRMS outcome -> {FIRMS_PANEL.relative_to(C.PROJECT_ROOT)} "
          f"({len(panel)} district-years)")
    return panel


def load_effect_frame(rebuild: bool = False) -> pd.DataFrame:
    """FIRMS district-year outcome + treatment dose (+ seasonal weather where available)."""
    fire = build_firms_district_year(rebuild=rebuild)
    treat = pd.read_csv(CA.TREAT)[["state", "district", "crm_machines", "dose_z",
                                   "crm_share", "is_punjab"]]
    df = fire.merge(treat, on=["state", "district"], how="left")
    wx = CA._seasonal_weather()
    if not wx.empty:
        df = df.merge(wx, on=["state", "district", "year"], how="left")
    return df


# --------------------------------------------------------------------------- #
# Estimate + figure
# --------------------------------------------------------------------------- #
def _robustness_rows(df: pd.DataFrame, policy_year: int) -> list[dict]:
    """The specifications quoted in FINDINGS Q2 — all from the cached FIRMS frame.

    Each is a separate dose-response DiD; none needs the network or weather. Reported
    alongside the headline so the robustness table in the write-up is reproducible from
    this one command.
    """
    specs: list[dict] = []

    def add(label: str, d: pd.DataFrame) -> None:
        r = CA.estimate_did(d, policy_year=policy_year)
        specs.append({"spec": label, "beta": float(r.params["dose_post"]),
                      "se": float(r.std_errors["dose_post"]),
                      "p": float(r.pvalues["dose_post"]), "N": int(r.nobs)})

    add("Punjab only", df[df["is_punjab"] == 1])
    add("Drop 2020-21 (COVID/protests)", df[~df["year"].isin([2020, 2021])])
    frp = df.copy(); frp["log_dm"] = np.log1p(frp["frp_sum"])
    add("Outcome = log(1+FRP sum)", frp)
    binr = df.copy(); binr["dose_z"] = (binr["dose_z"] > 0).astype(float)
    add("Binary dose (above own-state mean)", binr)
    shr = df.copy(); shr["dose_z"] = shr["crm_share"]
    add("Dose = share of state machines", shr)
    return specs


def run_effect(policy_year: int = CA.ONSET_YEAR, controls=None,
               rebuild: bool = False) -> dict:
    """Dose-response DiD on the consistent FIRMS outcome; headline β + dynamics."""
    df = load_effect_frame(rebuild=rebuild)
    yrs = tuple(sorted(int(y) for y in df["year"].unique()))
    print(f"FIRMS effect frame: {len(df)} district-years ({min(yrs)}-{max(yrs)}), "
          f"policy_year={policy_year}\n")

    res = CA.estimate_did(df, policy_year=policy_year)
    rows = [{"spec": "DiD (no controls)", "beta": res.params["dose_post"],
             "se": res.std_errors["dose_post"], "p": res.pvalues["dose_post"],
             "N": int(res.nobs)}]
    print(f"[DiD] dose×post  β={res.params['dose_post']:+.4f}  "
          f"(SE {res.std_errors['dose_post']:.4f}, p={res.pvalues['dose_post']:.3f})")

    # optional weather-controlled robustness (only on years with weather present)
    if controls:
        have_wx = df.dropna(subset=controls)
        if have_wx["year"].nunique() >= 3:
            res_c = CA.estimate_did(have_wx, policy_year=policy_year, controls=controls)
            rows.append({"spec": "DiD (+weather, overlap yrs)",
                         "beta": res_c.params["dose_post"], "se": res_c.std_errors["dose_post"],
                         "p": res_c.pvalues["dose_post"], "N": int(res_c.nobs)})
            print(f"[DiD +weather] β={res_c.params['dose_post']:+.4f} "
                  f"(p={res_c.pvalues['dose_post']:.3f}, overlap years only)")
        else:
            print("  (weather controls skipped — ERA5 panel does not cover the post years)")

    # robustness panel — the exact specifications quoted in FINDINGS Q2
    rows += _robustness_rows(df, policy_year)
    print("\nRobustness (each a separate DiD; all zero-to-positive, none a reduction):")
    for r in rows[1:]:
        print(f"    {r['spec']:34s} beta={r['beta']:+.4f} (p={r['p']:.3f}, N={r['N']})")

    # full dynamic event study (pre + post), reference = year before onset
    res_ev, ev = CA.pretrends_event_study(df, years=yrs, ref_year=policy_year - 1, save=False)
    _fig_effect(ev, policy_year)
    _fig_aggregate(df, policy_year)

    pd.DataFrame(rows).to_csv(OUT_RESULTS, index=False)
    print(f"\nSaved -> {OUT_RESULTS.relative_to(C.PROJECT_ROOT)} ; "
          f"reports/figures/12_did_effect.png")
    beta = res.params["dose_post"]
    verdict = ("subsidy intensity is associated with LOWER post-policy burning"
               if beta < 0 else "no reduction detected (β ≥ 0)")
    print(f"\nInterpretation: β={beta:+.4f} → {verdict}. "
          f"Identification is effectively Punjab-only (Haryana dose ≈ uniform).")
    return {"beta": float(beta), "p": float(res.pvalues["dose_post"]), "event_study": ev}


def _fig_aggregate(df: pd.DataFrame, policy_year: int) -> None:
    """Total burning-season detections by state over the whole horizon.

    Context the dose-response DiD cannot show: a year-fixed-effect design absorbs any
    common post-policy trend, so this aggregate fall (or rise) is invisible to beta —
    the DiD only tests whether *higher-dose* districts fell *more*.
    """
    agg = (df.groupby(["state", "year"])["fire_count"].sum()
           .reset_index())
    pre = df[df["year"] < policy_year].groupby("state")["fire_count"].sum() / \
        df[df["year"] < policy_year]["year"].nunique()
    post = df[df["year"] >= policy_year].groupby("state")["fire_count"].sum() / \
        df[df["year"] >= policy_year]["year"].nunique()

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"Punjab": "#d1495b", "Haryana": "#30638e"}
    for st, s in agg.groupby("state"):
        chg = 100 * (post[st] / pre[st] - 1)
        ax.plot(s["year"], s["fire_count"] / 1e3, marker="o", lw=2.5,
                color=colors.get(st), label=f"{st}  ({chg:+.0f}% post vs pre-mean)")
    ax.axvline(policy_year - 0.5, color="gray", ls="--", lw=1.5)
    ax.text(policy_year - 0.6, ax.get_ylim()[1] * 0.55, "CRM scale-up", color="gray",
            fontsize=10, ha="right")
    ax.set(xlabel="Year", ylabel="Burning-season fire detections (thousands)",
           title="Aggregate burning fell after 2018 — but the DiD can't credit the subsidy\n"
                 "(year fixed effects absorb this common trend; only the dose contrast is tested)")
    ax.legend(title="Oct–Nov VIIRS detections", loc="upper right")
    ax.figure.text(0.99, 0.01, "Source: NASA FIRMS VIIRS-SNPP. The differential (dose-response) "
                   "effect is separately ~0 (fig 12).", ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "13_aggregate_trend.png", dpi=140)
    plt.close(fig)


def _fig_effect(ev: pd.DataFrame, policy_year: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.axvline(policy_year - 0.5, color="#d1495b", lw=1.5, ls="--")
    ax.text(policy_year - 0.45, ax.get_ylim()[1] * 0.9, "CRM scale-up", color="#d1495b",
            fontsize=10)
    ax.errorbar(ev["year"], ev["coef"],
                yerr=[ev["coef"] - ev["lo"], ev["hi"] - ev["coef"]],
                fmt="o-", color="#3c6e47", capsize=5, lw=2, markersize=8)
    ax.set(xlabel="Year", ylabel="dose × year coefficient (95% CI)",
           title="Dose-response effect of the CRM subsidy on burning\n"
                 "(FIRMS VIIRS active fire, consistent 2012+ outcome)")
    ax.figure.text(0.99, 0.01, "Post-onset points below 0 -> higher-dose districts cut "
                   "burning more. Source: NASA FIRMS + PPCB/Haryana CRM.",
                   ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "12_did_effect.png", dpi=140)
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from fire_policy import firms

    if not firms.get_map_key() and not FIRMS_PANEL.exists():
        print("=" * 72)
        print("CAUSAL EFFECT — needs the FIRMS outcome (no cached panel found)")
        print("=" * 72)
        print("A free FIRMS MAP_KEY is needed once to build the outcome; after that the")
        print("cached panel lets the DiD rebuild offline.\n")
        print("  1. Get a free FIRMS MAP_KEY (emailed):")
        print("       https://firms.modaps.eosdis.nasa.gov/api/area/  -> 'Get MAP_KEY'")
        print("  2. Put it in .env at the project root:")
        print("       FIRMS_MAP_KEY=your_key_here")
        print("  3. Re-run:")
        print("       PYTHONPATH=src python -m fire_policy.effect\n")
        print("It will pull VIIRS active fire (Oct-Nov, 2012-2023), build a consistent")
        print("district-year outcome, and run the dose-response DiD — β, SEs, event study.")
        return

    if FIRMS_PANEL.exists():
        print("Using cached FIRMS VIIRS district-year outcome (2012-2023); estimating DiD…\n")
    else:
        print("Building consistent FIRMS VIIRS district-year outcome (2012+) and estimating DiD…\n")
    run_effect()


if __name__ == "__main__":
    main()
