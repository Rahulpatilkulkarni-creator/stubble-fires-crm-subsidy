"""Streamlit dashboard for the stubble-burning / crop-residue-subsidy project.

Everything here reads the processed panels the pipeline already wrote (data/processed),
so the app is just a view layer: no model training, no network calls at runtime.

Run it with the project venv:

    .venv/Scripts/python -m streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"

st.set_page_config(page_title="Stubble fires & the crop-residue subsidy",
                   page_icon="🔥", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PROC / "predictions_district_week.csv")


@st.cache_data(show_spinner=False)
def load_targeting() -> pd.DataFrame:
    return pd.read_csv(PROC / "causal_targeting.csv")


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    with open(PROC / "pb_hr_districts.geojson", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_effect() -> pd.DataFrame | None:
    """The DiD estimate. Only exists once effect.py has been run with a FIRMS key."""
    fp = PROC / "causal_effect_firms.csv"
    return pd.read_csv(fp) if fp.exists() else None


def fig_path(name: str) -> Path | None:
    p = FIGS / name
    return p if p.exists() else None


def skill_metrics(d: pd.DataFrame) -> dict:
    """R2, Spearman, top-decile ROC-AUC and precision@k over a set of district-weeks."""
    out = {"r2": float("nan"), "spearman": float("nan"),
           "auc": float("nan"), "prec": float("nan")}
    if len(d) < 5:
        return out
    y, yhat = d["log_dm"].to_numpy(float), d["pred_log"].to_numpy(float)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot > 0:
        out["r2"] = 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot
    out["spearman"] = float(pd.Series(yhat).corr(pd.Series(y), method="spearman"))
    thr = float(np.quantile(y, 0.90))
    hot = (y >= thr).astype(int)
    if 0 < hot.sum() < len(hot):
        try:
            from sklearn.metrics import roc_auc_score
            out["auc"] = float(roc_auc_score(hot, yhat))
        except Exception:
            pass
        k = int(hot.sum())
        top_k = np.argsort(-yhat)[:k]
        out["prec"] = float(hot[top_k].mean())
    return out


# --------------------------------------------------------------------------- #
# Sidebar — a bit of context so the app feels owned, not auto-generated
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### About")
    st.write(
        "A portfolio project on the paddy-stubble fires in Punjab and Haryana, and the "
        "crop-residue-management (CRM) subsidy meant to curb them."
    )
    st.write(
        "I built it from raw sources: SAGE-IGP satellite fire grids, ERA5 weather, and "
        "the actual government subsidy records (PPCB and Haryana CRM plans). No "
        "pre-cleaned analysis file."
    )
    st.caption("The full write-up is in reports/FINDINGS.md; the code is in src/fire_policy/.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Stubble fires and the crop-residue subsidy")
st.write(
    "Every autumn, farmers across Punjab and Haryana get two or three weeks to clear rice "
    "stubble before sowing wheat, and the cheapest way to do it is to set it alight. That "
    "smoke is a big part of why Delhi's air turns hazardous every November. I wanted to "
    "answer two things with the raw data: can you see next season's hotspots coming, and "
    "did the machinery subsidy actually reduce the burning?"
)

pred = load_predictions()
geo = load_geojson()
DISTRICTS = sorted({f["properties"]["district"] for f in geo["features"]})

tab_pred, tab_causal, tab_about = st.tabs(
    ["Forecasting the hotspots", "Did the subsidy work?", "Data & method"]
)

# =========================================================================== #
# TAB 1 — prediction
# =========================================================================== #
with tab_pred:
    st.subheader("Week-ahead hotspot forecast")
    st.write(
        "I model burning at the district-by-ISO-week level over the Oct-Nov season, "
        "training on 2012-16 and holding out 2017-18. The signal comes almost entirely "
        "from fire persistence (what burned last week), the crop calendar, and which "
        "district it is. Weather turns out to barely matter here: adding an 8-variable "
        "ERA5 block changes R² by about +0.004. Everything below is the held-out period."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    year = c1.selectbox("Season", sorted(pred["year"].unique()),
                        index=len(pred["year"].unique()) - 1)
    weeks = sorted(pred.loc[pred["year"] == year, "iso_week"].unique())
    # week 44-45 is the early-November peak; start there if it's in range
    default_wk = 44 if 44 in weeks else weeks[len(weeks) // 2]
    week = c2.select_slider("ISO week", options=weeks, value=default_wk)
    view = c3.radio("Layer", ["Predicted", "Actual"], horizontal=True,
                    help="Predicted is the model's week-ahead call; Actual is the observed SAGE burned mass.")

    val_col = "pred_tonnes" if view == "Predicted" else "dm_tonnes"

    # fill out all 43 districts so the map isn't patchy on quiet weeks
    slc = pred[(pred["year"] == year) & (pred["iso_week"] == week)]
    full = pd.DataFrame({"district": DISTRICTS}).merge(slc, on="district", how="left")
    full["value"] = full[val_col].fillna(0.0)
    full["value_log"] = np.log1p(full["value"])
    # keep the colour scale fixed across weeks so the map is comparable week to week
    gmax = np.log1p(pred.loc[pred["year"] == year, val_col].max())

    mc, wc = st.columns([3, 2])
    with mc:
        fig = px.choropleth_map(
            full, geojson=geo, locations="district", featureidkey="properties.district",
            color="value_log", range_color=(0, gmax),
            hover_name="district",
            hover_data={"value": ":,.0f", "value_log": False},
            color_continuous_scale="YlOrRd", map_style="carto-positron",
            center={"lat": 30.5, "lon": 75.9}, zoom=5.7, opacity=0.78, height=560,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_colorbar=dict(title=f"{view}<br>tonnes (log)"))
        fig.update_traces(marker_line_width=0.4, marker_line_color="white")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{view} burned mass, {year} week {week}. Hover for tonnes; grey districts had none.")

    with wc:
        st.markdown(f"**Top predicted districts, week {week}**")
        wl = slc.sort_values("pred_tonnes", ascending=False).head(10).copy()
        if not wl.empty:
            # did each predicted-hot district actually land in the week's top decile?
            thr = slc["dm_tonnes"].quantile(0.90)
            wl["top decile"] = wl["dm_tonnes"] >= thr
            show = wl[["district", "state", "pred_tonnes", "dm_tonnes", "top decile"]].rename(
                columns={"pred_tonnes": "predicted t", "dm_tonnes": "actual t"})
            st.dataframe(
                show.style.format({"predicted t": "{:,.0f}", "actual t": "{:,.0f}"}),
                hide_index=True, width="stretch", height=396,
            )
            st.caption("A rough watchlist: where I'd send inspectors that week. "
                       "'Top decile' flags whether it really was among the worst.")
        else:
            st.info("No rows for this week.")

    st.markdown("##### How good is it, over the whole season?")
    d = pred[pred["year"] == year].dropna(subset=["pred_log", "log_dm"])
    m = skill_metrics(d)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² (log)", f"{m['r2']:.2f}")
    k2.metric("Spearman", f"{m['spearman']:.2f}")
    k3.metric("ROC-AUC (top 10%)", f"{m['auc']:.2f}")
    k4.metric("Precision@k", f"{m['prec']:.2f}")
    st.caption("These are recomputed live from the predictions file each time you switch "
               "seasons. ROC-AUC and precision are about catching the worst-decile "
               "district-weeks, which is really what an early-warning tool has to get right.")

    with st.expander("Why weather barely helps (I checked)"):
        st.write(
            "I ran the model with and without an 8-variable ERA5 weather block. It moves "
            "the operational model by about +0.004 R², and actually hurts the "
            "no-persistence version by 0.09. That fits the story: the decision to burn is "
            "driven by the harvest calendar and economics, not the weather. Weather only "
            "decides whether a fire can be lit, not where the burning concentrates. So an "
            "early-warning tool needs last week's fire counts and the calendar, not a "
            "weather feed."
        )
        fi = fig_path("06_feature_importance.png")
        if fi:
            st.image(str(fi), caption="Feature importance: persistence, district and calendar dominate.")

# =========================================================================== #
# TAB 2 — causal
# =========================================================================== #
with tab_causal:
    st.subheader("Did the subsidy reduce burning?")
    st.write(
        "This is a dose-response difference-in-differences. Treatment isn't on/off; it's "
        "how many CRM machines a district received, standardized to a within-state z-score "
        "so Punjab's actual counts and Haryana's targets sit on the same scale. The model "
        "has district and year fixed effects, with standard errors clustered by district."
    )

    tg = load_targeting()

    st.markdown("##### The money went to the worst-burning districts")

    def _sp(sub):
        return sub["dose_z"].corr(sub["pre_log_dm"], method="spearman")
    sp_pb = _sp(tg[tg["is_punjab"] == 1])
    sp_hr = _sp(tg[tg["is_punjab"] == 0])

    sc1, sc2 = st.columns([3, 1])
    with sc1:
        tg2 = tg.copy()
        tg2["state"] = np.where(tg2["is_punjab"] == 1, "Punjab", "Haryana")
        fig = px.scatter(
            tg2, x="dose_z", y="pre_log_dm", color="state",
            size=tg2["crm_machines"].clip(lower=1), hover_name="district",
            color_discrete_map={"Punjab": "#d1495b", "Haryana": "#3c6e47"},
            labels={"dose_z": "CRM intensity (within-state z-score)",
                    "pre_log_dm": "pre-period burning (log dry matter)"},
            height=460,
        )
        b, a = np.polyfit(tg2["dose_z"], tg2["pre_log_dm"], 1)
        xs = np.linspace(tg2["dose_z"].min(), tg2["dose_z"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name="fit",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title="")
        st.plotly_chart(fig, width="stretch")
    with sc2:
        st.metric("Rank corr., Punjab", f"+{sp_pb:.2f}")
        st.metric("Rank corr., Haryana", f"+{sp_hr:.2f}")
        st.write(
            "Machines went where burning was already highest. Sensible as policy, but a "
            "problem for identification: a plain treated-vs-control comparison would make "
            "the subsidy look like it *caused* the fires it was sent to fight. That's why "
            "I use a within-district design that nets out each district's baseline."
        )

    st.divider()
    st.markdown("##### The pre-trends are only *partly* clean — a caveat I won't hide")
    ev1, ev2 = st.columns(2)
    p10 = fig_path("10_pretrends_eventstudy.png")
    p11 = fig_path("11_highlow_trajectories.png")
    if p10:
        ev1.image(str(p10), caption="Pooled pre-period event study (SAGE): dose-by-year "
                                    "coefficients mostly span zero (joint p = 0.18; p = 0.08 on the FIRMS outcome).")
    if p11:
        ev2.image(str(p11), caption="High- and low-dose districts track each other before the policy.")
    st.write(
        "Pooled across both states the pre-trends look parallel. But within **Punjab** — where "
        "the dose actually varies — the joint pre-trend test **fails (p = 0.005)**: the "
        "highest-intensity districts were already on a steeper burning path before the policy. "
        "That's a real limit on the causal reading below. The placebos, at least, behave: a "
        "fake 2015 onset gives β = −0.16 (p = 0.13) and the 2018 onset β = +0.08 (p = 0.62), so "
        "the estimator isn't inventing effects where there shouldn't be any."
    )

    st.divider()
    st.markdown("##### The effect: no dose-response reduction")
    eff = load_effect()
    if eff is not None and len(eff):
        row = eff.iloc[0]
        b, p, n = float(row["beta"]), float(row["p"]), int(row["N"])
        m1, m2, m3 = st.columns(3)
        m1.metric("dose × post (β)", f"{b:+.3f}")
        m2.metric("p-value", f"{p:.2f}")
        m3.metric("district-years", f"{n}")
        if b < 0 and p < 0.05:
            st.write(f"β = {b:+.3f} (p = {p:.2f}). Higher-intensity districts cut their burning "
                     f"more than lower-intensity ones after the subsidy scaled up.")
        else:
            st.write(
                f"β = {b:+.3f} (p = {p:.2f}), on a consistent FIRMS fire outcome spanning "
                "2012–2023. There's **no dose-response reduction** — the estimate is a precise "
                "zero, if anything faintly positive, and stays zero-to-positive across every "
                "specification I tried (Punjab-only, dropping the 2020–21 COVID/protest years, "
                "FRP intensity, binary treatment). The 95% CI rules out reductions bigger than "
                "about 10% per standard deviation of dose, but can't separate a small effect "
                "from none."
            )
        ec1, ec2 = st.columns(2)
        f12 = fig_path("12_did_effect.png")
        if f12:
            ec1.image(str(f12), caption="Dose-response effect by year: post-2018 coefficients "
                                        "bounce around zero, with no downward drift as machines accumulate.")
        f13 = fig_path("13_aggregate_trend.png")
        if f13:
            ec2.image(str(f13), caption="Aggregate burning did fall after 2018 (Haryana −44%, "
                                        "Punjab −18%) — but year effects absorb that, so the DiD can't credit the subsidy.")
        st.write(
            "The honest reading is two-sided: **burning fell** across both states after the "
            "scale-up, but **not in proportion to how many machines a district received** — the "
            "only thing this design can actually test. That aggregate decline is equally "
            "consistent with enforcement, straw-market alternatives, weather, or the subsidy "
            "working uniformly; what the data rule out is the specific *more machines here → "
            "less fire here* mechanism."
        )
    else:
        st.info(
            "The DiD estimate (`causal_effect_firms.csv`) isn't present. Rebuild it with a free "
            "FIRMS key: `FIRMS_MAP_KEY=... python -m fire_policy.effect` pulls a consistent VIIRS "
            "fire outcome across 2012–2023 and runs the estimator."
        )
    st.warning(
        "The caveat to keep in mind before quoting the number: identification is effectively "
        "Punjab-only (Haryana's doses are near-uniform, ~403 machines per district), and within "
        "Punjab the pre-trends fail (p = 0.005) — the highest-dose districts were already "
        "trending differently. So this is best read as *no evidence of a dose-response effect*, "
        "not a clean causal zero."
    )

# =========================================================================== #
# TAB 3 — data & method
# =========================================================================== #
with tab_about:
    st.subheader("Where the data comes from")
    st.write("Everything is built from raw sources and primary records. There's no "
             "ready-made analysis CSV underneath this.")
    st.markdown(
        "| Layer | Source | Coverage | Access |\n"
        "|---|---|---|---|\n"
        "| Fire (burned mass) | SAGE-IGP (Harvard Dataverse, CC0) | 2003-2018, 0.25° daily | keyless |\n"
        "| Weather | Open-Meteo ERA5 | 2012-2018, daily to weekly | keyless |\n"
        "| Treatment (CRM machines) | PPCB MIS + Haryana CRM plan | 2018-2022, district | primary PDFs |\n"
        "| Geography | geoBoundaries ADM2 (CC-BY) | 43 districts | keyless |\n"
        "| Post-2018 fire (causal outcome) | NASA FIRMS VIIRS-SNPP archive | 2012-2023, district | free key |\n"
    )
    st.write(
        "One thing worth calling out: the SAGE fire inventory ends exactly at 2018, the "
        "treatment year — fine for forecasting (train 2012-16, test 2017-18) and every "
        "pre-period check, but with no post-treatment outcome of its own. Rather than splice "
        "two products at the treatment boundary, the causal leg builds a single, consistent "
        "FIRMS VIIRS fire outcome across 2012–2023 (one free key, used once) and runs the DiD "
        "on that."
    )

    st.markdown("##### A couple of descriptive views")
    a1, a2 = st.columns(2)
    for col, name, cap in [
        (a1, "01_state_trend.png", "Punjab is about 86% of the two-state total."),
        (a2, "04_seasonal_timing.png", "The burning spikes hard in ISO weeks 44-45, early November."),
    ]:
        p = fig_path(name)
        if p:
            col.image(str(p), caption=cap)

    hp = ROOT / "reports" / "hotspot_map.html"
    if hp.exists():
        with st.expander("Interactive hotspot map"):
            st.iframe(hp, height=560)

    st.markdown("##### Reproducing it")
    st.write("Each stage writes to data/processed and can be re-run on its own:")
    st.code(
        "python -m fire_policy.geo         # build the district layer\n"
        "python -m fire_policy.sage        # fire panels from the SAGE grids\n"
        "python -m fire_policy.weather     # ERA5 weekly weather\n"
        "python -m fire_policy.treatment   # the CRM dose panel\n"
        "python -m fire_policy.eda         # descriptive figures\n"
        "python -m fire_policy.predict     # the forecast model + weather ablation\n"
        "python -m fire_policy.causal      # targeting, pre-trends, placebos\n"
        "python -m fire_policy.effect      # the DiD effect (needs a FIRMS key)",
        language="bash",
    )
    st.caption("Known limitations: the fire grid is coarse (0.25°) for small districts; "
               "Punjab's treatment is actual machines while Haryana's is one-year targets; "
               "and, as noted, the cross-district variation is mostly Punjab's.")
