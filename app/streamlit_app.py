"""Interactive dashboard for the stubble-fire × CRM-subsidy project.

Two questions, two tabs, plus an about page — all reading the *processed* panels this
pipeline produces (no re-computation, no network):

  1. "Where next?"  — the district×week early-warning model: an interactive hotspot map
     (predicted vs actual), a watchlist, and live skill metrics.
  2. "Did it work?" — the dose-response DiD story: the targeting confound, parallel
     pre-trends, and the honest status of the effect estimate (Punjab-only; one FIRMS key
     from a real number).

Run from the project root:

    PYTHONPATH=src .venv/Scripts/python -m streamlit run app/streamlit_app.py
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

st.set_page_config(page_title="Stubble fires vs. the CRM subsidy",
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
def load_treatment() -> pd.DataFrame:
    return pd.read_csv(PROC / "treatment_district.csv")


@st.cache_data(show_spinner=False)
def load_geojson() -> dict:
    with open(PROC / "pb_hr_districts.geojson", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_effect() -> pd.DataFrame | None:
    """The real DiD estimate — present only after a FIRMS key has been supplied."""
    fp = PROC / "causal_effect_firms.csv"
    return pd.read_csv(fp) if fp.exists() else None


def fig_path(name: str) -> Path | None:
    p = FIGS / name
    return p if p.exists() else None


def _skill_metrics(d: pd.DataFrame) -> dict:
    """R², Spearman, top-decile ROC-AUC and precision@k over a set of district-weeks."""
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


DISTRICTS = None  # populated after geojson load


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🔥 Stubble fires vs. the crop-residue subsidy")
st.markdown(
    "**Punjab & Haryana, the fires behind Delhi's November smog.** "
    "Two questions from raw satellite grids and primary subsidy records: *can we predict "
    "next season's hotspots*, and *did the equipment subsidy actually put them out?* "
    "See [`reports/FINDINGS.md`](https://github.com/) for the full write-up."
)

pred = load_predictions()
geo = load_geojson()
DISTRICTS = sorted({f["properties"]["district"] for f in geo["features"]})

tab_pred, tab_causal, tab_about = st.tabs(
    ["🔥  Where next? — early warning",
     "📉  Did it work? — the causal design",
     "📄  Data & method"]
)

# =========================================================================== #
# TAB 1 — Early warning
# =========================================================================== #
with tab_pred:
    st.subheader("Week-ahead hotspot early warning")
    st.caption(
        "A district×week LightGBM model (train 2012–16, test 2017–18) on the Oct–Nov "
        "burning window. It runs on fire *persistence* + the crop calendar + district "
        "identity — **not** weather (an 8-variable ERA5 block adds ΔR² = +0.004). "
        "Everything below is the held-out test period."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    year = c1.selectbox("Season (year)", sorted(pred["year"].unique()), index=len(pred["year"].unique()) - 1)
    weeks = sorted(pred.loc[pred["year"] == year, "iso_week"].unique())
    # ISO week 44–45 is the early-November peak — default the slider there if present.
    default_wk = 44 if 44 in weeks else weeks[len(weeks) // 2]
    week = c2.select_slider("ISO week", options=weeks, value=default_wk)
    view = c3.radio("Map layer", ["Predicted", "Actual"], horizontal=True,
                    help="Predicted = the model's week-ahead nowcast; Actual = observed SAGE burned mass.")

    val_col = "pred_tonnes" if view == "Predicted" else "dm_tonnes"

    # slice + complete to all 43 districts so the map is never patchy
    slc = pred[(pred["year"] == year) & (pred["iso_week"] == week)]
    full = pd.DataFrame({"district": DISTRICTS}).merge(slc, on="district", how="left")
    full["value"] = full[val_col].fillna(0.0)
    full["value_log"] = np.log1p(full["value"])
    # stable color range across weeks (global max of the chosen layer, this year)
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
        st.caption(f"**{view} burned mass — {year}, ISO week {week}.** "
                   "Hover for tonnes. Grey = no data / zero.")

    with wc:
        st.markdown(f"**🚨 Watchlist — top districts, week {week}**")
        wl = slc.sort_values("pred_tonnes", ascending=False).head(10).copy()
        if not wl.empty:
            # flag whether each predicted-hot district is *actually* in the week's top decile
            thr = slc["dm_tonnes"].quantile(0.90)
            wl["actually hot"] = np.where(wl["dm_tonnes"] >= thr, "✅", "—")
            show = wl[["district", "state", "pred_tonnes", "dm_tonnes", "actually hot"]].rename(
                columns={"pred_tonnes": "predicted t", "dm_tonnes": "actual t"})
            st.dataframe(
                show.style.format({"predicted t": "{:,.0f}", "actual t": "{:,.0f}"}),
                hide_index=True, width="stretch", height=396,
            )
        else:
            st.info("No rows for this week.")

    # ---- live skill metrics for the whole selected season -------------------
    st.markdown("##### Model skill this season (all weeks)")
    d = pred[pred["year"] == year].dropna(subset=["pred_log", "log_dm"])
    m = _skill_metrics(d)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R² (log)", f"{m['r2']:.2f}")
    k2.metric("Spearman", f"{m['spearman']:.2f}")
    k3.metric("ROC-AUC (top-10%)", f"{m['auc']:.2f}")
    k4.metric("Precision@k", f"{m['prec']:.2f}")
    st.caption(
        "Computed live from `predictions_district_week.csv`. ROC-AUC / precision are for "
        "catching each season's worst-decile district-weeks — the operational target."
    )

    with st.expander("Why weather barely helps (the ablation)"):
        st.markdown(
            "Adding an 8-variable ERA5 weather block moves the operational model by "
            "**ΔR² = +0.004** and *hurts* the no-persistence structural model by **−0.09**. "
            "The burn decision tracks the harvest calendar and economics; weather only gates "
            "whether a fire *can* be lit. An operational dashboard needs last week's fire "
            "counts and the calendar — not a weather feed."
        )
        fi = fig_path("06_feature_importance.png")
        if fi:
            st.image(str(fi), caption="Feature importance — persistence + calendar + district dominate.")

# =========================================================================== #
# TAB 2 — Causal design
# =========================================================================== #
with tab_causal:
    st.subheader("Did the subsidy reduce burning?")
    st.caption(
        "A **dose-response difference-in-differences**: treatment is a *continuous* CRM "
        "intensity (machines per district, harmonised to a within-state z-score), not an "
        "on/off switch. District + year fixed effects, district-clustered SEs."
    )

    tg = load_targeting()

    st.markdown("##### 1 · The subsidy was aimed at the worst-burning districts")
    # within-state Spearman, computed live
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
            labels={"dose_z": "CRM intensity  (within-state z-score)",
                    "pre_log_dm": "pre-period burning  (log dry matter)"},
            height=460,
        )
        # overall OLS guide line (numpy — no statsmodels dependency)
        b, a = np.polyfit(tg2["dose_z"], tg2["pre_log_dm"], 1)
        xs = np.linspace(tg2["dose_z"].min(), tg2["dose_z"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name="overall fit",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title="")
        st.plotly_chart(fig, width="stretch")
    with sc2:
        st.metric("Spearman — Punjab", f"+{sp_pb:.2f}")
        st.metric("Spearman — Haryana", f"+{sp_hr:.2f}")
        st.markdown(
            "Machines flowed to districts already burning most. **Good targeting, but a "
            "textbook confound** — a naive treated-vs-untreated comparison would blame the "
            "subsidy for the fires it was *sent to fight*. Hence a within-district DiD."
        )

    st.divider()
    st.markdown("##### 2 · Parallel pre-trends hold — the design is credible")
    ev1, ev2 = st.columns(2)
    p10 = fig_path("10_pretrends_eventstudy.png")
    p11 = fig_path("11_highlow_trajectories.png")
    if p10:
        ev1.image(str(p10), caption="Event study, clean pre-period (2012–17). Every dose×year "
                                    "CI spans zero; joint pre-trend test p = 0.18.")
    if p11:
        ev2.image(str(p11), caption="High- vs low-dose district trajectories move together "
                                    "before the policy.")
    st.markdown(
        "Placebo DiDs return null (fake-2015 onset β = −0.16, p = 0.13; the 2018 onset "
        "before machines deployed β = +0.08, p = 0.62). The estimator does **not** "
        "manufacture an effect where none should exist."
    )

    st.divider()
    st.markdown("##### 3 · The effect estimate")
    eff = load_effect()
    if eff is not None and len(eff):
        row = eff.iloc[0]
        b, p, n = float(row["beta"]), float(row["p"]), int(row["N"])
        m1, m2, m3 = st.columns(3)
        m1.metric("dose × post  (β)", f"{b:+.3f}")
        m2.metric("p-value", f"{p:.3f}")
        m3.metric("N (district-years)", f"{n}")
        verdict = ("higher-intensity districts cut burning **more** after the subsidy"
                   if b < 0 else "**no reduction** detected (β ≥ 0)")
        st.success(f"β = {b:+.3f} → {verdict}.")
        f12 = fig_path("12_did_effect.png")
        if f12:
            st.image(str(f12), caption="Dynamic dose-response effect (consistent FIRMS 2012+ outcome).")
    else:
        st.warning(
            "**Effect estimate pending one credential.** SAGE-IGP stops in 2018 — the "
            "subsidy's scale-up year — so the keyless record has no post-treatment outcome. "
            "`fire_policy/effect.py` rebuilds a *consistent* NASA FIRMS VIIRS outcome across "
            "2012+ (not a SAGE→FIRMS splice) and runs the same estimator. The analytic path "
            "is verified end-to-end on synthetic data (recovers a known β). It needs a free "
            "FIRMS `MAP_KEY` (emailed on signup), then one command:\n\n"
            "```bash\nFIRMS_MAP_KEY=<key>  PYTHONPATH=src python -m fire_policy.effect\n```"
        )
    st.info(
        "⚠️ **Read this before quoting any effect:** the identifying variation is "
        "**effectively Punjab-only**. Haryana's dose is near-uniform (one-year targets "
        "≈403/district), so it contributes almost no cross-sectional intensity contrast. "
        "The estimate really asks *“did Punjab's higher-intensity districts cut burning "
        "more?”* — not a clean two-state average."
    )

# =========================================================================== #
# TAB 3 — Data & method
# =========================================================================== #
with tab_about:
    st.subheader("Data & method")
    st.markdown(
        "Built end-to-end from **raw satellite grids and primary government records** — "
        "no pre-cleaned analysis CSV."
    )
    st.markdown(
        "| Layer | Source | Coverage | Access |\n"
        "|---|---|---|---|\n"
        "| Fire (burned mass) | **SAGE-IGP** (Harvard Dataverse, CC0) | 2003–2018, 0.25° daily | keyless |\n"
        "| Weather | **Open-Meteo ERA5** | 2012–2018, daily→weekly | keyless |\n"
        "| Treatment (CRM machines) | **PPCB MIS** + **Haryana CRM plan** | 2018–2022, district | primary PDFs |\n"
        "| Geography | **geoBoundaries** ADM2 (CC-BY) | 43 districts | keyless |\n"
        "| *Post-2018 fire (to finish DiD)* | *NASA FIRMS Area API* | *2012+* | *free `MAP_KEY` (emailed)* |\n"
    )
    st.markdown(
        "**The 2018 wall.** The keyless fire inventory ends exactly at the treatment year. "
        "That is fine for prediction (train 2012–16, test 2017–18) and every *pre-period* "
        "causal check; the post-treatment outcome is the one thing gated on a credential."
    )

    st.markdown("##### Where the burning is (descriptive)")
    a1, a2 = st.columns(2)
    for col, name, cap in [
        (a1, "01_state_trend.png", "Punjab is 86% of the two-state total."),
        (a2, "04_seasonal_timing.png", "Sharp peak in ISO weeks 44–45 (early November)."),
    ]:
        p = fig_path(name)
        if p:
            col.image(str(p), caption=cap)

    hp = ROOT / "reports" / "hotspot_map.html"
    if hp.exists():
        with st.expander("Interactive hotspot map (folium)"):
            st.iframe(hp, height=560)

    st.markdown("##### Reproduce")
    st.code(
        "python -m fire_policy.geo         # district layer\n"
        "python -m fire_policy.sage        # fire panels\n"
        "python -m fire_policy.weather     # ERA5 weekly weather\n"
        "python -m fire_policy.treatment   # harmonised CRM dose\n"
        "python -m fire_policy.eda         # figures 01–04\n"
        "python -m fire_policy.predict     # figures 05–08 + ablation\n"
        "python -m fire_policy.causal      # figures 09–11 + DiD checks\n"
        "python -m fire_policy.effect      # figure 12 + effect (needs FIRMS key)",
        language="bash",
    )
    st.caption("Honest limitations: 0.25° fire resolution is coarse for small districts; "
               "treatment is actual machines (Punjab) vs one-year targets (Haryana); the "
               "causal leg is a validated design + verified estimator, awaiting the FIRMS key.")
