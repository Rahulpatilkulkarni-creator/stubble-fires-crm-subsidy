"""Spatiotemporal early-warning model for stubble-burning hotspots.

The EDA showed seasonal weather barely correlates with seasonal burning, but the
*timing* is sharply weekly (burning peaks in ISO weeks 44-45). So we model at
district x week resolution and ask two honest questions:

  Model B  "structural / pre-season":  predict weekly district burning from weather +
           calendar + geography + district climatology, with NO within-season fire
           lags. How predictable is the hotspot pattern from drivers known ahead of
           time?

  Model A  "operational / week-ahead":  add autoregressive fire lags (last week's
           observed burning, season-to-date, same week last year) -- the real
           early-warning nowcast, run mid-season with last week's fire counts in hand.

Both are compared against a climatology baseline (each district-week's historical
average from the training years) under an honest forward split: train 2012-2016,
test 2017-2018. Target is log1p(dry-matter-burned in tonnes); metrics are reported
on both the log scale (RMSE/MAE/R2) and in the operational ranking sense
(rank correlation, ROC-AUC and precision@k for catching the worst district-weeks).

Inputs (built by sage.build_sage_weekly_panel + weather.build_weather_weekly_panel):
  data/processed/fire_sage_district_week.csv
  data/processed/weather_district_week.csv
Outputs: reports/figures/05..08_*.png and data/processed/predictions_district_week.csv
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fire_policy import config as C
from fire_policy.geo import district_centroids

sns.set_theme(style="whitegrid", context="talk")

FIRE_WEEK = C.PROCESSED_DIR / "fire_sage_district_week.csv"
WEATHER_WEEK = C.PROCESSED_DIR / "weather_district_week.csv"

TRAIN_YEARS = (2012, 2013, 2014, 2015, 2016)
TEST_YEARS = (2017, 2018)
VALID_YEAR = 2016  # last train year, held out for early stopping

WEATHER_FEATS = ["tmax_mean", "tmin_mean", "precip_sum", "rain_days", "dry_days",
                 "wind_max_mean", "wind_calm_days", "et0_sum"]
WEATHER_LAGS = ["precip_lag1", "dry_days_lag1", "et0_lag1", "cum_dry_days", "cum_precip"]
CALENDAR = ["iso_week", "week_of_season", "week_sin", "week_cos"]
SPATIAL = ["lat", "lon", "is_punjab", "district"]
CLIM = ["district_clim", "week_clim"]
FIRE_LAGS = ["dm_lag1", "dm_lag2", "dm_cum", "dm_lastyear"]

BASE_FEATURES = WEATHER_FEATS + WEATHER_LAGS + CALENDAR + SPATIAL + CLIM
LAG_FEATURES = BASE_FEATURES + FIRE_LAGS


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def build_frame() -> pd.DataFrame:
    """Merge weekly fire + weather, engineer calendar / spatial / lag features."""
    fire = pd.read_csv(FIRE_WEEK)
    wx = pd.read_csv(WEATHER_WEEK) if WEATHER_WEEK.exists() else pd.DataFrame()

    df = fire.copy()
    df["log_dm"] = np.log1p(df["dm_tonnes"])

    if not wx.empty:
        df = df.merge(wx.drop(columns=[c for c in ("n_days",) if c in wx]),
                      on=["state", "district", "year", "iso_week"], how="left")

    # calendar --------------------------------------------------------------- #
    df["week_of_season"] = df["iso_week"] - 44          # centred on the peak week
    df["week_sin"] = np.sin(2 * np.pi * df["iso_week"] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df["iso_week"] / 52.0)

    # spatial ---------------------------------------------------------------- #
    cen = district_centroids()
    df = df.merge(cen, on=["state", "district"], how="left")
    df["is_punjab"] = (df["state"] == "Punjab").astype(int)

    df = df.sort_values(["district", "year", "iso_week"]).reset_index(drop=True)
    grp = df.groupby(["district", "year"], sort=False)

    # within-season fire lags (known mid-season) ----------------------------- #
    df["dm_lag1"] = grp["log_dm"].shift(1)
    df["dm_lag2"] = grp["log_dm"].shift(2)
    cum_incl = grp["dm_tonnes"].cumsum()
    df["dm_cum"] = np.log1p(cum_incl - df["dm_tonnes"])   # season-to-date, exclusive

    # same district & calendar-week, previous year
    prev = df[["district", "iso_week", "year", "log_dm"]].copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"log_dm": "dm_lastyear"})
    df = df.merge(prev, on=["district", "iso_week", "year"], how="left")

    # weather lags / cumulative dryness -------------------------------------- #
    for src, dst in [("precip_sum", "precip_lag1"), ("dry_days", "dry_days_lag1"),
                     ("et0_sum", "et0_lag1")]:
        df[dst] = (df.groupby(["district", "year"], sort=False)[src].shift(1)
                   if src in df else np.nan)
    df["cum_dry_days"] = (df.groupby(["district", "year"], sort=False)["dry_days"].cumsum()
                          if "dry_days" in df else np.nan)
    df["cum_precip"] = (df.groupby(["district", "year"], sort=False)["precip_sum"].cumsum()
                        if "precip_sum" in df else np.nan)

    df["district"] = df["district"].astype("category")
    return df


def add_climatology(df: pd.DataFrame, train_years=TRAIN_YEARS) -> pd.DataFrame:
    """District / week / district-week mean log-DM from TRAIN years only (leak-free).

    district_clim + week_clim are model features; dw_clim is the baseline predictor.
    """
    tr = df[df["year"].isin(train_years)]
    gmean = tr["log_dm"].mean()
    d_clim = tr.groupby("district", observed=True)["log_dm"].mean()
    w_clim = tr.groupby("iso_week")["log_dm"].mean()
    dw_clim = tr.groupby(["district", "iso_week"], observed=True)["log_dm"].mean()

    out = df.copy()
    out["district_clim"] = out["district"].map(d_clim).fillna(gmean)
    out["week_clim"] = out["iso_week"].map(w_clim).fillna(gmean)
    out = out.merge(dw_clim.rename("dw_clim").reset_index(),
                    on=["district", "iso_week"], how="left")
    out["dw_clim"] = out["dw_clim"].fillna(out["district_clim"]).fillna(gmean)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                  r2_score, roc_auc_score)
    y_log_pred = np.clip(y_log_pred, 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_log_true, y_log_pred)))
    mae = float(mean_absolute_error(y_log_true, y_log_pred))
    r2 = float(r2_score(y_log_true, y_log_pred))
    yt, yp = np.expm1(y_log_true), np.expm1(y_log_pred)
    rho = float(spearmanr(yt, yp).correlation)
    thr = np.quantile(yt, 0.90)
    hi = (yt >= thr).astype(int)
    if hi.sum() and hi.sum() < len(hi):
        auc = float(roc_auc_score(hi, yp))
        k = int(hi.sum())
        topk = np.argsort(-yp)[:k]
        prec_k = float(hi[topk].mean())
    else:
        auc, prec_k = float("nan"), float("nan")
    return {"RMSE_log": rmse, "MAE_log": mae, "R2_log": r2,
            "Spearman": rho, "ROC_AUC_top10pct": auc, "Precision@k": prec_k}


# --------------------------------------------------------------------------- #
# Train / evaluate one model
# --------------------------------------------------------------------------- #
def train_eval(df: pd.DataFrame, features: list[str], label: str,
               train_years=TRAIN_YEARS, test_years=TEST_YEARS, valid_year=VALID_YEAR):
    import lightgbm as lgb

    # Use only features actually present (weather may not be fetched yet -> ablation).
    feats = [f for f in features if f in df.columns]
    dropped = [f for f in features if f not in df.columns]
    if dropped:
        print(f"  [{label}] weather not present, dropping {len(dropped)} feats: "
              f"{dropped}")

    fit_years = [y for y in train_years if y != valid_year]
    tr = df[df["year"].isin(fit_years)]
    va = df[df["year"] == valid_year]
    te = df[df["year"].isin(test_years)].copy()

    Xtr, ytr = tr[feats], tr["log_dm"]
    Xva, yva = va[feats], va["log_dm"]
    Xte, yte = te[feats], te["log_dm"]

    cat = ["district"] if "district" in feats else "auto"
    model = lgb.LGBMRegressor(
        objective="regression", n_estimators=1200, learning_rate=0.03,
        num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l2",
              categorical_feature=cat,
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(0)])

    te["pred_log"] = model.predict(Xte, num_iteration=model.best_iteration_)
    m = _metrics(yte.to_numpy(), te["pred_log"].to_numpy())
    m["model"] = label
    m["best_iter"] = int(model.best_iteration_ or model.n_estimators)
    return model, te, m, feats


def eval_baseline(df: pd.DataFrame, test_years=TEST_YEARS) -> dict:
    te = df[df["year"].isin(test_years)]
    m = _metrics(te["log_dm"].to_numpy(), te["dw_clim"].to_numpy())
    m["model"] = "climatology"
    return m


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_pred_vs_actual(te_a: pd.DataFrame, te_b: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    for ax, te, title in [(axes[0], te_b, "Model B (structural, no fire lags)"),
                          (axes[1], te_a, "Model A (operational, +fire lags)")]:
        yt = np.expm1(te["log_dm"]) / 1e3
        yp = np.expm1(te["pred_log"]) / 1e3
        ax.scatter(yt, yp, alpha=0.35, s=28,
                   c=np.where(te["is_punjab"] == 1, "#d1495b", "#30638e"))
        hi = max(yt.max(), yp.max())
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set(xlabel="Actual DM (thousand t)", title=title)
    axes[0].set_ylabel("Predicted DM (thousand t)")
    fig.suptitle("Weekly district burning: predicted vs actual (test 2017-2018)", y=1.01)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c) for c in ("#d1495b", "#30638e")]
    axes[1].legend(handles, ["Punjab", "Haryana"], loc="upper left")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "05_pred_vs_actual.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_feature_importance(model, features: list[str]) -> None:
    gain = model.booster_.feature_importance(importance_type="gain")
    imp = (pd.Series(gain, index=features).sort_values(ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#3c6e47")
    ax.set(xlabel="Gain", title="What drives the week-ahead forecast? (Model A, top 15)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "06_feature_importance.png", dpi=140)
    plt.close(fig)


def fig_earlywarning_heatmap(te_a: pd.DataFrame, year: int = 2018, n: int = 18) -> None:
    d = te_a[te_a["year"] == year].copy()
    if d.empty:
        return
    order = (d.groupby("district", observed=True)["dm_tonnes"].sum()
             .sort_values(ascending=False).head(n).index)
    d = d[d["district"].isin(order)]
    act = (d.pivot_table(index="district", columns="iso_week",
                         values="log_dm", observed=True).reindex(order))
    prd = (d.pivot_table(index="district", columns="iso_week",
                         values="pred_log", observed=True).reindex(order))
    vmax = float(np.nanmax([act.to_numpy(), prd.to_numpy()]))
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, mat, title in [(axes[0], act, f"ACTUAL burning by week ({year})"),
                           (axes[1], prd, f"PREDICTED (Model A) ({year})")]:
        sns.heatmap(mat, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
                    cbar_kws={"label": "log(1+DM tonnes)"})
        ax.set(xlabel="ISO week", title=title)
    axes[0].set_ylabel("District")
    fig.suptitle("Early-warning check: does the model light up the right district-weeks?",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "07_earlywarning_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_bar(metrics: list[dict]) -> None:
    md = pd.DataFrame(metrics).set_index("model")
    show = ["Spearman", "ROC_AUC_top10pct", "Precision@k", "R2_log"]
    md = md[show]
    fig, ax = plt.subplots(figsize=(12, 7))
    md.T.plot(kind="bar", ax=ax, color=["#8d99ae", "#30638e", "#d1495b"])
    ax.set(ylabel="Score", title="Test-set skill (2017-2018): higher is better")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.legend(title="Model", loc="lower right")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "08_model_skill.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = build_frame()
    df = add_climatology(df)
    has_weather = df["tmax_mean"].notna().any() if "tmax_mean" in df else False
    print(f"Modeling frame: {len(df)} district-weeks | weather merged: {has_weather}")
    print(f"Train {TRAIN_YEARS} -> test {TEST_YEARS}\n")

    metrics = [eval_baseline(df)]
    model_b, te_b, m_b, _ = train_eval(df, BASE_FEATURES, "structural (B)")
    model_a, te_a, m_a, feats_a = train_eval(df, LAG_FEATURES, "operational (A)")
    metrics += [m_b, m_a]

    md = pd.DataFrame(metrics)[
        ["model", "RMSE_log", "MAE_log", "R2_log", "Spearman",
         "ROC_AUC_top10pct", "Precision@k"]]
    print("Test-set performance (2017-2018):")
    print(md.round(3).to_string(index=False))

    fig_pred_vs_actual(te_a, te_b)
    fig_feature_importance(model_a, feats_a)
    fig_earlywarning_heatmap(te_a, year=2018)
    fig_metrics_bar([{**m, "model": n} for m, n in
                     [(metrics[0], "climatology"), (m_b, "structural (B)"),
                      (m_a, "operational (A)")]])

    out = C.PROCESSED_DIR / "predictions_district_week.csv"
    keep = ["state", "district", "year", "iso_week", "dm_tonnes", "log_dm",
            "pred_log", "dw_clim"]
    te_a[keep].assign(pred_tonnes=np.expm1(te_a["pred_log"])).to_csv(out, index=False)

    print("\nSaved figures -> reports/figures/ (05_pred_vs_actual, 06_feature_importance,")
    print("                 07_earlywarning_heatmap, 08_model_skill)")
    print(f"Saved predictions -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
