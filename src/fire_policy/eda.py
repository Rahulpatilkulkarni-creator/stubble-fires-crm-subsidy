"""Exploratory analysis: figures + an interactive hotspot map from the built panels.

Consumes:
  data/processed/fire_sage_district_year.csv   (historical burned mass, 2012-2018)
  data/processed/weather_district_season.csv   (seasonal weather covariates)
  data/raw/sage_igp/SAGE_*.nc                   (daily grids, for seasonal timing)

Writes PNG figures to reports/figures/ and a folium choropleth to reports/.
Everything here is descriptive -- it sets up the causal (DiD) and predictive work.
"""
from __future__ import annotations

import glob
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from fire_policy import config as C
from fire_policy.geo import get_region_districts

sns.set_theme(style="whitegrid", context="talk")
FIRE_PANEL = C.PROCESSED_DIR / "fire_sage_district_year.csv"
WEATHER_PANEL = C.PROCESSED_DIR / "weather_district_season.csv"
SAGE_DIR = C.RAW_DIR / "sage_igp"
_SRC = "Source: SAGE-IGP (Liu et al. 2020, CC0) + Open-Meteo ERA5"


def _load():
    fire = pd.read_csv(FIRE_PANEL)
    weather = pd.read_csv(WEATHER_PANEL) if WEATHER_PANEL.exists() else pd.DataFrame()
    return fire, weather


# --------------------------------------------------------------------------- #
def fig_state_trend(fire: pd.DataFrame) -> None:
    tot = (fire.groupby(["year", "state"])["dm_tonnes"].sum() / 1e6).unstack("state")
    fig, ax = plt.subplots(figsize=(10, 6))
    for state in tot.columns:
        ax.plot(tot.index, tot[state], marker="o", linewidth=2.5, label=state)
    ax.set(xlabel="Year", ylabel="Dry matter burned (million tonnes)",
           title="Post-monsoon stubble burning: Punjab vs Haryana (2012-2018)")
    ax.legend(title="State")
    ax.figure.text(0.99, 0.01, _SRC, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "01_state_trend.png", dpi=140)
    plt.close(fig)


def fig_top_districts(fire: pd.DataFrame, n: int = 15) -> None:
    mean_dm = (fire.groupby(["state", "district"])["dm_tonnes"].mean()
               .sort_values(ascending=False).head(n) / 1e3)
    order = mean_dm.reset_index()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = order["state"].map({"Punjab": "#d1495b", "Haryana": "#30638e"})
    ax.barh(order["district"][::-1], order["dm_tonnes"][::-1],
            color=list(colors)[::-1])
    ax.set(xlabel="Mean seasonal dry matter burned (thousand tonnes)",
           title=f"Top {n} burning districts (2012-2018 average)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#d1495b", "#30638e"]]
    ax.legend(handles, ["Punjab", "Haryana"], title="State")
    ax.figure.text(0.99, 0.01, _SRC, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "02_top_districts.png", dpi=140)
    plt.close(fig)


def fig_fire_weather(fire: pd.DataFrame, weather: pd.DataFrame) -> dict:
    if weather.empty:
        return {}
    m = fire.merge(weather, on=["state", "district", "year"], how="inner")
    if m.empty:
        return {}
    m["dm_kt"] = m["dm_tonnes"] / 1e3
    drivers = [("rain_days", "Rain days in season"),
               ("precip_sum", "Season precipitation (mm)"),
               ("et0_sum", "Season evapotranspiration (mm)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    corrs = {}
    for ax, (col, label) in zip(axes, drivers):
        r = m[["dm_kt", col]].corr().iloc[0, 1]
        corrs[col] = r
        sns.regplot(data=m, x=col, y="dm_kt", ax=ax,
                    scatter_kws=dict(alpha=0.4, s=40), line_kws=dict(color="black"))
        ax.set(xlabel=label, ylabel="DM burned (thousand tonnes)",
               title=f"r = {r:+.2f}")
    fig.suptitle("District-season burning vs weather drivers (2015-2018 overlap)",
                 y=1.02)
    fig.text(0.99, 0.01, _SRC, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "03_fire_vs_weather.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return corrs


def fig_seasonal_timing() -> None:
    """Regional daily DM averaged across years -> when does burning peak?"""
    paths = sorted(glob.glob(str(SAGE_DIR / "SAGE_2*.nc")))
    w, s, e, n = C.REGION_BBOX
    series = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            sub = ds.sel(lon=slice(w, e), lat=slice(n, s))  # lat is descending
            daily = sub["DM"].sum(["lat", "lon"]).to_series() / 1e9  # -> million t
            daily.index = pd.to_datetime(daily.index)
            series.append(daily.groupby([daily.index.month, daily.index.day]).sum())
    if not series:
        return
    stacked = pd.concat(series, axis=1)
    mean_day = stacked.mean(axis=1)
    lo, hi = stacked.min(axis=1), stacked.max(axis=1)
    x = [pd.Timestamp(2001, mm, dd) for mm, dd in mean_day.index]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(x, lo, hi, alpha=0.2, color="#d1495b", label="min-max across years")
    ax.plot(x, mean_day.values, color="#d1495b", linewidth=2.5, label="mean")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set(xlabel="Day of season", ylabel="Regional DM burned (million tonnes/day)",
           title="When does Punjab-Haryana stubble burning peak? (2012-2018)")
    ax.legend()
    ax.figure.text(0.99, 0.01, _SRC, ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "04_seasonal_timing.png", dpi=140)
    plt.close(fig)


def map_hotspots(fire: pd.DataFrame) -> None:
    import folium
    gdf = get_region_districts()[["state", "district", "geometry"]].copy()
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    mean_dm = (fire.groupby("district")["dm_tonnes"].mean() / 1e3).rename("mean_dm_kt")
    gdf = gdf.merge(mean_dm, on="district", how="left")
    gdf["mean_dm_kt"] = gdf["mean_dm_kt"].fillna(0).round(1)

    m = folium.Map(location=[30.3, 75.9], zoom_start=7, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=gdf.to_json(), data=gdf,
        columns=["district", "mean_dm_kt"], key_on="feature.properties.district",
        fill_color="YlOrRd", fill_opacity=0.75, line_opacity=0.4, nan_fill_color="lightgray",
        legend_name="Mean seasonal dry matter burned, thousand tonnes (2012-2018)",
    ).add_to(m)
    folium.GeoJson(
        gdf.to_json(),
        style_function=lambda _: {"fillOpacity": 0, "weight": 0},
        tooltip=folium.GeoJsonTooltip(
            fields=["district", "state", "mean_dm_kt"],
            aliases=["District", "State", "Mean DM (kt)"]),
    ).add_to(m)
    m.save(str(C.REPORTS_DIR / "hotspot_map.html"))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    fire, weather = _load()
    print(f"Fire panel: {len(fire)} rows ({fire.year.min()}-{fire.year.max()}); "
          f"weather: {len(weather)} rows")

    fig_state_trend(fire)
    fig_top_districts(fire)
    corrs = fig_fire_weather(fire, weather)
    fig_seasonal_timing()
    map_hotspots(fire)

    print("\nFire-weather correlations (district-season, 2015-2018):")
    for k, v in corrs.items():
        print(f"  DM vs {k}: r = {v:+.2f}")
    print("\nSaved figures -> reports/figures/  (01_state_trend, 02_top_districts, "
          "03_fire_vs_weather, 04_seasonal_timing)")
    print("Saved map    -> reports/hotspot_map.html")


if __name__ == "__main__":
    main()
