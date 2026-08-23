"""Open-Meteo weather fetchers (free, keyless).

Two endpoints:
  - ERA5 archive  (historical, ~1940→present, ~5-day lag)  -> `fetch_archive`
  - Forecast      (next ~16 days)                           -> `fetch_forecast`

We pull the drivers that matter for stubble burning and smoke dispersion:
wind (does smoke clear or stagnate?), humidity & precipitation (can residue
even be burned?), temperature, and evapotranspiration (dryness).
"""
from __future__ import annotations

import sys
import time

import pandas as pd
import requests

from fire_policy import config as C

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 60
_SESSION = requests.Session()

DEFAULT_DAILY = [
    "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]
# Hourly relative humidity + wind, aggregated to daily stats below.
DEFAULT_HOURLY = ["relative_humidity_2m", "wind_speed_10m"]


# Representative anchor points across the burning belt (name, lat, lon).
# Phase 2 replaces these with true district centroids.
ANCHOR_POINTS = [
    ("Sangrur (PB)", 30.25, 75.84),
    ("Bathinda (PB)", 30.21, 74.95),
    ("Ferozepur (PB)", 30.93, 74.61),
    ("Ludhiana (PB)", 30.90, 75.86),
    ("Amritsar (PB)", 31.63, 74.87),
    ("Karnal (HR)", 29.69, 76.99),
    ("Kaithal (HR)", 29.80, 76.40),
    ("Fatehabad (HR)", 29.51, 75.45),
    ("Sirsa (HR)", 29.53, 75.03),
    ("Hisar (HR)", 29.15, 75.72),
]


def _get_json(url: str, params: dict, max_retries: int = 6) -> dict:
    """GET JSON with polite backoff on 429 / 5xx (Open-Meteo rate limits)."""
    last = None
    for attempt in range(max_retries):
        r = _SESSION.get(url, params=params, timeout=_TIMEOUT)
        last = r
        if r.status_code == 429 or r.status_code >= 500:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(60.0, 3.0 * (2 ** attempt))
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    last.raise_for_status()  # retries exhausted -> surface the last error
    return last.json()


def _daily_frame(js: dict, lat: float, lon: float, name: str | None) -> pd.DataFrame:
    daily = js.get("daily", {})
    if not daily:
        return pd.DataFrame()
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df["lat"], df["lon"] = lat, lon
    if name:
        df["site"] = name
    return df


def _aggregate_hourly(js: dict) -> pd.DataFrame:
    """Collapse hourly RH / wind to daily mean & min/max."""
    hourly = js.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    h = pd.DataFrame(hourly)
    h["date"] = pd.to_datetime(h["time"]).dt.floor("D")
    agg = {}
    if "relative_humidity_2m" in h:
        agg["rh_mean"] = ("relative_humidity_2m", "mean")
        agg["rh_min"] = ("relative_humidity_2m", "min")
    if "wind_speed_10m" in h:
        agg["wind_mean"] = ("wind_speed_10m", "mean")
    if not agg:
        return pd.DataFrame()
    out = h.groupby("date").agg(**agg).reset_index()
    return out


def fetch_archive(lat: float, lon: float, start_date: str, end_date: str,
                  name: str | None = None, daily=None, hourly=None,
                  timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    """Historical daily weather for one point (ERA5 archive)."""
    daily = DEFAULT_DAILY if daily is None else daily
    hourly = DEFAULT_HOURLY if hourly is None else hourly
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ",".join(daily),
        "timezone": timezone,
    }
    if hourly:
        params["hourly"] = ",".join(hourly)
    js = _get_json(ARCHIVE_URL, params)
    daily_df = _daily_frame(js, lat, lon, name)
    hourly_df = _aggregate_hourly(js) if hourly else pd.DataFrame()
    if not daily_df.empty and not hourly_df.empty:
        daily_df = daily_df.merge(hourly_df, on="date", how="left")
    return daily_df


def fetch_archive_multi(coords, start_date: str, end_date: str, daily=None,
                        timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    """Archive daily weather for MANY points in ONE call (Open-Meteo multi-location).

    coords: iterable of (key, lat, lon). Returns a long frame with a 'key' column so
    callers can split the identity back out. Batching all points into a single request
    keeps a full historical build to a handful of calls -- essential for staying under
    the API's hourly request limit.
    """
    daily = DEFAULT_DAILY if daily is None else daily
    coords = list(coords)
    params = {
        "latitude": ",".join(str(c[1]) for c in coords),
        "longitude": ",".join(str(c[2]) for c in coords),
        "start_date": start_date, "end_date": end_date,
        "daily": ",".join(daily), "timezone": timezone,
    }
    js = _get_json(ARCHIVE_URL, params)
    locs = js if isinstance(js, list) else [js]
    frames = []
    for (key, lat, lon), loc in zip(coords, locs):
        d = loc.get("daily") if isinstance(loc, dict) else None
        if not d or "time" not in d:
            continue
        f = pd.DataFrame(d)
        f["date"] = pd.to_datetime(f["time"])
        f = f.drop(columns=["time"])
        f["key"], f["lat"], f["lon"] = key, lat, lon
        frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_forecast(lat: float, lon: float, name: str | None = None,
                   days: int = 16, daily=None, hourly=None,
                   timezone: str = "Asia/Kolkata") -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon, "forecast_days": days,
        "daily": ",".join(daily or DEFAULT_DAILY),
        "hourly": ",".join(hourly or DEFAULT_HOURLY),
        "timezone": timezone,
    }
    js = _get_json(FORECAST_URL, params)
    daily_df = _daily_frame(js, lat, lon, name)
    hourly_df = _aggregate_hourly(js)
    if not daily_df.empty and not hourly_df.empty:
        daily_df = daily_df.merge(hourly_df, on="date", how="left")
    return daily_df


def fetch_archive_points(points=None, start_date="2023-10-01", end_date="2023-11-30",
                         polite_sleep: float = 0.2) -> pd.DataFrame:
    """Archive weather for several anchor points, stacked into one frame."""
    points = points or ANCHOR_POINTS
    frames = []
    for name, lat, lon in points:
        try:
            frames.append(fetch_archive(lat, lon, start_date, end_date, name=name))
        except Exception as exc:
            print(f"  ! {name} failed: {exc}", file=sys.stderr)
        time.sleep(polite_sleep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_weather_panel(years=None, season_months=None, save: bool = True,
                        resume: bool = True, polite_sleep: float = 1.5) -> pd.DataFrame:
    """District x year seasonal weather covariates for the whole study horizon.

    Pulls the ERA5 daily archive at each district's representative point, restricts
    to the burning-season months, and aggregates the drivers that gate burning and
    smoke dispersion. Keyless. Resumable: a re-run only fetches districts not already
    in the saved panel, so a rate-limited / interrupted run can be finished cheaply.
    """
    from fire_policy.geo import district_centroids
    years = years or C.STUDY_YEARS
    season_months = tuple(season_months or C.BURNING_SEASON_MONTHS)
    pts = district_centroids()
    daily_vars = ["temperature_2m_max", "precipitation_sum",
                  "wind_speed_10m_max", "et0_fao_evapotranspiration"]
    start, end = f"{min(years)}-01-01", f"{max(years)}-12-31"
    out = C.PROCESSED_DIR / "weather_district_season.csv"
    cols = ["state", "district", "year", "wind_mean", "wind_calm_days",
            "precip_sum", "rain_days", "tmax_mean", "et0_sum"]

    existing = pd.read_csv(out) if (resume and out.exists()) else pd.DataFrame()
    done = set(existing["district"]) if len(existing) else set()

    rows = []
    for i, r in enumerate(pts.itertuples(index=False), 1):
        if r.district in done:
            continue
        try:
            df = fetch_archive(r.lat, r.lon, start, end, daily=daily_vars, hourly=[])
        except Exception as exc:
            print(f"  ! [{i}/{len(pts)}] {r.district} failed: {exc}", file=sys.stderr)
            continue
        if df.empty:
            continue
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        s = df[df["month"].isin(season_months)]
        g = s.groupby("year").agg(
            wind_mean=("wind_speed_10m_max", "mean"),
            wind_calm_days=("wind_speed_10m_max", lambda x: int((x < 10).sum())),
            precip_sum=("precipitation_sum", "sum"),
            rain_days=("precipitation_sum", lambda x: int((x > 1).sum())),
            tmax_mean=("temperature_2m_max", "mean"),
            et0_sum=("et0_fao_evapotranspiration", "sum"),
        ).reset_index()
        g["state"], g["district"] = r.state, r.district
        rows.append(g)
        print(f"  [{i}/{len(pts)}] {r.district}: {len(g)} yrs", file=sys.stderr)
        time.sleep(polite_sleep)

    new = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    panel = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    if len(panel):
        panel = (panel[cols].drop_duplicates(["district", "year"])
                 .sort_values(["state", "district", "year"]).reset_index(drop=True))
    if save and len(panel):
        panel.to_csv(out, index=False)
    return panel


def build_weather_weekly_panel(years=range(2012, 2019), season_months=(9, 10, 11),
                               save: bool = True, resume: bool = True,
                               polite_sleep: float = 1.0) -> pd.DataFrame:
    """District x (year, iso_week) weather features for the prediction model.

    ERA5 archive aggregated to ISO weeks over the pre/peak burning window (Sep-Nov).
    All districts are fetched together per year via the multi-location call, so a full
    build is ~7 light requests (well under the hourly rate limit) and is saved after
    each year (resumable/watchable). Keyless.
    """
    from fire_policy.geo import district_centroids
    season_months = tuple(season_months)
    pts = district_centroids()
    coords = [(f"{r.state}||{r.district}", float(r.lat), float(r.lon))
              for r in pts.itertuples(index=False)]
    daily_vars = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                  "wind_speed_10m_max", "et0_fao_evapotranspiration"]
    out = C.PROCESSED_DIR / "weather_district_week.csv"
    cols = ["state", "district", "year", "iso_week", "tmax_mean", "tmin_mean",
            "precip_sum", "rain_days", "dry_days", "wind_max_mean", "wind_calm_days",
            "et0_sum", "n_days"]

    existing = pd.read_csv(out) if (resume and out.exists()) else pd.DataFrame()
    done_years = set(existing["year"]) if len(existing) else set()
    panel = existing.copy()

    for y in years:
        if y in done_years:
            continue
        s0 = f"{y}-{min(season_months):02d}-01"
        s1 = f"{y}-{max(season_months):02d}-30"
        try:
            raw = fetch_archive_multi(coords, s0, s1, daily=daily_vars)
        except Exception as exc:
            print(f"  ! year {y} failed: {exc}", file=sys.stderr, flush=True)
            continue
        if raw.empty:
            continue
        raw[["state", "district"]] = raw["key"].str.split(r"\|\|", expand=True, regex=True)
        raw["year"] = raw["date"].dt.year
        raw["iso_week"] = raw["date"].dt.isocalendar().week.astype(int)
        g = raw.groupby(["state", "district", "year", "iso_week"]).agg(
            tmax_mean=("temperature_2m_max", "mean"),
            tmin_mean=("temperature_2m_min", "mean"),
            precip_sum=("precipitation_sum", "sum"),
            rain_days=("precipitation_sum", lambda x: int((x > 1).sum())),
            dry_days=("precipitation_sum", lambda x: int((x < 0.2).sum())),
            wind_max_mean=("wind_speed_10m_max", "mean"),
            wind_calm_days=("wind_speed_10m_max", lambda x: int((x < 10).sum())),
            et0_sum=("et0_fao_evapotranspiration", "sum"),
            n_days=("date", "count"),
        ).reset_index()
        panel = pd.concat([panel, g], ignore_index=True)
        if save:
            (panel[cols].drop_duplicates(["district", "year", "iso_week"])
             .sort_values(["state", "district", "year", "iso_week"])
             .to_csv(out, index=False))
        print(f"  year {y}: {g['district'].nunique()} districts, {len(g)} "
              f"district-weeks (saved)", file=sys.stderr, flush=True)
        time.sleep(polite_sleep)

    if len(panel):
        panel = (panel[cols].drop_duplicates(["district", "year", "iso_week"])
                 .sort_values(["state", "district", "year", "iso_week"])
                 .reset_index(drop=True))
    return panel


def main() -> None:
    # Demo: last burning season (Oct–Nov 2023) across the anchor points.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Fetching ERA5 archive weather for the burning belt (Oct-Nov 2023)...")
    df = fetch_archive_points(start_date="2023-10-01", end_date="2023-11-30")
    if df.empty:
        print("No weather returned.")
        return
    out = C.INTERIM_DIR / "weather_anchor_points_2023.csv"
    df.to_csv(out, index=False)
    print(f"Rows: {len(df):,} across {df['site'].nunique()} sites")
    cols = [c for c in ["site", "date", "wind_speed_10m_max", "precipitation_sum",
                        "rh_mean", "temperature_2m_max"] if c in df.columns]
    print(df[cols].head(8).to_string(index=False))
    print(f"\nSaved -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
