"""NASA FIRMS fetchers.

Three ways to get thermal-anomaly (fire) detections:

1. `fetch_regional()`      — keyless near-real-time feeds (last 24h/48h/7d), South Asia.
                             Works with no key; used for current-season monitoring.
2. `fetch_area()`          — keyed Area API for arbitrary historical date ranges.
                             Needs a free FIRMS MAP_KEY in `.env` (FIRMS_MAP_KEY=...).
3. `load_archive_csv()`    — read a CSV exported from the FIRMS Archive Download tool.

All three return a standardized DataFrame (see `_standardize`) so downstream code
does not care which sensor or route the rows came from.
"""
from __future__ import annotations

import io
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional at import time
    pass

import os

from fire_policy import config as C

_TIMEOUT = 60
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "fire-policy-research/0.1 (portfolio)"})


# --------------------------------------------------------------------------- #
# Low-level HTTP
# --------------------------------------------------------------------------- #
def _get_text(url: str) -> str:
    r = _SESSION.get(url, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.text


def _csv_text_to_df(text: str) -> pd.DataFrame:
    """Parse FIRMS CSV text, guarding against auth/error bodies."""
    head = text.lstrip()[:80].lower()
    if head.startswith("invalid") or "<html" in head or head.startswith("error"):
        raise RuntimeError(f"FIRMS returned an error body: {text[:200]!r}")
    df = pd.read_csv(io.StringIO(text), dtype={"acq_time": str})
    return df


# --------------------------------------------------------------------------- #
# Standardization
# --------------------------------------------------------------------------- #
def _standardize(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Unify MODIS vs VIIRS columns into one schema.

    MODIS uses `brightness` / `bright_t31`; VIIRS uses `bright_ti4` / `bright_ti5`.
    """
    df = df.copy()
    if "brightness" not in df.columns and "bright_ti4" in df.columns:
        df["brightness"] = df["bright_ti4"]
    if "bright_t31" not in df.columns and "bright_ti5" in df.columns:
        df["bright_t31"] = df["bright_ti5"]

    df["sensor"] = sensor

    # acq_time is HHMM (UTC); zero-pad and build a UTC timestamp.
    if "acq_time" in df.columns:
        df["acq_time"] = df["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
        dt = pd.to_datetime(
            df["acq_date"].astype(str) + df["acq_time"],
            format="%Y-%m-%d%H%M", errors="coerce", utc=True,
        )
        df["acq_datetime_utc"] = dt

    keep = [
        "latitude", "longitude", "acq_date", "acq_time", "acq_datetime_utc",
        "satellite", "sensor", "confidence", "version", "frp", "daynight",
        "brightness", "bright_t31",
    ]
    return df[[c for c in keep if c in df.columns]]


# --------------------------------------------------------------------------- #
# Filtering / feature helpers
# --------------------------------------------------------------------------- #
def filter_bbox(df: pd.DataFrame, bbox=C.REGION_BBOX) -> pd.DataFrame:
    w, s, e, n = bbox
    m = (df.longitude.between(w, e)) & (df.latitude.between(s, n))
    return df.loc[m].reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["acq_date"], errors="coerce")
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["doy"] = d.dt.dayofyear
    df["in_burning_season"] = df["month"].isin(C.BURNING_SEASON_MONTHS)
    return df


# --------------------------------------------------------------------------- #
# 1. Keyless regional feeds
# --------------------------------------------------------------------------- #
def fetch_regional(sensor: str = "VIIRS_SNPP", window: str = "7d",
                   save: bool = True) -> pd.DataFrame:
    """Download a keyless South-Asia regional feed for one sensor.

    sensor ∈ FIRMS_REGIONAL_FEEDS keys; window ∈ {'24h','48h','7d'}.
    """
    if sensor not in C.FIRMS_REGIONAL_FEEDS:
        raise ValueError(f"sensor must be one of {list(C.FIRMS_REGIONAL_FEEDS)}")
    if window not in C.FIRMS_REGIONAL_WINDOWS:
        raise ValueError(f"window must be one of {C.FIRMS_REGIONAL_WINDOWS}")

    url = C.FIRMS_BASE + C.FIRMS_REGIONAL_FEEDS[sensor].format(window=window)
    df = _standardize(_csv_text_to_df(_get_text(url)), sensor)
    if save:
        out = C.RAW_DIR / f"firms_regional_{sensor}_{window}.csv"
        df.to_csv(out, index=False)
    return df


def fetch_all_regional(window: str = "7d", save: bool = True) -> pd.DataFrame:
    """All three sensors' regional feeds, concatenated + standardized."""
    frames = []
    for sensor in C.FIRMS_REGIONAL_FEEDS:
        try:
            frames.append(fetch_regional(sensor, window=window, save=save))
        except Exception as exc:  # keep going if one sensor feed is down
            print(f"  ! {sensor} feed failed: {exc}", file=sys.stderr)
    if not frames:
        raise RuntimeError("all regional feeds failed")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. Keyed Area API (historical / custom ranges)
# --------------------------------------------------------------------------- #
def get_map_key() -> str | None:
    return os.environ.get("FIRMS_MAP_KEY") or None


def _daterange_chunks(start: date, end: date, step: int):
    cur = start
    while cur <= end:
        span = min(step, (end - cur).days + 1)
        yield cur, span
        cur += timedelta(days=span)


def fetch_area(source: str, start_date: str, end_date: str,
               map_key: str | None = None, bbox=C.REGION_BBOX,
               save: bool = True, polite_sleep: float = 0.3) -> pd.DataFrame:
    """Historical pull over [start_date, end_date] via the keyed Area API.

    `source` should be an archive source for old dates (e.g. 'VIIRS_SNPP_SP')
    or an NRT source for the last ~2 months (e.g. 'VIIRS_SNPP_NRT').
    """
    map_key = map_key or get_map_key()
    if not map_key:
        raise RuntimeError(
            "No FIRMS MAP_KEY found. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/ , then set FIRMS_MAP_KEY "
            "in a .env file (copy .env.example)."
        )
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    sensor = source.replace("_SP", "").replace("_NRT", "")

    frames = []
    chunks = list(_daterange_chunks(start, end, C.FIRMS_AREA_MAX_DAYS))
    for i, (chunk_start, span) in enumerate(chunks, 1):
        url = C.FIRMS_BASE + C.FIRMS_AREA_TEMPLATE.format(
            map_key=map_key, source=source, bbox=C.bbox_str(bbox),
            day_range=span, date=chunk_start.isoformat(),
        )
        try:
            part = _csv_text_to_df(_get_text(url))
            if len(part):
                frames.append(_standardize(part, sensor))
        except Exception as exc:
            print(f"  ! chunk {i}/{len(chunks)} ({chunk_start} +{span}d) failed: {exc}",
                  file=sys.stderr)
        if polite_sleep:
            time.sleep(polite_sleep)

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if save and len(df):
        out = C.RAW_DIR / f"firms_area_{source}_{start_date}_{end_date}.csv"
        df.to_csv(out, index=False)
    return df


# --------------------------------------------------------------------------- #
# 3. Archive-download CSV loader
# --------------------------------------------------------------------------- #
def load_archive_csv(path: str | Path, sensor: str | None = None) -> pd.DataFrame:
    """Load a CSV exported from the FIRMS Archive Download tool."""
    raw = pd.read_csv(path, dtype={"acq_time": str})
    if sensor is None:
        sensor = "VIIRS" if "bright_ti4" in raw.columns else "MODIS"
    return _standardize(raw, sensor)


# --------------------------------------------------------------------------- #
# CLI: pull current fires over Punjab & Haryana (real data, no key needed)
# --------------------------------------------------------------------------- #
def _summary(df: pd.DataFrame) -> None:
    print(f"\nRegion-filtered detections: {len(df):,}")
    if df.empty:
        print("(No detections in-region for this window - expected outside Oct-Nov burning season.)")
        return
    print(f"Date range (UTC): {df.acq_datetime_utc.min()}  ->  {df.acq_datetime_utc.max()}")
    print("\nBy sensor:")
    print(df.sensor.value_counts().to_string())
    print("\nBy day/night:")
    print(df.daynight.value_counts().to_string())
    print(f"\nFRP (fire radiative power, MW): mean={df.frp.mean():.1f}  max={df.frp.max():.1f}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Fetching keyless FIRMS regional feeds (South Asia, last 7 days)...")
    allsa = fetch_all_regional(window="7d", save=True)
    print(f"South-Asia detections pulled: {len(allsa):,}")

    region = add_time_features(filter_bbox(allsa))
    out = C.INTERIM_DIR / "firms_current_punjab_haryana.csv"
    region.to_csv(out, index=False)
    _summary(region)
    print(f"\nSaved -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
