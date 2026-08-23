"""Build the district × period fire panel from standardized FIRMS points.

This is the join point of the project: raw point detections in, an analysis-ready
panel out. Both the causal (DiD) and predictive models consume a panel produced here,
so the aggregation rules (confidence handling, balancing, season windowing) live in
one place.
"""
from __future__ import annotations

import pandas as pd

from fire_policy import config as C
from fire_policy.geo import assign_points_to_districts, get_region_districts


# --------------------------------------------------------------------------- #
# Confidence handling (MODIS is numeric 0-100; VIIRS is low/nominal/high)
# --------------------------------------------------------------------------- #
def add_confidence_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = pd.to_numeric(df["confidence"], errors="coerce")
    cat = df["confidence"].astype(str).str.lower()
    df["high_conf"] = (num >= 80) | cat.eq("high")
    df["low_conf"] = (num < 30) | cat.eq("low")
    return df


def _prep_points(fires: pd.DataFrame) -> pd.DataFrame:
    df = add_confidence_flags(fires)
    d = pd.to_datetime(df["acq_date"], errors="coerce")
    df["date"] = d
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["iso_week"] = d.dt.isocalendar().week.astype("Int64")
    df["in_season"] = df["month"].isin(C.BURNING_SEASON_MONTHS)
    return df


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def fires_to_panel(fires: pd.DataFrame, freq: str = "season",
                   drop_low_conf: bool = True, balanced: bool = True) -> pd.DataFrame:
    """Aggregate standardized FIRMS points to a district panel.

    freq:
      'year'   -> district × calendar year (all months)
      'season' -> district × year, burning-season months only (DiD outcome)
      'week'   -> district × (year, iso_week) (spatiotemporal model)
    balanced: fill absent district×period cells with zero fire counts.
    """
    df = _prep_points(fires)
    if drop_low_conf:
        df = df[~df["low_conf"]]

    tagged = assign_points_to_districts(df)
    tagged = tagged.dropna(subset=["district"]).copy()

    if freq == "season":
        tagged = tagged[tagged["in_season"]]
        keys = ["state", "district", "year"]
    elif freq == "year":
        keys = ["state", "district", "year"]
    elif freq == "week":
        keys = ["state", "district", "year", "iso_week"]
    else:
        raise ValueError("freq must be 'year', 'season', or 'week'")

    panel = (
        tagged.groupby(keys)
        .agg(fire_count=("latitude", "size"),
             high_conf_count=("high_conf", "sum"),
             frp_mean=("frp", "mean"),
             frp_sum=("frp", "sum"))
        .reset_index()
    )

    if balanced:
        panel = _balance(panel, freq, keys)
    return panel.sort_values(keys).reset_index(drop=True)


def _balance(panel: pd.DataFrame, freq: str, keys: list[str]) -> pd.DataFrame:
    """Reindex to the full district × period grid, filling zeros."""
    districts = get_region_districts()[["state", "district"]].drop_duplicates()
    years = sorted(panel["year"].dropna().unique())
    if not years:
        return panel

    grid = districts.assign(_k=1).merge(
        pd.DataFrame({"year": years, "_k": 1}), on="_k").drop(columns="_k")
    if freq == "week":
        weeks = sorted(panel["iso_week"].dropna().unique())
        grid = grid.assign(_k=1).merge(
            pd.DataFrame({"iso_week": weeks, "_k": 1}), on="_k").drop(columns="_k")

    out = grid.merge(panel, on=keys, how="left")
    count_cols = ["fire_count", "high_conf_count", "frp_sum"]
    out[count_cols] = out[count_cols].fillna(0)
    out["frp_mean"] = out["frp_mean"].fillna(0.0)
    return out


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Smoke test on whatever fire pull is available (current feed by default).
    src = C.INTERIM_DIR / "firms_current_punjab_haryana.csv"
    if not src.exists():
        print(f"No fire input at {src}; run `python -m fire_policy.firms` first.")
        return
    fires = pd.read_csv(src)
    print(f"Input detections: {len(fires)}")
    for freq in ("year", "season", "week"):
        panel = fires_to_panel(fires, freq=freq, balanced=(freq != "week"))
        nz = (panel["fire_count"] > 0).sum()
        print(f"\nfreq={freq}: panel rows={len(panel)}, non-zero cells={nz}")
        print(panel.sort_values('fire_count', ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
