"""Administrative geography: turn fire points into district/state labels.

Uses geoBoundaries (CC-BY) India ADM1 (states) + ADM2 (districts). geoBoundaries
stores names with diacritics (e.g. "Haryāna") and its district layer carries no
parent-state field, so we:
  1. normalize names to ASCII,
  2. assign each district to a state by a spatial join (district representative
     point within a state polygon),
  3. clip to Punjab + Haryana and cache a small GeoJSON.
"""
from __future__ import annotations

import unicodedata
import warnings

import geopandas as gpd
import pandas as pd

from fire_policy import config as C

ADM1_PATH = C.RAW_DIR / "geoBoundaries_IND_ADM1.geojson"
ADM2_PATH = C.RAW_DIR / "geoBoundaries_IND_ADM2.geojson"
REGION_DISTRICTS_PATH = C.PROCESSED_DIR / "pb_hr_districts.geojson"


def _ascii(s) -> str:
    if not isinstance(s, str):
        return s
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().strip()


def load_states() -> gpd.GeoDataFrame:
    g = gpd.read_file(ADM1_PATH)
    g["state"] = g["shapeName"].map(_ascii)
    return g.to_crs(4326)


def load_districts() -> gpd.GeoDataFrame:
    g = gpd.read_file(ADM2_PATH)
    g["district"] = g["shapeName"].map(_ascii)
    return g.to_crs(4326)


def build_region_districts(save: bool = True) -> gpd.GeoDataFrame:
    """Clip India districts to Punjab + Haryana, attaching the state label."""
    states = load_states()
    pb_hr = states[states["state"].isin(["Punjab", "Haryana"])][["state", "geometry"]]
    if pb_hr.empty:
        raise RuntimeError("Punjab/Haryana not found in ADM1 — check name normalization.")

    districts = load_districts()[["district", "shapeID", "geometry"]]
    reps = districts.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # representative_point on geographic CRS
        reps["geometry"] = districts.representative_point()

    joined = gpd.sjoin(reps, pb_hr, predicate="within", how="inner")
    out = (
        districts[districts["shapeID"].isin(set(joined["shapeID"]))]
        .merge(joined[["shapeID", "state"]], on="shapeID", how="left")
        .sort_values(["state", "district"])
        .reset_index(drop=True)
    )
    if save:
        out.to_file(REGION_DISTRICTS_PATH, driver="GeoJSON")
    return out


def get_region_districts() -> gpd.GeoDataFrame:
    if REGION_DISTRICTS_PATH.exists():
        return gpd.read_file(REGION_DISTRICTS_PATH)
    return build_region_districts()


def assign_points_to_districts(df: pd.DataFrame, lon: str = "longitude",
                               lat: str = "latitude") -> gpd.GeoDataFrame:
    """Spatial-join a point DataFrame to Punjab/Haryana districts.

    Rows outside both states (bbox corners in Pakistan/Rajasthan) get NaN
    district/state — kept so callers can see the match rate.
    """
    districts = get_region_districts()
    pts = gpd.GeoDataFrame(
        df.copy(), geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326
    )
    joined = gpd.sjoin(pts, districts[["district", "state", "geometry"]],
                       predicate="within", how="left")
    return joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])


def district_centroids() -> pd.DataFrame:
    """One representative interior point per district (for per-district weather pulls)."""
    import warnings
    g = get_region_districts()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts = g.representative_point()
    return pd.DataFrame({
        "state": g["state"].to_numpy(),
        "district": g["district"].to_numpy(),
        "lat": pts.y.to_numpy(),
        "lon": pts.x.to_numpy(),
    })


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Building Punjab + Haryana district layer from geoBoundaries...")
    gdf = build_region_districts(save=True)
    print(f"Region districts: {len(gdf)}")
    print(gdf.groupby("state").size().to_string())
    print(f"Saved -> {REGION_DISTRICTS_PATH.relative_to(C.PROJECT_ROOT)}")

    fires_csv = C.INTERIM_DIR / "firms_current_punjab_haryana.csv"
    if fires_csv.exists():
        fires = pd.read_csv(fires_csv)
        tagged = assign_points_to_districts(fires)
        matched = tagged["district"].notna().sum()
        print(f"\nCurrent fires assigned to a PB/HR district: {matched}/{len(tagged)}")
        by_d = (tagged.dropna(subset=["district"])
                .groupby(["state", "district"]).size()
                .sort_values(ascending=False))
        if len(by_d):
            print("\nTop districts (current 7-day, off-season):")
            print(by_d.head(10).to_string())
        out = C.INTERIM_DIR / "firms_current_district_tagged.csv"
        tagged.drop(columns="geometry").to_csv(out, index=False)
        print(f"\nSaved -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
