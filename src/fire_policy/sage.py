"""SAGE-IGP agricultural-fire loader -> district x year burned-mass panel.

SAGE-IGP (Liu et al. 2020, Harvard Dataverse doi:10.7910/DVN/JUMXOL, CC0) gives
gridded post-monsoon dry-matter burned over the Indo-Gangetic Plain on a 0.25 deg
lat/lon grid, daily, Sep-Dec, for 2003-2018. We use it as the *keyless* historical
fire record for Punjab & Haryana.

Each grid cell's DM (kg) is summed over the burning-season months, then allocated to
districts by AREA-WEIGHTED overlay: a cell that overlaps two districts contributes to
each in proportion to the overlap area. This handles the coarse 0.25 deg (~667 km2)
grid far better than cell-centre assignment, so small districts are not dropped.

Two honest limits of this source:
  * Resolution: 0.25 deg is coarse relative to small districts; totals are approximate.
  * Coverage ends in 2018 (the CRM scale-up year), so SAGE supports EDA, prediction
    training, and pre-trend checks -- NOT the post-2018 leg of the DiD. That needs a
    (self-issued) FIRMS MAP_KEY, wired via fire_policy.firms.fetch_area.
"""
from __future__ import annotations

import glob
import itertools
import re
import sys

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import box

from fire_policy import config as C
from fire_policy.geo import get_region_districts

SAGE_DIR = C.RAW_DIR / "sage_igp"
_EQUAL_AREA = 6933          # World Cylindrical Equal Area (metres) for area ratios
_CELL_HALF = 0.125          # half of the 0.25 deg grid step


def _year_from_path(path: str) -> int | None:
    m = re.search(r"SAGE_(\d{4})\.nc$", path.replace("\\", "/"))
    return int(m.group(1)) if m else None


def load_year_cells(path: str, season_months=None) -> pd.DataFrame:
    """One SAGE file -> nonzero grid cells with season-summed DM (kg).

    Returns columns: lon, lat, dm_kg  (burning-season total per 0.25 deg cell).
    """
    season_months = tuple(season_months or C.BURNING_SEASON_MONTHS)
    with xr.open_dataset(path) as ds:
        dm = ds["DM"].sel(time=ds["time"].dt.month.isin(season_months))
        dm_season = dm.sum("time", skipna=True)  # (lat, lon)
        df = dm_season.to_dataframe(name="dm_kg").reset_index()
    return df[df["dm_kg"] > 0][["lon", "lat", "dm_kg"]].reset_index(drop=True)


def _districts_layer() -> gpd.GeoDataFrame:
    g = get_region_districts()[["state", "district", "geometry"]].copy()
    if g.crs is None:
        g = g.set_crs(4326)
    else:
        g = g.to_crs(4326)
    g["geometry"] = g.geometry.buffer(0)  # heal any minor invalidities
    return g


def allocate_cells_to_districts(cells: pd.DataFrame,
                                districts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Area-weighted allocation of cell DM to districts.

    Each cell is a 0.25 deg square; its DM is split among intersecting districts in
    proportion to intersection area. Returns state, district, dm_kg.
    """
    if cells.empty:
        return pd.DataFrame(columns=["state", "district", "dm_kg"])
    geom = [box(x - _CELL_HALF, y - _CELL_HALF, x + _CELL_HALF, y + _CELL_HALF)
            for x, y in zip(cells["lon"], cells["lat"])]
    gcells = gpd.GeoDataFrame(
        cells.assign(cell_id=range(len(cells))), geometry=geom, crs=4326)
    gcells["cell_area"] = gcells.to_crs(_EQUAL_AREA).area.to_numpy()

    inter = gpd.overlay(
        gcells[["cell_id", "dm_kg", "cell_area", "geometry"]],
        districts, how="intersection")
    if inter.empty:
        return pd.DataFrame(columns=["state", "district", "dm_kg"])
    inter["inter_area"] = inter.to_crs(_EQUAL_AREA).area.to_numpy()
    inter["dm_alloc"] = inter["dm_kg"] * (inter["inter_area"] / inter["cell_area"])
    return (inter.groupby(["state", "district"])["dm_alloc"].sum()
            .reset_index().rename(columns={"dm_alloc": "dm_kg"}))


def _balance(panel: pd.DataFrame, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    keys = districts[["state", "district"]].drop_duplicates()
    years = sorted(panel["year"].unique())
    grid = (keys.assign(_k=1)
            .merge(pd.DataFrame({"year": years, "_k": 1}), on="_k").drop(columns="_k"))
    out = grid.merge(panel, on=["state", "district", "year"], how="left")
    out["dm_kg"] = out["dm_kg"].fillna(0.0)
    return out


def build_sage_panel(save: bool = True, season_months=None) -> pd.DataFrame:
    """All SAGE years -> balanced district x year seasonal dry-matter-burned panel."""
    paths = sorted(p for p in glob.glob(str(SAGE_DIR / "SAGE_*.nc"))
                   if _year_from_path(p))
    if not paths:
        return pd.DataFrame()
    districts = _districts_layer()

    frames = []
    for path in paths:
        year = _year_from_path(path)
        cells = load_year_cells(path, season_months=season_months)
        region_total = float(cells["dm_kg"].sum())
        alloc = allocate_cells_to_districts(cells, districts)
        alloc["year"] = year
        frames.append(alloc)
        retained = alloc["dm_kg"].sum()
        print(f"  {year}: {len(cells):4d} burning cells | "
              f"DM allocated to PB/HR = {retained/1e9:6.2f} Mt "
              f"({100*retained/region_total:4.1f}% of IGP-domain {region_total/1e9:.2f} Mt)",
              flush=True)

    panel = pd.concat(frames, ignore_index=True)
    panel = _balance(panel, districts)
    panel["dm_tonnes"] = panel["dm_kg"] / 1000.0
    panel = (panel[["state", "district", "year", "dm_kg", "dm_tonnes"]]
             .sort_values(["state", "district", "year"]).reset_index(drop=True))
    if save:
        out = C.PROCESSED_DIR / "fire_sage_district_year.csv"
        panel.to_csv(out, index=False)
    return panel


# --------------------------------------------------------------------------- #
# Weekly panel (prediction target). The grid is fixed across years, so the
# cell -> district area weights are computed once and reused for every week.
# --------------------------------------------------------------------------- #
def cell_district_weights(districts: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """Area fraction of each 0.25 deg grid cell inside each district (computed once).

    Returns lon, lat, state, district, weight  where weight = intersection_area /
    cell_area. A cell straddling two districts yields multiple rows; weights over a
    cell sum to <=1 (shortfall = area outside Punjab/Haryana).
    """
    districts = districts if districts is not None else _districts_layer()
    paths = sorted(p for p in glob.glob(str(SAGE_DIR / "SAGE_*.nc"))
                   if _year_from_path(p))
    with xr.open_dataset(paths[0]) as ds:
        lons, lats = ds["lon"].values, ds["lat"].values
    coords = list(itertools.product([float(x) for x in lons], [float(y) for y in lats]))
    geom = [box(x - _CELL_HALF, y - _CELL_HALF, x + _CELL_HALF, y + _CELL_HALF)
            for x, y in coords]
    gcells = gpd.GeoDataFrame(
        {"lon": [c[0] for c in coords], "lat": [c[1] for c in coords]},
        geometry=geom, crs=4326)
    gcells["cell_area"] = gcells.to_crs(_EQUAL_AREA).area.to_numpy()
    inter = gpd.overlay(gcells, districts, how="intersection")
    if inter.empty:
        return pd.DataFrame(columns=["lon", "lat", "state", "district", "weight"])
    inter["weight"] = inter.to_crs(_EQUAL_AREA).area.to_numpy() / inter["cell_area"]
    return inter[["lon", "lat", "state", "district", "weight"]]


def build_sage_weekly_panel(save: bool = True, season_months=None) -> pd.DataFrame:
    """All SAGE years -> balanced district x (year, iso_week) burned-mass panel."""
    season_months = tuple(season_months or C.BURNING_SEASON_MONTHS)
    weights = cell_district_weights()
    keys = weights[["state", "district"]].drop_duplicates()
    paths = sorted(p for p in glob.glob(str(SAGE_DIR / "SAGE_*.nc"))
                   if _year_from_path(p))

    frames, year_weeks = [], []
    for path in paths:
        year = _year_from_path(path)
        with xr.open_dataset(path) as ds:
            df = ds["DM"].to_dataframe(name="dm_kg").reset_index()
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"].dt.month.isin(season_months) & (df["dm_kg"] > 0)].copy()
        df["iso_week"] = df["time"].dt.isocalendar().week.astype(int)

        # full set of season weeks for this year (so zero-burning weeks are kept)
        srange = pd.date_range(f"{year}-{min(season_months):02d}-01",
                               f"{year}-{max(season_months):02d}-30")
        for wk in srange.isocalendar().week.astype(int).unique():
            year_weeks.append((year, int(wk)))

        cw = df.groupby(["lon", "lat", "iso_week"])["dm_kg"].sum().reset_index()
        merged = cw.merge(weights, on=["lon", "lat"], how="inner")
        merged["dm_alloc"] = merged["dm_kg"] * merged["weight"]
        g = (merged.groupby(["state", "district", "iso_week"])["dm_alloc"].sum()
             .reset_index().rename(columns={"dm_alloc": "dm_kg"}))
        g["year"] = year
        frames.append(g)

    panel = pd.concat(frames, ignore_index=True)
    yw = pd.DataFrame(sorted(set(year_weeks)), columns=["year", "iso_week"])
    grid = (keys.assign(_k=1).merge(yw.assign(_k=1), on="_k").drop(columns="_k"))
    panel = grid.merge(panel, on=["state", "district", "year", "iso_week"], how="left")
    panel["dm_kg"] = panel["dm_kg"].fillna(0.0)
    panel["dm_tonnes"] = panel["dm_kg"] / 1000.0
    panel = (panel[["state", "district", "year", "iso_week", "dm_kg", "dm_tonnes"]]
             .sort_values(["state", "district", "year", "iso_week"]).reset_index(drop=True))
    if save:
        panel.to_csv(C.PROCESSED_DIR / "fire_sage_district_week.csv", index=False)
    return panel


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Building SAGE-IGP district x year burned-mass panel...")
    panel = build_sage_panel(save=True)
    if panel.empty:
        print(f"No SAGE files found in {SAGE_DIR}")
        return
    print(f"\nPanel: {len(panel)} rows | "
          f"{panel.district.nunique()} districts x {panel.year.nunique()} years "
          f"({panel.year.min()}-{panel.year.max()})")

    print("\nState-year total dry matter burned (million tonnes):")
    piv = (panel.groupby(["year", "state"])["dm_tonnes"].sum() / 1e6).unstack("state")
    print(piv.round(2).to_string())

    print("\nTop district-years by dry matter burned (tonnes):")
    top = panel.nlargest(10, "dm_tonnes")[["state", "district", "year", "dm_tonnes"]]
    print(top.to_string(index=False))

    unresolved = (panel.groupby("district")["dm_kg"].sum() == 0).sum()
    print(f"\nDistricts with zero DM in all years (grid-unresolved / no burning): {unresolved}")

    out = C.PROCESSED_DIR / "fire_sage_district_year.csv"
    print(f"\nSaved -> {out.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
