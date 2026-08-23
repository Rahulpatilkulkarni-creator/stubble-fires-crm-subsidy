"""Central configuration: geography, time windows, data sources, paths.

Everything spatial/temporal that the rest of the package depends on lives here
so the analysis is reproducible and easy to re-scope.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Geography — Punjab + Haryana (the stubble-burning belt feeding Delhi smog)
# --------------------------------------------------------------------------- #
# Bounding box as (west, south, east, north) i.e. (lon_min, lat_min, lon_max, lat_max).
# Generously covers both states; trimmed to precise district polygons in Phase 2.
REGION_BBOX = (73.5, 27.5, 77.8, 32.6)

STATES = ("Punjab", "Haryana")

# District rosters (current administrative units). Used for panel aggregation once
# GADM/administrative boundaries are attached in Phase 2.
PUNJAB_DISTRICTS = [
    "Amritsar", "Barnala", "Bathinda", "Faridkot", "Fatehgarh Sahib", "Fazilka",
    "Ferozepur", "Gurdaspur", "Hoshiarpur", "Jalandhar", "Kapurthala", "Ludhiana",
    "Malerkotla", "Mansa", "Moga", "Sri Muktsar Sahib", "Pathankot", "Patiala",
    "Rupnagar", "Sahibzada Ajit Singh Nagar", "Sangrur",
    "Shahid Bhagat Singh Nagar", "Tarn Taran",
]
HARYANA_DISTRICTS = [
    "Ambala", "Bhiwani", "Charkhi Dadri", "Faridabad", "Fatehabad", "Gurugram",
    "Hisar", "Jhajjar", "Jind", "Kaithal", "Karnal", "Kurukshetra", "Mahendragarh",
    "Nuh", "Palwal", "Panchkula", "Panipat", "Rewari", "Rohtak", "Sirsa",
    "Sonipat", "Yamunanagar",
]

# Known high-burning districts (for spot-checks / narrative; not a modeling input).
HOTSPOT_DISTRICTS = ["Sangrur", "Bathinda", "Ferozepur", "Moga", "Muktsar",
                     "Kaithal", "Karnal", "Fatehabad", "Sirsa", "Jind"]

# --------------------------------------------------------------------------- #
# Time windows
# --------------------------------------------------------------------------- #
# Kharif paddy-residue burning peaks Oct–Nov; a wider window captures early/late tails.
BURNING_SEASON_MONTHS = (10, 11)
BURNING_SEASON_WIDE = (9, 10, 11, 12)

# Study horizon. VIIRS-SNPP archive starts 2012; MODIS from 2000.
STUDY_YEARS = list(range(2015, 2025))

# Treatment onset for the DiD. The central "Promotion of Agricultural Mechanization
# for In-Situ Management of Crop Residue" (CRM) subsidy scheme scaled up from 2018-19.
# Confirm/refine against the subsidy-uptake data before finalizing.
POLICY_YEAR = 2018

# --------------------------------------------------------------------------- #
# NASA FIRMS sources
# --------------------------------------------------------------------------- #
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov"

# Keyless near-real-time regional feeds (last 24h / 48h / 7d), South Asia region.
FIRMS_REGIONAL_FEEDS = {
    "MODIS":        "/data/active_fire/modis-c6.1/csv/MODIS_C6_1_South_Asia_{window}.csv",
    "VIIRS_SNPP":   "/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_South_Asia_{window}.csv",
    "VIIRS_NOAA20": "/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_South_Asia_{window}.csv",
}
FIRMS_REGIONAL_WINDOWS = ("24h", "48h", "7d")

# Keyed Area API (historical + custom ranges). Requires a free MAP_KEY in .env.
# Template order: MAP_KEY -> SOURCE -> AREA(w,s,e,n) -> DAY_RANGE(1..5) -> DATE(YYYY-MM-DD)
FIRMS_AREA_TEMPLATE = "/api/area/csv/{map_key}/{source}/{bbox}/{day_range}/{date}"
FIRMS_AREA_MAX_DAYS = 5
# Standard-processing (archive) sources for historical years:
FIRMS_ARCHIVE_SOURCES = ("MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
# Near-real-time sources (recent ~2 months):
FIRMS_NRT_SOURCES = ("MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT")


def bbox_str(bbox: tuple[float, float, float, float] = REGION_BBOX) -> str:
    """FIRMS Area API expects 'west,south,east,north'."""
    return ",".join(str(x) for x in bbox)
