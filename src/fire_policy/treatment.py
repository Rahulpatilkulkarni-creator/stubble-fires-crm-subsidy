"""Treatment panel: turn the raw CRM subsidy records into a district-level *dose*.

The causal question is whether the Crop-Residue-Management (CRM) equipment subsidy
reduced stubble burning. That subsidy is not a clean on/off switch delivered on a
staggered timeline -- it is a *continuous intensity* rolled out to every district
from 2018-19. So the treatment is a dose, and the DiD is a dose-response design.

Two honest data-harmonisation problems, both handled here:

  1. The two states report different things. Punjab (PPCB MIS) gives *actual machines
     delivered*, cumulative 2018-19 -> 2021-22. Haryana gives 2018-19 *targets*
     (machines + subsidy INR). Raw counts are therefore NOT level-comparable across
     states. We make them comparable two ways, both invariant to the state-level
     scale: (a) `crm_share` -- a district's share of its own state's total; and
     (b) `dose_z` -- the within-state z-score. The DiD uses `dose_z` so that district
     and year fixed effects absorb every state-level level difference and only the
     *within-state, across-district* variation in intensity identifies the effect.

  2. District names differ from the geoBoundaries layer (spelling / renamed units).
     NAME_FIX crosswalks them and we assert a 1:1 match before saving.

Input : data/raw/treatment/{punjab,haryana}_crm_district_year.csv
Output: data/processed/treatment_district.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fire_policy import config as C
from fire_policy.geo import get_region_districts

TREAT_DIR = C.RAW_DIR / "treatment"
OUT = C.PROCESSED_DIR / "treatment_district.csv"
_EQUAL_AREA = 6933  # World Cylindrical Equal Area (metres) for district areas

# Treatment-source spelling -> geoBoundaries ADM2 spelling.
NAME_FIX = {
    # Punjab
    "Ferozepur": "Firozpur",
    "Ropar": "Rupnagar",
    "Sas Nagar": "Sahibzada Ajit Singh Nagar",
    "Sbs Nagar": "Shahid Bhagat Singh Nagar",
    "Sri Muktsar Sahib": "Muktsar",
    # Haryana
    "Gurugram": "Gurgaon",
}


def _pick(df: pd.DataFrame, metric: str, out_name: str) -> pd.DataFrame:
    """Extract one metric as a district-indexed column, summing any duplicates."""
    sub = df[df["metric"] == metric]
    if sub.empty:
        return pd.DataFrame(columns=["district", out_name])
    g = sub.groupby("district", as_index=False)["value"].sum()
    return g.rename(columns={"value": out_name})


def _district_areas() -> pd.DataFrame:
    g = get_region_districts()[["state", "district", "geometry"]].copy()
    if g.crs is None:
        g = g.set_crs(4326)
    g["area_km2"] = g.to_crs(_EQUAL_AREA).area.to_numpy() / 1e6
    return g[["state", "district", "area_km2"]]


def build_treatment_panel(save: bool = True) -> pd.DataFrame:
    """District treatment panel with harmonised, cross-state-comparable dose measures."""
    pb = pd.read_csv(TREAT_DIR / "punjab_crm_district_year.csv")
    hr = pd.read_csv(TREAT_DIR / "haryana_crm_district_year.csv")

    # --- Punjab: actual machines delivered (cumulative 2018-19..2021-22) --------- #
    pb_m = _pick(pb, "crm_machines_total", "crm_machines")
    pb_hs = _pick(pb, "crm_machines_happy_seeder", "happy_seeder")
    pb_ss = _pick(pb, "crm_machines_super_seeder", "super_seeder")
    pb_t = (pb_m.merge(pb_hs, on="district", how="left")
                .merge(pb_ss, on="district", how="left"))
    pb_t["state"] = "Punjab"
    pb_t["crm_subsidy_lakh"] = np.nan          # Punjab MIS reports machines, not INR
    pb_t["source_kind"] = "actual_machines_cum_2018_22"

    # --- Haryana: 2018-19 targeted machines + subsidy ---------------------------- #
    hr_m = _pick(hr, "crm_machines_target_total", "crm_machines")
    hr_s = _pick(hr, "crm_subsidy_target_total", "crm_subsidy_lakh")
    hr_hs = _pick(hr, "crm_machines_target_happy_seeder", "happy_seeder")
    hr_t = hr_m.merge(hr_s, on="district", how="left").merge(hr_hs, on="district", how="left")
    hr_t["state"] = "Haryana"
    hr_t["super_seeder"] = np.nan              # Haryana list has no separate super-seeder
    hr_t["source_kind"] = "target_machines_2018_19"

    cols = ["state", "district", "crm_machines", "crm_subsidy_lakh",
            "happy_seeder", "super_seeder", "source_kind"]
    panel = pd.concat([pb_t[cols], hr_t[cols]], ignore_index=True)

    # --- crosswalk names & validate against the geography layer ------------------ #
    panel["district"] = panel["district"].replace(NAME_FIX)
    geo = _district_areas()
    geo_keys = set(zip(geo["state"], geo["district"]))
    bad = [(s, d) for s, d in zip(panel["state"], panel["district"])
           if (s, d) not in geo_keys]
    if bad:
        raise ValueError(f"Treatment districts not matched in geo layer: {bad}")
    panel = panel.merge(geo, on=["state", "district"], how="left")

    # --- comparable dose measures (both state-scale invariant) ------------------- #
    grp = panel.groupby("state")["crm_machines"]
    panel["crm_state_total"] = grp.transform("sum")
    panel["crm_share"] = panel["crm_machines"] / panel["crm_state_total"]
    panel["dose_z"] = grp.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    panel["machines_per_1000km2"] = 1000.0 * panel["crm_machines"] / panel["area_km2"]
    panel["is_punjab"] = (panel["state"] == "Punjab").astype(int)

    panel = panel.sort_values(["state", "crm_machines"], ascending=[True, False]) \
                 .reset_index(drop=True)
    if save:
        panel.to_csv(OUT, index=False)
    return panel


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    panel = build_treatment_panel(save=True)
    print(f"Treatment panel: {len(panel)} districts "
          f"({panel.is_punjab.sum()} Punjab / {(1-panel.is_punjab).sum()} Haryana)\n")

    for st in ("Punjab", "Haryana"):
        s = panel[panel.state == st]
        kind = s["source_kind"].iloc[0]
        print(f"{st}: total CRM machines = {int(s.crm_machines.sum()):,} ({kind})")
        top = s.nlargest(5, "crm_machines")[["district", "crm_machines", "crm_share"]]
        for _, r in top.iterrows():
            print(f"    {r.district:28s} {int(r.crm_machines):6,d}  "
                  f"({r.crm_share*100:4.1f}% of state)")
    print(f"\nSaved -> {OUT.relative_to(C.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
