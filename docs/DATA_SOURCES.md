# Data Sources

This project joins three layers: a satellite **outcome** (fires), free **weather** drivers,
and a hard-to-assemble **policy treatment** (subsidy uptake). This document records exactly
where each comes from, at what granularity, and how hard it is to extract — including the
honest gaps.

> Sourcing note: a reconnaissance pass (2026-08-23) found many Indian government portals block
> automated fetches (PIB 403, DES/`aps.dac.gov.in` refused, ICRISAT DLD down, several JS-only
> report viewers). Items below marked **(browser)** are publicly viewable but require a real
> browser or headless scraper — not a simple scripted GET.

---

## 1. Fire detections — the OUTCOME (NASA FIRMS)

Thermal-anomaly detections from MODIS (C6.1) and VIIRS (SNPP + NOAA-20).

| Route | Coverage | Key? | Use |
|---|---|---|---|
| Regional feeds (`.../data/active_fire/...South_Asia_{24h,48h,7d}.csv`) | last 7 days | **No** | current-season monitoring; verified live |
| Area API (`/api/area/csv/{KEY}/{SOURCE}/{w,s,e,n}/{days≤5}/{date}`) | full history | **Yes (free)** | historical panel + model training |
| Archive Download tool export | full history | Earthdata login | bulk manual fallback |

Historical sources: `MODIS_SP`, `VIIRS_SNPP_SP`, `VIIRS_NOAA20_SP` (standard processing).
VIIRS-SNPP archive starts 2012; MODIS from 2000. Implemented in `src/fire_policy/firms.py`.

Cross-check option: **SAGE-IGP** agricultural-fire emissions inventory, Harvard Dataverse
`doi:10.7910/DVN/JUMXOL` (2003–2018, Punjab/Haryana/UP/Bihar) — district-aggregatable.

## 2. Weather — CONTROLS / FEATURES (Open-Meteo)

Free, keyless. ERA5 archive (~1940→present, ~5-day lag) + 16-day forecast.
Drivers pulled: wind speed/gusts (smoke dispersion vs. stagnation), relative humidity &
precipitation (can residue burn at all?), temperature, ET₀ (dryness). See `weather.py`.

## 3. Subsidy uptake — the TREATMENT (the hard part)

**Bottom line: a true district×year measure exists but must be assembled from 2–3 sources.**
State×year is the published ceiling for the central scheme's aggregates.

### 3a. DBT Agri-Machinery MIS — authoritative, national, district×FY *(browser)*
- Portal: https://agrimachinery.nic.in/Index/Index
- Target allocation (district): https://www.agrimachinery.nic.in/Index/TargetInfo
- Beneficiary report (district drill): https://www.agrimachinery.nic.in/Report/Report/BeneficiaryReport
- CHC subsidy approved (district drill): https://www.agrimachinery.nic.in/Reports/ReportTemplate.aspx?Page=SubSidyApprovedCHC_Drill1
- Organized by scheme × implement × state × district × block × financial year (CRM ~2018-19→present).
- **Difficulty: HARD** — interactive ASP.NET report viewers; needs a headless-browser scraper or manual per-state pulls. No one-click CSV.

### 3b. Haryana — strongest district-level state source *(browser)*
- CRM dashboard: https://agriharyana.gov.in/dashboardcrm ; CRM portal: https://www.agriharyanacrm.com/
- District target letters & beneficiary lists under `agriharyana.gov.in/data/...` (CRM 2023-24 → 2026-27; some are large scanned PDFs → OCR).
- **Granularity: district×year (beneficiary-level).** Difficulty: medium.

### 3c. Punjab — no clean dept dataset; use pollution-board plans *(browser)*
- Punjab Dept of Agriculture CRM page is a dead link; district machine tables live inside
  **PPCB / CAQM stubble-burning Action Plans** (PDF), e.g.
  https://ppcb.punjab.gov.in/ (Action Plans + Action Taken Reports).
- PPCB FAQ (verified) state total: **1,38,022 CRM machines; 24,736 custom hiring centres** (state aggregate).
- **Granularity: district×year inside PDFs.** Difficulty: medium (PDF table extraction).

### 3d. Validation only — Parliament answers / PIB
- Lok Sabha/Rajya Sabha answers (https://sansad.in) and PIB (https://pib.gov.in) table funds
  released & machines distributed at **state×year** — the granularity ceiling; use to sanity-check
  assembled district totals.

**Proxy ladder (if full district×year is too costly):**
1. CRM machines *targeted/sanctioned* per district-year (DBT `TargetInfo` + HR letters).
2. Custom Hiring Centres established per district-year (DBT CHC reports).
3. Beneficiary counts per district-year.
4. State×year machines/funds (validation fallback only).

## 4. Panel scaffold + controls

- **Dipoppa & Gulzar (2024), *Nature* 634:1125** — `doi:10.1038/s41586-024-08046-z`
  (PDF: nature.com/articles/s41586-024-08046-z.pdf; PMC11525172). **CC-BY** replication package:
  district×year South-Asia **burning panel + covariates + code**. Its treatment is *bureaucrat
  incentives*, not subsidies — so use it as the **panel/method scaffold**, then overlay CRM treatment.
- **SHRUG** (devdatalab.org/shrug) — harmonized `shrid` district keys + `core keys` for
  cross-census district split/merge harmonization (essential for a 2016–2023 panel). Open.
- **DES "District-wise season-wise crop production" (APY)** — district×year paddy **area** (1997→);
  native portals block fetch; mirrored on Kaggle (`abhinand05/crop-production-in-india`) **(browser)**.
- **ICRISAT District Level Database** (data.icrisat.org/dld) — district×year crop area/yield
  ~1966–2017 **(host was down during recon — plan a mirror)**.
- **India State & District Evolution DB 1872–2025** — Harvard Dataverse `doi:10.7910/DVN/D1AGUR`
  for split/merge bookkeeping.

## 5. Key papers

1. Jack, Jayachandran, Kala, Pande (2025), "Money (Not) to Burn," *AER: Insights* 7(1):39–55
   (NBER WP 30690; RCT AEARCTR-0004508) — Punjab PES cash RCT; village geographies + fire methods; openICPSR replication.
2. Keil et al. (2020), "…is the Happy Seeder a profitable alternative?" *Int. J. Agric. Sustainability*,
   `doi:10.1080/14735903.2020.1834277` — Happy Seeder adoption economics (Punjab/Haryana survey).
3. Gupta, Ridhima (2012), SANDEE WP — evaluates the machinery-technology policy response directly (check data appendix).
4. Krishna (2023), "Economics of Crop Residue Management," *Annu. Rev. Resour. Econ.*,
   `doi:10.1146/annurev-resource-101422-090019` — synthesis; mine for adoption/subsidy data citations.
