# CRM / Happy-Seeder subsidy TREATMENT data — provenance

Difference-in-differences treatment variable: district-level Crop Residue Management (CRM)
equipment-subsidy rollout/intensity in **Punjab** and **Haryana**, intended window 2015–2022
(bracketing the 2018-19 national CRM scale-up).

**Date accessed: 2026-08-23.** All sources are public official/government (state agriculture
departments, Punjab Pollution Control Board) or the Internet Archive Wayback Machine.
**No numbers are fabricated.** Every value is extracted from the cited PDF/table. Where a
source could not be extracted it is recorded as a gap, not filled.

Tidy schema (all CSVs): `state, district, financial_year, metric, value, unit, source_url, notes`
(long/tidy format — one metric value per row).

---

## FILE 1 — `punjab_crm_district_year.csv`  (330 rows, 22 districts) ⭐ best district source
- **What:** CRM machinery **actually provided under subsidy**, **cumulative 2018-19 → 2021-22**,
  by district × 15 machine types + a per-district `crm_machines_total`.
  Machine types: happy_seeder, super_seeder, paddy_straw_chopper, mulcher, rmb_plough,
  zero_till_drill, super_sms, rotary_slasher, shrub_master, cutter_cum_spreader, rotavator,
  baler, rake, crop_reaper_reaper_binder.
- **Source:** Punjab Pollution Control Board (PPCB), *Action Plan for Control of Stubble Burning
  2022*, **page 27**, table "District Wise CRM Machinery provided under Subsidy 2018-19 to 2021-22".
  URL: https://ppcb.punjab.gov.in/media/documents/FINAL_Action_Plan_Stubble_Burning_29May2022.pdf
  (local copy: `source_pdfs/PPCB_ActionPlan_StubbleBurning_2022.pdf`)
- **Extraction:** `pdfplumber` table extraction from p.27 (clean digital table, not scanned).
- **Validation:** Σ of 22 district grand totals = **90,422**, exactly matching the state CRM-machine
  **baseline "as on 2022-23" = 90,422 on p.18** of the same document, and the Super-SMS column
  total (5,972) matches the p.18 Super-SMS availability figure. Two independent internal cross-checks pass.
- **Caveats:**
  - `financial_year = 2018-19_to_2021-22_cum` — this is a **4-year cumulative stock**, NOT annual flow.
    Year-by-year Punjab district splits are not published in this document.
  - This is **achievement** (machines delivered), a stronger treatment measure than targets.
  - 22 districts; Malerkotla (created 2021) is not yet separated (its area sits within Sangrur).
    Names normalized: "Mukatsar"→Sri Muktsar Sahib, "Fazillka"→Fazilka.

## FILE 2 — `haryana_crm_district_year.csv`  (414 rows, 21 districts)
Two blocks, distinguished by `metric`/`unit`/`financial_year`:

**(a) 2018-19 machine + subsidy TARGETS** (individual farmers) — `financial_year = 2018-19`
- **What:** District × 8 machine-type **physical targets** (`crm_machines_target_*`, unit=machines)
  and matching **subsidy targets** (`crm_subsidy_target_*`, unit=INR_lakh, = 50% cost norm),
  plus per-district `_total`.
- **Source:** Haryana Dept of Agriculture & Farmers Welfare, *"Districtwise Targets of Agricultural
  Implements under New Scheme 'Promotion of Agricultural Mechanization for In-Situ Management of Crop
  Residue in the States of Punjab, Haryana, UP and NCT of Delhi' during the Year 2018-19"*.
  Retrieved via Wayback Machine (original `agriharyana.gov.in/assets/images/whatsnew/Districtwise_Targets_2018-19.pdf`):
  https://web.archive.org/web/20180516224922id_/http://agriharyana.gov.in/assets/images/whatsnew/Districtwise_Targets_2018-19.pdf
  (local copy: `source_pdfs/HR_Districtwise_Targets_CRM_2018-19.pdf`)
- **Validation:** physical Σ = **5,563** machines; subsidy Σ = **3,999.9 ≈ 4,000 lakh (Rs 40 cr)**;
  the physical figures match the parallel "individual farmers" PDF (`HR_Districtwise_Targets_IndividualFarmers_2018-19.pdf`) exactly.

**(b) Custom Hiring Centres (CHCs)** — `financial_year = 2017-18` and `2018-19`
- **What:** `crm_chc_established` (CHCs established during 2017-18, the pre-CRM baseline) and
  `crm_chc_target` (CHCs proposed for 2018-19), unit=CHCs, by district (18 districts).
- **Source:** Haryana Agri Dept, *"District wise Targets for Establishment of Custom Hiring Centres …
  In-Situ Management of Crop Residue … 2018-19"*, via Wayback:
  https://web.archive.org/web/2018id_/http://agriharyana.gov.in/assets/images/whatsnew/Districtwise_targets_for_Establishment_of_Custom_Hiring_Centres_2018-19.pdf
  (local copy: `source_pdfs/HR_Districtwise_CHC_Targets_2018-19.pdf`)
- **Validation:** Σ established 2017-18 = **420**; Σ proposed 2018-19 = **900**.
- **Extraction:** `pdfplumber` text extraction (digital PDFs).
- **Caveats:**
  - Haryana values are **TARGETS/allocations, not verified achievement** (proxy-ladder rung 1).
  - **Only 2018-19** is available (the first CRM year). agriharyana.gov.in's live site hosts only
    2023-24→2026-27 docs (mostly scanned; SMAM not CRM-specific), and the Wayback Machine archived
    agriharyana only for 2018-2019 — so **Haryana 2019-20, 2020-21, 2021-22 district data is a gap**
    from free sources (see blockers).
  - 21 districts in the machine table (Charkhi Dadri, created 2016, was not separately allocated);
    18 in the CHC table. Names normalized to current usage: "Narnaul"→Mahendragarh, "Gurgaon"→Gurugram.

## FILE 3 — `crm_state_year.csv`  (8 rows) — state×year validation / context
- State-total roll-ups of Files 1–2 (Punjab 90,422 cum; Haryana 2018-19: 5,563 machines,
  4,000 lakh, 900 CHCs proposed, 420 CHCs baseline), plus **Punjab annual new-machine targets**
  2022-23=25,000 / 2023-24=30,000 / 2024-25=30,000 from PPCB Action Plan 2022 p.18
  (`crm_machines_target_annual`; outside the 2018-22 window, context only).
- Use for sanity-checking assembled district totals. `district = ALL`.

## Reference — `source_pdfs/DBT_AgriMachinery_MIS_reference.json`
- Public code/metadata harvested from the **DBT Agri-Machinery MIS** AngularJS dropdown API
  (agrimachinery.nic.in): CRM states (Punjab=3, Haryana=6, UP=9, Delhi=7), CRM scheme id = **29**,
  Haryana district codes, and confirmation that **financial years 2018-19 … 2026-27** are queryable.
  These drive the (browser-only) district reports — see blockers.

---

## The single best district×year source found
The **DBT Agri-Machinery MIS** (agrimachinery.nic.in) is the authoritative national source:
CRM data organized by scheme(29) × state × **district** × implement × **financial year (2018-19→)**,
for exactly the 4 CRM states. Its public JSON dropdowns confirm full coverage of the study window,
**but the actual counts are gated**: the JSON data endpoint `Report/Report/GetFilterReport` returns
`"you are not authenticated."`, and the public SSRS report viewers
(`Reports/ReportViewr.aspx?Page=Expenditure_RPT_NEW`, `…=Waiting_List`) are `.aspx` Microsoft
ReportViewer controls that require a real/headless browser (ViewState + cascading AJAX postbacks) —
not scriptable with curl. **Recommended next step for full district×year coverage: drive that SSRS
report with Playwright/Selenium** (scheme=CRM, state=6/3, iterate FY 2018→2022, export the grid).

For an *extracted, verified* district cross-section today, **PPCB Action Plan 2022 p.27 (File 1)**
is the best — all 22 Punjab districts × machine type, cumulative over the whole window, achievement-based.

## Top 3 blockers
1. **DBT MIS district data is login-gated / browser-gated.** The clean JSON report endpoint needs
   officer authentication; the public equivalent is a browser-only SSRS viewer. This is the only
   source with true annual district×year for both states — hence the biggest blocker.
2. **Haryana publishes district CRM docs only for the current cycle; the archive is thin.** The live
   site has 2023-27 (largely scanned, SMAM-mixed) and Wayback captured agriharyana only in 2018-19.
   Result: Haryana district data is limited to **2018-19** here; 2019-2022 is missing from free sources.
3. **CAQM / PIB are not scriptable.** caqm.nic.in is a JS-only SPA (0 static links; would hold both
   states' district tables for later years) and PIB returns HTTP 403 to automated fetches, so
   external state×year validation and later-year CAQM district tables couldn't be pulled.

## Definitional break to respect in the DiD
Punjab (File 1) = **cumulative achievement**; Haryana (File 2) = **single-year 2018-19 targets**.
They are **not** directly comparable levels. For a two-state DiD, either (i) obtain matching
achievement×year for both states via the DBT SSRS route above, or (ii) use each state's series as a
within-state relative-intensity ranking (district share of state total), which is robust to the
target-vs-achievement and cumulative-vs-annual differences.
