"""fire_policy — stubble-burning fires vs. crop-residue subsidy policy.

Two questions:
  1. Causal: did the Happy Seeder / CRM equipment subsidy reduce farm fires?
     (difference-in-differences across districts and time)
  2. Predictive: where will fires spike next burning season?
     (spatiotemporal early-warning model)

Data:
  - Fire detections: NASA FIRMS (MODIS + VIIRS thermal anomalies)
  - Weather: Open-Meteo (ERA5 archive + forecast)
  - Policy / agriculture: district-level CRM subsidy + crop-calendar data
"""

__version__ = "0.1.0"
