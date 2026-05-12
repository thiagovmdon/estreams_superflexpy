# Precipitation Correction Files

This directory contains the climatology tables used to derive and apply the Garonne precipitation correction in the model input preparation (Part-2a and Part-2b).

## Files

- `garonne_clim.csv` — Garonne catchment climatology derived from EStreams time series (mean precipitation, temperature, observed runoff). Input to Part-2a.
- `moselle_clim.csv` — Same for Moselle catchments. Input to Part-2a.
- `garonne_clim_with_kpre.csv` — Extended version of `garonne_clim.csv` with the derived `k_pre` correction factor added. Output of Part-2a; used as input by Part-2b.

## Correction approach

E-OBS precipitation data in EStreams underestimates high-elevation rainfall in the Garonne (Clerc-Schwarzenbach and do Nascimento, 2025). The correction factor is estimated from a regression of mean elevation against the ratio of CAMELS-FR to EStreams long-term mean precipitation:

```
k_pre = exp(0.0107 + 2.46e-4 * elevation_mean)
```

This factor is applied multiplicatively to the daily Garonne precipitation time series before model inputs are assembled.

> Clerc-Schwarzenbach, F. and do Nascimento, T. V. M.: Evaluating E-OBS forcing data for large-sample hydrology using model performance diagnostics, Hydrol. Earth Syst. Sci., 30, 119–140, https://doi.org/10.5194/hess-30-119-2026, 2026.