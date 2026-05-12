# Simulation Output Files

This directory contains the **NetCDF simulation outputs** from both the SuperflexPy process-based model and the NeuralHydrology LSTM, used in Part-5 and Part-6 for performance evaluation and figure generation.

**These files are not tracked in this GitHub repository.** Download them from the Supporting Information Zenodo archive:

> Medeiros do Nascimento, T. V. (2026). Data from "Assessing the Impact of Geological Map Detail on Process-Based and Data-Driven Hydrological Models" (0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18392387

Place the downloaded folders directly under `results/sim/` before running Part-5 or Part-6 notebooks.

---

## Directory structure

```
results/sim/
├── sim_garonne/
│   ├── calibration/
│   │   ├── simu_cal_Group_1.nc
│   │   └── simu_cal_Group_2.nc
│   ├── space/
│   │   ├── simu_Group_1.nc
│   │   └── simu_Group_2.nc
│   └── space-time/
│       ├── simu_Group_1.nc
│       └── simu_Group_2.nc
├── sim_moselle/
│   ├── calibration/
│   │   ├── simu_cal_Group_1.nc  ...  simu_cal_Group_7.nc
│   ├── space/
│   │   ├── simu_Group_1.nc  ...  simu_Group_7.nc
│   └── space-time/
│       ├── simu_Group_1.nc  ...  simu_Group_7.nc
└── sim_lstm/
    ├── space/
    ├── time/
    └── space-time/
```

---

## Description

- **`sim_garonne/` and `sim_moselle/`** — SuperflexPy outputs for each catchment, split by transfer type (calibration, space-only transfer, space–time transfer) and group.
- **`sim_lstm/`** — Aggregated LSTM predictions in NetCDF format, derived from the raw NeuralHydrology run directories in `results/LSTM/` (see that directory's readme for details on the raw runs).
