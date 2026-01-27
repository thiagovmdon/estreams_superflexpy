# Simulation Results

This directory documents the **simulation outputs** used in the study.
All result files are **not tracked in the GitHub repository** due to their size and are instead provided as **Supporting Information via Zenodo**, published alongside the paper.

---

## Directory structure

results/sim/
├── sim_garonne/
├── sim_moselle/
└── sim_lstm/


---

## 1. sim_garonne

**Path:**
results/sim/sim_garonne/


This directory contains **SuperflexPy simulation outputs** for the **Garonne catchment**, stored as NetCDF (`.nc`) files.

### 1.1 Calibration simulations

**Path:**
results/sim/sim_garonne/calibration/

Files:
- `simu_cal_Group_1.nc`
- `simu_cal_Group_2.nc`

Each file corresponds to a **calibration group**, as defined in the experimental setup.

### 1.2 Spatial simulations

**Path:**
results/sim/sim_garonne/space/

Files:
- `simu_Group_1.nc`
- `simu_Group_2.nc`

These simulations apply calibrated parameters across space only.

### 1.3 Space–time simulations

**Path:**
results/sim/sim_garonne/space-time/

Files:
- `simu_Group_1.nc`
- `simu_Group_2.nc`

These outputs correspond to combined **space–time transfer experiments**.

---

## 2. sim_moselle

**Path:**
results/sim/sim_moselle/

This directory mirrors the structure of `sim_garonne`, but for the **Moselle catchment**.

### 2.1 Calibration simulations

**Path:**
results/sim/sim_moselle/calibration/


Files:
- `simu_cal_Group_1.nc`
- `simu_cal_Group_2.nc`
- `simu_cal_Group_3.nc`
- `simu_cal_Group_4.nc`
- `simu_cal_Group_5.nc`
- `simu_cal_Group_6.nc`
- `simu_cal_Group_7.nc`


### 2.2 Spatial simulations

**Path:**
results/sim/sim_moselle/space/

Files:
- `simu_Group_1.nc`
- `simu_Group_2.nc`
- `simu_Group_3.nc`
- `simu_Group_4.nc`
- `simu_Group_5.nc`
- `simu_Group_6.nc`
- `simu_Group_7.nc`

### 2.3 Space–time simulations

**Path:**
results/sim/sim_moselle/space-time/

Files:
- `simu_Group_1.nc`
- `simu_Group_2.nc`
- `simu_Group_3.nc`
- `simu_Group_4.nc`
- `simu_Group_5.nc`
- `simu_Group_6.nc`
- `simu_Group_7.nc`

---

## 3. sim_lstm

**Path:**
results/sim/sim_lstm/

This directory contains **processed LSTM simulation outputs** derived from the NeuralHydrology experiments.

- Outputs are stored in NetCDF format.
- Files aggregate LSTM predictions for use in evaluation and comparison with SuperflexPy simulations.
- The raw NeuralHydrology run directories are documented separately under `results/LSTM/`.

---

## Availability and reproducibility

- All simulation outputs in this directory are provided in the **Zenodo Supporting Information** associated with the paper.
- This GitHub repository contains only the **code and lightweight metadata** required to reproduce the experiments.
- Users should download the Zenodo archive to access the full set of simulation results.

---

## Notes

- Group definitions, experiment design, and evaluation metrics are described in the paper.
- File naming conventions are consistent across catchments to facilitate automated analysis.