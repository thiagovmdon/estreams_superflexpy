# Group-Level Parameter Outputs

This directory documents the **raw group-level parameter files** produced by the **process-based model experiments**.
These files are **direct outputs from the computing cluster** and are not tracked in the GitHub repository.

All files in this directory are provided as **Supporting Information via Zenodo**, together with the full set of **LSTM and process-based model runs** associated with the paper.

---

## Purpose

The files in this directory contain the **best-performing parameter sets** obtained for each:

- catchment
- experiment type
- group definition

They are conceptually analogous to the **raw NeuralHydrology LSTM run directories**, but for the **process-based (SuperflexPy) models**.

---

## Directory structure

results/groups/
├── garonne_best_params_contcompt_Group_2.csv
├── moselle_best_params_contcompt_Group_3_2.csv
├── moselle_best_params_regicompt_Group_4_2.csv
└── ...


---

## File naming convention

Each file follows the pattern:

<catchment>best_params<experiment>Group<group_id>[ _<replicate> ].csv

### Components

- `<catchment>`  
  Catchment name (e.g. `garonne`, `moselle`)

- `<experiment>`  
  Experiment type (e.g. `contcompt`, `regicompt`)

- `Group_<group_id>`  
  Group identifier used in the experimental design

- `_<replicate>` (optional)  
  Replicate or secondary group split, when applicable

### Examples

garonne_best_params_contcompt_Group_2.csv
moselle_best_params_contcompt_Group_3_2.csv
moselle_best_params_regicompt_Group_4_2.csv


---

## File contents

Each CSV file contains:

- basin identifiers
- calibrated parameter values
- associated metadata required to reproduce simulations

These parameter sets are used as inputs for the simulation experiments stored under:

results/sim/

---

## Availability and reproducibility

- These CSV files are **raw, unfiltered outputs** from cluster-based calibration runs.
- They are excluded from GitHub to keep the repository lightweight.
- The **complete set of raw parameter files**, together with:
  - NeuralHydrology LSTM runs
  - process-based model runs
- is available in the **Zenodo Supporting Information** published alongside the paper.

---

## Notes

- Group definitions and experiment design are described in the Methods section of the paper.
- File naming is consistent across catchments to enable automated post-processing.
