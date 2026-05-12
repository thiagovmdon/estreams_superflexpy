# Group-Level Calibration Parameter Files

This directory contains the **best-performing parameter sets** from the SCE-UA calibration runs, one file per basin group and geology experiment. These are the direct outputs from the HPC cluster that feed into Part-3 (calibration simulations) and Part-4 (evaluation simulations).

**These files are not tracked in this GitHub repository.** Download them from the Supporting Information Zenodo archive:

> Medeiros do Nascimento, T. V. (2026). Data from "Assessing the Impact of Geological Map Detail on Process-Based and Data-Driven Hydrological Models" (0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18392387

Place the downloaded CSV files directly in this directory (`results/groups/`) before running Part-3 or Part-4 notebooks.

---

## File naming convention

```
{basin}_best_params_{experiment}_Group_{id}[_{replicate}].csv
```

- `{basin}` — `garonne` or `moselle`
- `{experiment}` — experiment label (see table below)
- `Group_{id}` — calibration group index (1–7 for Moselle, 1–2 for Garonne)
- `_{replicate}` — optional suffix marking the second calibration time window

### Experiment labels

| Label | Description |
|---|---|
| `regicompt` | Regional-scale geology (BD LISA / high-resolution national maps) |
| `globcompt` | Global-scale geology (GLiM) |
| `contcompt` | Continental-scale geology (IHME-1500) |
| `nogeot` | No geology — single general HRU, no permeability weighting (baseline) |
| `randomcompt` | Random geology — Dirichlet-sampled permeability fractions |

Additional variant experiments also present in the directory (used in sensitivity analyses):

| Label | Description |
|---|---|
| `regicomptWD` / `globcomptWD` | Regional/global geology with root depth as additional parameter |
| `regicomptkge` / `globcomptkge` | Regional/global geology calibrated with KGE instead of modified NSE |

### Examples

```
moselle_best_params_regicompt_Group_3.csv
moselle_best_params_regicompt_Group_3_2.csv
garonne_best_params_contcompt_Group_1.csv
garonne_best_params_contcompt_Group_1_2.csv
```

---

## File contents

Each CSV contains calibrated parameter values for all basins in that group, plus basin identifiers and associated metadata needed to reproduce the simulations in `results/sim/`.
