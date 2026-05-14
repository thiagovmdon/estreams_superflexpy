# Repository for **"Assessing the Impact of Geological Map Detail on Process-Based and Data-Driven Hydrological Models"**

by do Nascimento et al. (2026) — *Water Resources Research*

---

## What is this about?

How much does the resolution of a geological map actually matter for streamflow prediction in ungauged basins? That is the central question of this study. We test five levels of geological information — from no geology at all, through global and continental datasets, all the way up to high-resolution regional maps — inside two very different modelling frameworks: a process-based model (SuperflexPy) and a data-driven model (LSTM via NeuralHydrology).

The study covers **130 catchments** across two river systems: the **Moselle** (108 gauges, including Luxembourg, French, Belgium and German tributaries) and the **Garonne** (22 French gauges). Both basins have strong geological gradients, from crystalline basement in the Vosges and Massif Central to sedimentary plains (do Nascimento et al., 2025), making them good test cases for this kind of experiment.

The short answer: yes, better geological catchment attributes helps in both model types and particularly when predicting in ungauged basins.

> do Nascimento, T. V. M., Rudlang, J., Gnann, S., Seibert, J., Hrachowitz, M., and Fenicia, F.: How do geological map details influence the identification of geology-streamflow relationships in large-sample hydrology studies?, Hydrol. Earth Syst. Sci., 29, 7173–7200, https://doi.org/10.5194/hess-29-7173-2025, 2025.


---

## Modelling setup

**Two model types:**

- **Process-based (PB):** SuperflexPy with three HRU types (high / medium / low permeability), one per geology class. HRU weights come directly from the permeability fractions of each map. Calibrated with SCE-UA (SPOTPY library) on a computing cluster, 7 groups for Moselle and 2 for Garonne (leave-one-group-out spatial cross-validation).
- **Data-driven (LSTM):** NeuralHydrology `cudalstm` with 128 hidden units, trained with static geological attributes as additional inputs. 5-fold spatial cross-validation × 5 random seeds.

> Dal Molin, M., Kavetski, D., & Fenicia, F. (2021). SuperflexPy 1.3.0: an open-source Python framework for building, testing, and improving conceptual hydrological models. Geoscientific Model Development, 14(11), 7047–7072. https://doi.org/10.5194/gmd-14-7047-2021

> Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library for Deep Learningresearch in hydrology. Journal of Open Source Software, 7(71), 4050. https://doi.org/10.21105/joss.04050

**Five geology experiments:**

| Experiment | Label | Description |
|---|---|---|
| No geology | `nog` | Baseline — no permeability attributes |
| Random geology | `ran` | Randomly sampled permeability fractions (reproducibility control) |
| Global scale | `glo` | GLiM global lithological map (Hartmann & Moosdorf, 2012) |
| Continental scale | `con` | IHME-1500 |
| Regional scale | `reg` | High-resolution national geological maps (e.g., BD LISA for France) — the most detailed |

**Performance metric:** modified NSE on square-root transformed flows (expo = 0.5), which gives more balanced weight to low and high flows compared to the standard NSE.

**Precipitation correction (Garonne only):** EStreams precipitation is known to underestimate high-elevation rainfall in the Garonne. A catchment-specific correction factor `k_pre = exp(0.0107 + 2.46×10⁻⁴ × elevation_mean)` is derived from a regression against CAMELS-FR long-term mean precipitation and applied multiplicatively to all Garonne daily precipitation time series before model input. The corrected climatology is stored in `results/precipitation_correction/garonne_clim_with_kpre.csv` and used in both the process-based (Part-2a) and LSTM (Part-2b) input preparation steps.

> Clerc‐Schwarzenbach, F., & do Nascimento, T. V. M. (2026). Evaluating E‐OBS forcing data for large‐sample hydrology using model performance diagnostics. HydrologyandEarthSystemSciences, 30(1), 119–140. https://doi.org/10.5194/hess‐30‐119‐2026

---

## Repository structure

```
estreams_superflexpy/
├── code/                   ← all Jupyter notebooks
│   └── 00_cluster/         ← Python scripts for HPC calibration (SuperflexPy + SPOTPY)
├── data/                   ← input CSV/xlsx files and model inputs (.npy)
├── results/                ← simulation outputs (.nc), figures, and tables
└── environments/           ← environment.yml and requirements.txt
```

---

## Workflow

The notebooks are designed to be run in order. Each one picks up where the previous left off.

### Step 0 — Regional geology for the Garonne
- [Part-0](./code/Part-0-exporting-geology-regional-garonne.ipynb) — extract lithological fractions from the BD LISA shapefile for the 22 Garonne catchments → produces `estreams_geology_garonne_regional_attributes.csv`

### Step 1 — Filter catchments and define groups
- [Part-1a (Moselle)](./code/Part-1a-folds-PB-Moselle-filtering-preprocessing.ipynb) — quality filtering, network delineation, 7-fold CV group assignments → `network_estreams_moselle_108_gauges.csv`
- [Part-1b (Garonne)](./code/Part-1b-folds-PB-Garonne-filtering-preprocessing.ipynb) — same for Garonne (2-fold CV) → `network_estreams_garonne_22_gauges.csv`

### Step 2 — Export model inputs
- [Part-2a (Process-based)](./code/Part-2a-model-PB-export-input-files.ipynb) — build forcing/observation arrays and HRU weight `.npy` files for each geology experiment and group → `data/models/input/`; also derives the Garonne precipitation correction (`k_pre`)
- [Part-2b (LSTM)](./code/Part-2b-model-LSTM-export-input-files.ipynb) — build `attributes.csv`, basin list files, and the 125 NeuralHydrology config files; applies `k_pre` correction to Garonne precipitation

### Step 3 — Calibration (process-based only)
> The actual parameter search (SCE-UA) is run on a HPC cluster using the scripts in `code/00_cluster/`. The notebooks below assume calibrated parameters are already available in `results/groups/`.

- [Part-3a (Moselle)](./code/Part-3a-model-PB-Moselle-calibration.ipynb) — run forward model with best parameters, save calibration-period simulations
- [Part-3b (Garonne)](./code/Part-3b-model-PB-Garonne-calibration.ipynb) — same for Garonne

### Step 4 — Evaluation simulations (process-based)
- [Part-4a — Moselle, geology experiments](./code/Part-4a-model-PB-Moselle-evaluation-geology.ipynb)
- [Part-4b — Moselle, no-geology benchmark](./code/Part-4b-model-PB-Moselle-evaluation-no-geology.ipynb)
- [Part-4c — Moselle, random geology](./code/Part-4c-model-PB-Moselle-evaluation-random.ipynb)
- [Part-4d — Garonne, geology experiments](./code/Part-4d-model-PB-Garonne-evaluation-geology.ipynb)
- [Part-4e — Garonne, no-geology benchmark](./code/Part-4e-model-PB-Garonne-evaluation-no-geology.ipynb)
- [Part-4f — Garonne, random geology](./code/Part-4f-model-PB-Garonne-evaluation-random.ipynb)

### Step 5 — Figures and analysis
- [Part-5 (Process-based)](./code/Part-5-figures-and-analysis-Bucket.ipynb) — all PB model figures, performance tables, Wilcoxon tests, SI tables
- [Part-6 (LSTM)](./code/Part-6-figures-and-analysis-LSTM.ipynb) — all LSTM figures, performance tables, Wilcoxon tests, SI tables

---

## Data

All input data needed to run the notebooks:

- **EStreams v1.4** (streamflow time series + gauge metadata + catchment shapefiles): https://doi.org/10.5281/zenodo.17598150
- **Filtered catchment attributes** (do Nascimento et al., 2025): https://github.com/thiagovmdon/LSH-quality_geology
- **Supporting Information** — regional geology for Moselle, calibration parameter CSVs (`results/groups/`), and all simulation outputs (`results/sim/`, `results/LSTM/`):

>do Nascimento, T. V. M. (2026). Data from "Assessing the Impact of Geological Map Detail on Process-Based and Data-Driven Hydrological Models" (0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18392387

Set the `path_estreams` and `path_data` variables in the Configurations cell of each notebook before running. For calibration parameters and simulation outputs, download from the SI Zenodo above and place them in the corresponding `results/` subdirectories — each has a `readme.md` with placement instructions.

> **Note on `observations.npy`:** The observed streamflow arrays (`data/models/input/subset_*/observations.npy`) are excluded from this repository due to redistribution licence restrictions. They are derived directly from EStreams and are regenerated automatically when Part-2a is run from scratch with the raw EStreams data downloaded.

> **Tip — downloading raw streamflow time series:** If you want to use open-source Python APIs to download the original gauge discharge records directly, the **RivRetrieve** Python package provides a unified interface to download streamflow data from multiple national hydrological services, including the French and Belgium portals used in this study. See: https://github.com/kratzert/RivRetrieve-Python

---

## Environment

```bash
# With conda (recommended)
conda env create -f environments/environment.yml
conda activate estreams_superflexpy

# Or with pip
pip install -r environments/requirements.txt
```

Key packages: `Python 3.9`, `pandas`, `numpy`, `xarray`, `geopandas`, `rasterio`, `matplotlib`, `seaborn`, `scipy`, `spotpy`, `superflexpy`, `neuralhydrology`, `hydroanalysis`, `tqdm`

---

## Citation

If you use this code or data, please cite the accompaining paper:

> do Nascimento, T. V. M., Rudlang, J., Gnann, S., Seibert, J., Hrachowitz, M., & Fenicia,F. (2026).Assessing the impact of geological map detail on process‐based and data‐driven hydrological models. Water Resources Research,  62, e2025WR042375. https://doi.org/10.1029/2025WR042375
 
---

## Contact
All code was written in Python by the author, and code improvements were assisted by AI tools during development.

Thiago Nascimento — thiago.nascimento@eawag.ch  
Eawag, Swiss Federal Institute of Aquatic Science and Technology