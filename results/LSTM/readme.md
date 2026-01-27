# LSTM Experiment Results

This directory contains the output of **NeuralHydrology LSTM experiments** for the Moselle catchment, organized by random seed.

## Folder structure

There are **five seed-specific folders**, one per random seed used in the experiments:

- `03_seed28`
- `03_seed33`
- `03_seed44`
- `03_seed55`
- `03_seed77`

Each folder contains **all model runs associated with that seed**.

## Runs per seed

For each seed, a total of **20 runs** were performed, corresponding to the following experiment types:

- `reg` – regional
- `con` – continental
- `glo` – global
- `nog` – no-grouping
- `ran` – random (baseline)

Each experiment type includes multiple runs, resulting in **100 runs total** across all seeds.

## Example path

results/LSTM/03_seed28/moselle_con_seed28_01_1701_160651/

This example corresponds to:
- **Seed:** 28  
- **Experiment type:** continental (`con`)  
- **Run ID:** `01_1701_160651`  
- **Model output:** standard NeuralHydrology result structure (config, metrics, predictions, etc.)

## Notes

- All folders inside each `03_seedXX` directory are **raw NeuralHydrology outputs**.
- This directory is excluded from version control to avoid committing large model artifacts.


## Availability and reproducibility

- These files are **raw, unfiltered outputs** from NeuralHydrology training runs.
- They are excluded from GitHub to keep the repository lightweight.
- The **complete set of raw parameter files**, together with:
  - process-based model runs
- is available in the **Zenodo Supporting Information** published alongside the paper.

---

## Notes

- Folds definitions and experiment design are described in the Methods section of the paper.
- File naming is consistent across catchments to enable automated post-processing.
