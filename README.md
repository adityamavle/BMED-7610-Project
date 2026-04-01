# BMED 7610 Project

Replication of the Gallant Lab Shortclips voxelwise encoding tutorial, with the baseline focused on the two tutorial feature spaces:

- `motion_energy`
- `wordnet`

Part 1 of this repo is a strict baseline replication. Part 2 is reserved for new feature-space experiments after the baseline is documented and stable.

## Environment

Use the Conda environment:

- `neuro_env`

The local dataset and generated outputs are intentionally ignored by Git.

## Repo Layout

- `scripts/`: runnable entry points for download, baseline fitting, inspection, and feature-space prototyping
- `src/`: small project package utilities
- `sources.md`: reference links for the tutorial and dataset

## Baseline Workflow

1. Download the baseline tutorial subset:
   ```powershell
   C:\Users\adity\anaconda3\envs\neuro_env\python scripts/download_shortclips.py
   ```
2. Compute explainable variance:
   ```powershell
   C:\Users\adity\anaconda3\envs\neuro_env\python scripts/compute_explainable_variance.py
   ```
3. Fit the WordNet model:
   ```powershell
   C:\Users\adity\anaconda3\envs\neuro_env\python scripts/fit_wordnet_model.py
   ```
4. Fit the motion-energy model:
   ```powershell
   C:\Users\adity\anaconda3\envs\neuro_env\python scripts/fit_motion_energy_model.py
   ```
5. Fit the combined banded-ridge comparison:
   ```powershell
   C:\Users\adity\anaconda3\envs\neuro_env\python scripts/fit_banded_ridge_model.py
   ```

Or run the baseline sequence through:

```powershell
C:\Users\adity\anaconda3\envs\neuro_env\python scripts/run_baseline.py
```

## Script Summary

- `scripts/download_shortclips.py`: downloads the baseline feature, mapper, and response files used in Part 1
- `scripts/compute_explainable_variance.py`: computes explainable variance summaries for subject `S01`
- `scripts/fit_wordnet_model.py`: fits the tutorial WordNet ridge model and saves voxelwise test `R^2` outputs
- `scripts/fit_motion_energy_model.py`: fits the tutorial motion-energy ridge model and saves voxelwise test `R^2` outputs
- `scripts/fit_banded_ridge_model.py`: compares joint modeling with banded ridge versus concatenated ridge
- `scripts/run_baseline.py`: convenience entry point for the baseline sequence
- `scripts/inspect_shortclips_hdf.py`: prints HDF dataset keys and shapes for quick inspection
- `scripts/download_shortclips_stimuli.py`: downloads stimulus HDF files for feature-space prototyping
- `scripts/prototype_contrast_feature.py`: prototype extractor for a simple contrast-based feature from one stimulus run

## Notes

- The current code targets `S01`.
- The repo does not commit dataset files or generated results under `data/` or `outputs/`.
