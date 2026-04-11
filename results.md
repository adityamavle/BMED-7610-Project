# Results Summary

## Standalone Contrast vs Baselines on `S01`

The standalone contrast feature space underperformed both tutorial baselines overall.

### Mean `R^2`

- `contrast`: `0.0007115848711691797`
- `wordnet`: `0.0032528219744563103`
- `motion_energy`: `0.019858969375491142`

Mean differences:

- `contrast - wordnet`: `-0.0025412371032871306`
- `contrast - motion_energy`: `-0.019147384504321963`

### Distribution Summary

`contrast`

- median `R^2`: `0.0`
- min `R^2`: `-0.14610648155212402`
- max `R^2`: `0.09991490840911865`
- 90th percentile: `0.0043409825302660465`
- 95th percentile: `0.01158167701214552`
- 99th percentile: `0.035425134003162384`
- positive-score voxels: `33135 / 84038`
- fraction positive: `0.39428591827506604`

`wordnet`

- median `R^2`: `0.0`
- min `R^2`: `-0.5556453466415405`
- max `R^2`: `0.48039180040359497`
- 90th percentile: `0.026593834161758423`
- 95th percentile: `0.0661860778927803`
- 99th percentile: `0.2132135033607483`
- positive-score voxels: `39610 / 84038`
- fraction positive: `0.4713343963445108`

`motion_energy`

- median `R^2`: `1.1920928955078125e-07`
- min `R^2`: `-0.12287616729736328`
- max `R^2`: `0.6084827184677124`
- 90th percentile: `0.046162139624357224`
- 95th percentile: `0.1295187622308731`
- 99th percentile: `0.39466941356658936`
- positive-score voxels: `43647 / 84038`
- fraction positive: `0.5193721887717461`

## Interpretation

- Contrast alone does not beat either baseline globally.
- Contrast is weaker than `wordnet` and much weaker than `motion_energy` as a standalone encoding model.
- The remaining scientific question is whether contrast adds unique variance in a joint model rather than whether it is the best standalone model.

## Standalone Optical Flow vs Baselines on `S01`

The standalone optical-flow feature space performed substantially better than the contrast feature space and also outperformed `wordnet` overall, but it did not beat the tutorial `motion_energy` baseline.

The optical-flow feature space used dense OpenCV Farneback flow on resized luminance movies. The saved tutorial-format feature file is `optical_flow.hdf` with:

- `X_train`: `(3600, 113)`
- `X_test`: `(270, 113)`
- `run_onsets`: `[0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300]`
- feature design: global directional/magnitude summaries plus `6 x 6` patchwise `vx`, `vy`, and magnitude summaries

### Mean `R^2`

- `optical_flow`: `0.009855721145868301`
- `contrast`: `0.0007115848711691797`
- `wordnet`: `0.0032528219744563103`
- `motion_energy`: `0.019858969375491142`

Mean differences:

- `optical_flow - contrast`: `0.009144136274699121`
- `optical_flow - wordnet`: `0.006602899171411991`
- `optical_flow - motion_energy`: `-0.01000324822962284`

### Distribution Summary

`optical_flow`

- median `R^2`: `1.1920928955078125e-07`
- min `R^2`: `-0.09366655349731445`
- max `R^2`: `0.39549779891967773`
- 90th percentile: `0.02267465554177761`
- 95th percentile: `0.06466001272201538`
- 99th percentile: `0.19650165736675262`
- positive-score voxels: `43998 / 84038`
- fraction positive: `0.523548870748947`

### Interpretation

- Optical flow is a much stronger standalone Part 2 candidate than contrast.
- Optical flow beats `wordnet` on mean `R^2` and fraction of positive-score voxels.
- Optical flow remains below `motion_energy` on mean `R^2`, upper-tail performance, and maximum voxel score.
- The feature is still scientifically useful because it is directional and interpretable; the key next test is whether optical flow explains unique variance beyond `motion_energy`, especially in motion-sensitive visual regions.
- The main optical-flow prediction-accuracy flatmap is `outputs/shortclips/optical_flow/S01_optical_flow_scores_flatmap.png`.

## Relevant Files

- `outputs/shortclips/contrast/S01_contrast_scores.npy`
- `outputs/shortclips/contrast/S01_contrast_scores_flatmap.png`
- `outputs/shortclips/optical_flow/S01_optical_flow_scores.npy`
- `outputs/shortclips/optical_flow/S01_optical_flow_scores_flatmap.png`
- `outputs/shortclips/wordnet/S01_wordnet_scores.npy`
- `outputs/shortclips/motion_energy/S01_motion_energy_scores.npy`
- `/home/hice1/amavle3/scratch/voxelwise_tutorials_data/shortclips/features/optical_flow.hdf`
