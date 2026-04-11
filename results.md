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

## Relevant Files

- `outputs/shortclips/contrast/S01_contrast_scores.npy`
- `outputs/shortclips/contrast/S01_contrast_scores_flatmap.png`
- `outputs/shortclips/wordnet/S01_wordnet_scores.npy`
- `outputs/shortclips/motion_energy/S01_motion_energy_scores.npy`
