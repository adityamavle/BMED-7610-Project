# Project Agents Guide

## Project
Replicate the Gallant Lab Shortclips voxelwise encoding tutorial before proposing or implementing any extensions.

Primary references:
- `https://gallantlab.org/voxelwise_tutorials/pages/index.html`
- `https://gallantlab.org/voxelwise_tutorials/pages/voxelwise_modeling.html`
- `https://gallantlab.org/voxelwise_tutorials/notebooks/shortclips/README.html`
- `https://gin.g-node.org/gallantlab/shortclips`
- `https://doi.gin.g-node.org/10.12751/g-node.vy1zjd/`

## Part 1
Reproduce the tutorial results on the two tutorial feature spaces, exactly following the tutorial workflow:
- `motion_energy`
- `wordnet`

This first milestone is a strict replication task. Do not add new feature spaces, alternative datasets, new model families, or hypothesis extensions until this baseline is complete and documented.

## Scope For Part 1
Agents working on Part 1 should:
- use the Gallant Lab Shortclips dataset
- follow the tutorial preprocessing and data-loading assumptions
- fit the same voxelwise encoding models used in the tutorial
- evaluate the same feature spaces the tutorial uses
- preserve the tutorial's train/test logic and evaluation metrics
- produce figures or summary outputs comparable to the tutorial

Agents working on Part 1 should not:
- add efficient-coding features yet
- substitute a different dataset
- introduce eye-tracking, EEG, epilepsy, or animal data
- change the regression method unless the tutorial itself requires it
- optimize for novelty before the baseline is reproduced

## Expected Technical Flow
1. Download or access the Shortclips tutorial data subset.
2. Load the fMRI responses and tutorial-provided feature spaces.
3. Reproduce the single-feature-space modeling steps.
4. Reproduce the joint or comparative modeling steps used in the tutorial.
5. Confirm that results are qualitatively consistent with the tutorial outputs.
6. Record any deviations, missing dependencies, or reproducibility issues.

## Success Criteria For Part 1
Part 1 is complete when:
- both `motion_energy` and `wordnet` pipelines run end-to-end
- voxelwise prediction scores are produced without custom methodological changes
- outputs are qualitatively consistent with the tutorial
- the replication procedure is documented clearly enough to rerun

## Deliverables For Part 1
- a runnable replication workflow
- a short log of environment setup and data requirements
- saved outputs or figures for `motion_energy` and `wordnet`
- a brief note stating what matched the tutorial and what did not

## Part 2
After the baseline is replicated, identify and evaluate a different feature space that extends the Shortclips voxelwise modeling setup.

The default direction for Part 2 is an efficient-coding feature space derived from natural-scene statistics, but Part 2 may also begin with a comparison of candidate feature spaces before implementation.

## Scope For Part 2
Agents working on Part 2 should:
- start only after Part 1 is complete
- define the scientific motivation for the new feature space
- explain how the new feature space differs from `motion_energy` and `wordnet`
- ensure the feature space can be computed from the Shortclips stimuli
- first check whether the candidate feature space is plausibly useful with respect to explainable variance before committing to full model training
- integrate the new feature space into the same voxelwise encoding framework
- compare standalone and joint performance against the tutorial baselines

Agents working on Part 2 should not:
- change the baseline replication outputs retroactively to fit the new idea
- mix in unrelated modalities unless explicitly approved
- use a different dataset unless the research question changes and that change is documented

## Candidate Direction For Part 2
Preferred first extension:
- an efficient-coding feature space based on interpretable stimulus statistics such as contrast, edge density, spatial entropy, orientation entropy, temporal surprise, or compressibility

## Success Criteria For Part 2
Part 2 is complete when:
- one new feature space is clearly defined and justified
- the feature extraction pipeline is reproducible
- the new feature space is evaluated in the same Shortclips voxelwise framework
- results are compared against `motion_energy` and `wordnet`
- any unique predictive contribution is documented clearly

## Part 2 Evaluation Order
For any proposed new feature space in Part 2:
1. Derive the feature space from the Shortclips stimuli.
2. Run an initial screening step against explainable variance to judge whether the feature is plausible or worth pursuing.
3. If the screening looks reasonable, fit the voxelwise encoding model with the same train/test logic used in Part 1.
4. Compare both mean voxelwise `R^2` and spatial patterns of prediction against the tutorial baselines.

## Working Rule
If there is any tradeoff between exact tutorial replication and experimentation, choose exact replication for Part 1.
