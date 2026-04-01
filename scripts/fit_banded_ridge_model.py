import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV, MultipleKernelRidgeCV
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer
from scipy.stats import zscore
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import get_data_home, load_hdf5_array
from voxelwise_tutorials.utils import explainable_variance, generate_leave_one_run_out, zscore_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shortclips" / "banded_ridge"


def main() -> None:
    directory = get_data_home(dataset="shortclips")
    print(directory)

    subject = "S01"

    file_name = os.path.join(directory, "responses", f"{subject}_responses.hdf")
    Y_train = load_hdf5_array(file_name, key="Y_train")
    Y_test = load_hdf5_array(file_name, key="Y_test")

    print("(n_samples_train, n_voxels) =", Y_train.shape)
    print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)

    ev = explainable_variance(Y_test)
    mask = ev > 0.1
    print("(n_voxels_mask,) =", mask.sum())

    run_onsets = load_hdf5_array(file_name, key="run_onsets")
    print(run_onsets)

    Y_train = zscore_runs(Y_train, run_onsets)
    Y_test = zscore(Y_test, axis=1)
    Y_test = Y_test.mean(0)
    Y_test = zscore(Y_test, axis=0)

    Y_train = np.nan_to_num(Y_train)
    Y_test = np.nan_to_num(Y_test)

    file_name = os.path.join(directory, "features", "wordnet.hdf")
    X_train_wordnet = load_hdf5_array(file_name, key="X_train").astype("float32")
    X_test_wordnet = load_hdf5_array(file_name, key="X_test").astype("float32")

    file_name = os.path.join(directory, "features", "motion_energy.hdf")
    X_train_motion = load_hdf5_array(file_name, key="X_train").astype("float32")
    X_test_motion = load_hdf5_array(file_name, key="X_test").astype("float32")

    X_train = np.concatenate([X_train_wordnet, X_train_motion], axis=1)
    X_test = np.concatenate([X_test_wordnet, X_test_motion], axis=1)

    print("(n_samples_train, n_features_total) =", X_train.shape)
    print("(n_samples_test, n_features_total) =", X_test.shape)

    n_samples_train = X_train.shape[0]
    cv = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)

    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

    solver = "random_search"
    n_iter = 20
    alphas = np.logspace(1, 20, 20)
    n_targets_batch = 200
    n_alphas_batch = 5
    n_targets_batch_refit = 200

    solver_params = dict(
        n_iter=n_iter,
        alphas=alphas,
        n_targets_batch=n_targets_batch,
        n_alphas_batch=n_alphas_batch,
        n_targets_batch_refit=n_targets_batch_refit,
        progress_bar=True,
    )

    mkr_model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver=solver,
        solver_params=solver_params,
        cv=cv,
    )

    start_wordnet = 0
    stop_wordnet = X_train_wordnet.shape[1]
    stop_motion = stop_wordnet + X_train_motion.shape[1]

    column_kernelizer = ColumnKernelizer(
        [
            (
                "wordnet",
                make_pipeline(
                    StandardScaler(with_mean=True, with_std=False),
                    Delayer(delays=[1, 2, 3, 4]),
                    Kernelizer(),
                ),
                slice(start_wordnet, stop_wordnet),
            ),
            (
                "motion_energy",
                make_pipeline(
                    StandardScaler(with_mean=True, with_std=False),
                    Delayer(delays=[1, 2, 3, 4]),
                    Kernelizer(),
                ),
                slice(stop_wordnet, stop_motion),
            ),
        ]
    )

    pipeline = make_pipeline(
        column_kernelizer,
        mkr_model,
    )

    pipeline.fit(X_train, Y_train[:, mask])
    scores_mask = pipeline.score(X_test, Y_test[:, mask])
    scores_mask = backend.to_numpy(scores_mask)
    print("(n_voxels_mask,) =", scores_mask.shape)

    n_voxels = Y_train.shape[1]
    scores = np.zeros(n_voxels)
    scores[mask] = scores_mask
    print("(n_voxels,) =", scores.shape)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{subject}_banded_ridge_scores.npy", scores)
    np.save(OUTPUT_DIR / f"{subject}_banded_ridge_mask.npy", mask)

    baseline_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Delayer(delays=[1, 2, 3, 4]),
        KernelRidgeCV(
            alphas=alphas,
            cv=cv,
            solver_params=dict(
                n_targets_batch=500,
                n_alphas_batch=5,
                n_targets_batch_refit=100,
            ),
        ),
    )

    baseline_pipeline.fit(X_train, Y_train[:, mask])
    scores_baseline_mask = baseline_pipeline.score(X_test, Y_test[:, mask])
    scores_baseline_mask = backend.to_numpy(scores_baseline_mask)

    scores_baseline = np.zeros(n_voxels)
    scores_baseline[mask] = scores_baseline_mask
    np.save(OUTPUT_DIR / f"{subject}_concatenated_ridge_scores.npy", scores_baseline)

    plt.figure(figsize=(5, 5))
    plt.hist2d(scores_baseline, scores, bins=100, range=[[-0.1, 0.6], [-0.1, 0.6]], cmap="magma")
    plt.plot([-0.1, 0.6], [-0.1, 0.6], color="white", linewidth=1)
    plt.xlabel("KernelRidgeCV")
    plt.ylabel("MultipleKernelRidgeCV")
    plt.title("Generalization R2 scores")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_banded_vs_baseline_hist2d.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
