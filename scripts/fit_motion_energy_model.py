import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.viz import plot_alphas_diagnostic
from scipy.stats import zscore
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import get_data_home, load_hdf5_array
from voxelwise_tutorials.utils import generate_leave_one_run_out, zscore_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shortclips" / "motion_energy"


def main() -> None:
    directory = get_data_home(dataset="shortclips")
    print(directory)

    subject = "S01"

    file_name = os.path.join(directory, "responses", f"{subject}_responses.hdf")
    Y_train = load_hdf5_array(file_name, key="Y_train")
    Y_test = load_hdf5_array(file_name, key="Y_test")

    print("(n_samples_train, n_voxels) =", Y_train.shape)
    print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)

    run_onsets = load_hdf5_array(file_name, key="run_onsets")
    Y_train = zscore_runs(Y_train, run_onsets)
    Y_test = zscore(Y_test, axis=1)
    Y_test = Y_test.mean(0)
    Y_test = zscore(Y_test, axis=0)

    print("(n_samples_test, n_voxels) =", Y_test.shape)

    Y_train = np.nan_to_num(Y_train)
    Y_test = np.nan_to_num(Y_test)

    feature_space = "motion_energy"
    file_name = os.path.join(directory, "features", f"{feature_space}.hdf")
    X_train = load_hdf5_array(file_name, key="X_train")
    X_test = load_hdf5_array(file_name, key="X_test")

    print("(n_samples_train, n_features) =", X_train.shape)
    print("(n_samples_test, n_features) =", X_test.shape)

    n_samples_train = X_train.shape[0]
    cv = generate_leave_one_run_out(n_samples_train, run_onsets)
    cv = check_cv(cv)

    scaler = StandardScaler(with_mean=True, with_std=False)
    delayer = Delayer(delays=[1, 2, 3, 4])

    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    alphas = np.logspace(1, 20, 20)
    kernel_ridge_cv = KernelRidgeCV(
        alphas=alphas,
        cv=cv,
        solver_params=dict(
            n_targets_batch=500,
            n_alphas_batch=5,
            n_targets_batch_refit=100,
        ),
    )

    pipeline = make_pipeline(
        scaler,
        delayer,
        kernel_ridge_cv,
    )

    pipeline.fit(X_train, Y_train)

    scores_motion_energy = pipeline.score(X_test, Y_test)
    scores_motion_energy = backend.to_numpy(scores_motion_energy)

    print("(n_voxels,) =", scores_motion_energy.shape)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{subject}_motion_energy_scores.npy", scores_motion_energy)

    try:
        from voxelwise_tutorials.viz import plot_flatmap_from_mapper
    except ModuleNotFoundError as exc:
        plot_flatmap_from_mapper = None
        print(f"Skipping flatmap plotting because a visualization dependency is missing: {exc}")

    mapper_file = os.path.join(directory, "mappers", f"{subject}_mappers.hdf")
    if plot_flatmap_from_mapper is not None:
        ax = plot_flatmap_from_mapper(scores_motion_energy, mapper_file, vmin=0, vmax=0.5)
        ax.figure.tight_layout()
        ax.figure.savefig(OUTPUT_DIR / f"{subject}_motion_energy_scores_flatmap.png", dpi=150)
        plt.close(ax.figure)

    best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
    np.save(OUTPUT_DIR / f"{subject}_motion_energy_best_alphas.npy", best_alphas)
    plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_motion_energy_alpha_diagnostic.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
