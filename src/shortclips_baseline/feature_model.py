import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.viz import plot_alphas_diagnostic
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import get_data_home, load_hdf5_array
from voxelwise_tutorials.utils import generate_leave_one_run_out, zscore_runs


def _contrast_feature_names(n_features: int) -> list[str]:
    if n_features == 37:
        names = ["global_rms"]
        for row in range(6):
            for col in range(6):
                names.append(f"patch_r{row}_c{col}")
        return names
    return [f"feature_{index:03d}" for index in range(n_features)]


def _optical_flow_feature_names(n_features: int) -> list[str]:
    if n_features == 113:
        names = ["global_vx", "global_vy", "global_magnitude", "global_sin_theta", "global_cos_theta"]
        for prefix in ("vx", "vy", "mag"):
            for row in range(6):
                for col in range(6):
                    names.append(f"patch_{prefix}_r{row}_c{col}")
        return names
    return [f"feature_{index:03d}" for index in range(n_features)]


def infer_feature_names(feature_space: str, n_features: int) -> list[str]:
    if feature_space == "contrast":
        return _contrast_feature_names(n_features)
    if feature_space == "optical_flow":
        return _optical_flow_feature_names(n_features)
    return [f"feature_{index:03d}" for index in range(n_features)]


def _save_feature_importance_plot(
    feature_importance: np.ndarray,
    feature_names: list[str],
    output_file: Path,
) -> None:
    plt.figure(figsize=(max(8, len(feature_names) * 0.25), 4))
    x = np.arange(len(feature_names))
    plt.bar(x, feature_importance)
    plt.xticks(x, feature_names, rotation=90)
    plt.ylabel("L2 norm across voxels")
    plt.title("Average coefficient importance by feature")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def _save_component_plot(
    components: np.ndarray,
    feature_names: list[str],
    output_file: Path,
) -> None:
    first_component = components[0]
    plt.figure(figsize=(max(8, len(feature_names) * 0.25), 4))
    x = np.arange(len(feature_names))
    plt.bar(x, first_component)
    plt.xticks(x, feature_names, rotation=90)
    plt.ylabel("Loading")
    plt.title("First PCA component of average coefficients")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


def load_responses(
    directory: str,
    subject: str,
    response_preprocessing: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    response_file = os.path.join(directory, "responses", f"{subject}_responses.hdf")
    Y_train = load_hdf5_array(response_file, key="Y_train")
    Y_test = load_hdf5_array(response_file, key="Y_test")
    response_run_onsets = load_hdf5_array(response_file, key="run_onsets")

    print("(n_samples_train, n_voxels) =", Y_train.shape)
    print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)
    print("response_preprocessing =", response_preprocessing)

    if response_preprocessing == "visual_zscore":
        Y_train = zscore_runs(Y_train, response_run_onsets)
        Y_test = zscore(Y_test, axis=1)
        Y_test = Y_test.mean(0)
        Y_test = zscore(Y_test, axis=0)
    elif response_preprocessing == "mean_only":
        Y_test = Y_test.mean(0)
    else:
        raise ValueError(
            "response_preprocessing must be one of "
            "('visual_zscore', 'mean_only')."
        )

    print("(n_samples_test, n_voxels) =", Y_test.shape)

    Y_train = np.nan_to_num(Y_train)
    Y_test = np.nan_to_num(Y_test)
    return Y_train, Y_test, response_run_onsets


def fit_single_feature_space_model(
    feature_space: str,
    output_dir: Path,
    response_preprocessing: str,
    score_vmax: float = 0.4,
) -> None:
    directory = get_data_home(dataset="shortclips")
    print(directory)

    subject = "S01"
    Y_train, Y_test, response_run_onsets = load_responses(
        directory=directory,
        subject=subject,
        response_preprocessing=response_preprocessing,
    )

    feature_file = os.path.join(directory, "features", f"{feature_space}.hdf")
    X_train = load_hdf5_array(feature_file, key="X_train")
    X_test = load_hdf5_array(feature_file, key="X_test")

    print("(n_samples_train, n_features) =", X_train.shape)
    print("(n_samples_test, n_features) =", X_test.shape)

    feature_run_onsets = load_hdf5_array(feature_file, key="run_onsets")
    print(feature_run_onsets)

    if not np.array_equal(response_run_onsets, feature_run_onsets):
        raise ValueError(
            f"Response and {feature_space} run_onsets differ: "
            f"{response_run_onsets.tolist()} != {feature_run_onsets.tolist()}."
        )

    n_samples_train = X_train.shape[0]
    cv = generate_leave_one_run_out(n_samples_train, response_run_onsets)
    cv = check_cv(cv)

    pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=False),
        Delayer(delays=[1, 2, 3, 4]),
        KernelRidgeCV(
            alphas=np.logspace(1, 20, 20),
            cv=cv,
            solver_params=dict(
                n_targets_batch=500,
                n_alphas_batch=5,
                n_targets_batch_refit=100,
            ),
        ),
    )

    backend = set_backend("torch_cuda", on_error="warn")
    print(backend)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    pipeline.fit(X_train, Y_train)

    scores = pipeline.score(X_test, Y_test)
    scores = backend.to_numpy(scores)
    print("(n_voxels,) =", scores.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{subject}_{feature_space}_scores.npy", scores)

    plot_flatmap_from_mapper = None
    plot_3d_flatmap_from_mapper = None
    try:
        from voxelwise_tutorials.viz import plot_3d_flatmap_from_mapper, plot_flatmap_from_mapper
    except Exception as exc:
        print(f"Skipping flatmap plotting because a visualization dependency is missing: {exc}")

    mapper_file = os.path.join(directory, "mappers", f"{subject}_mappers.hdf")
    if plot_flatmap_from_mapper is not None:
        ax = plot_flatmap_from_mapper(scores, mapper_file, vmin=0, vmax=score_vmax)
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / f"{subject}_{feature_space}_scores_flatmap.png", dpi=150)
        plt.close(ax.figure)

    best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
    np.save(output_dir / f"{subject}_{feature_space}_best_alphas.npy", best_alphas)
    plot_alphas_diagnostic(best_alphas=best_alphas, alphas=np.logspace(1, 20, 20))
    plt.tight_layout()
    plt.savefig(output_dir / f"{subject}_{feature_space}_alpha_diagnostic.png", dpi=150)
    plt.close()

    primal_coef = pipeline[-1].get_primal_coef()
    primal_coef = backend.to_numpy(primal_coef)
    print("(n_delays * n_features, n_voxels) =", primal_coef.shape)
    np.save(output_dir / f"{subject}_{feature_space}_primal_coef.npy", primal_coef)

    coef_norm = np.linalg.norm(primal_coef, axis=0)
    coef_norm[coef_norm == 0] = 1.0
    primal_coef = primal_coef / coef_norm[None]
    primal_coef *= np.sqrt(np.maximum(0, scores))[None]

    delayer = pipeline.named_steps["delayer"]
    primal_coef_per_delay = delayer.reshape_by_delays(primal_coef, axis=0)
    print("(n_delays, n_features, n_voxels) =", primal_coef_per_delay.shape)
    average_coef = np.mean(primal_coef_per_delay, axis=0)
    print("(n_features, n_voxels) =", average_coef.shape)
    np.save(output_dir / f"{subject}_{feature_space}_average_coef.npy", average_coef)

    feature_names = infer_feature_names(feature_space, X_train.shape[1])
    feature_importance = np.linalg.norm(average_coef, axis=1)
    np.save(output_dir / f"{subject}_{feature_space}_feature_importance.npy", feature_importance)
    _save_feature_importance_plot(
        feature_importance=feature_importance,
        feature_names=feature_names,
        output_file=output_dir / f"{subject}_{feature_space}_feature_importance.png",
    )

    n_components = min(4, average_coef.shape[0])
    pca = PCA(n_components=n_components)
    pca.fit(average_coef.T)
    components = pca.components_
    transformed = pca.transform(average_coef.T).T
    print("(n_components, n_features) =", components.shape)
    print("PCA explained variance =", pca.explained_variance_ratio_)
    np.save(output_dir / f"{subject}_{feature_space}_components.npy", components)
    np.save(output_dir / f"{subject}_{feature_space}_average_coef_transformed.npy", transformed)

    _save_component_plot(
        components=components,
        feature_names=feature_names,
        output_file=output_dir / f"{subject}_{feature_space}_first_component_loadings.png",
    )

    if plot_flatmap_from_mapper is not None:
        vmax = np.percentile(np.abs(transformed[0]), 99.9)
        ax = plot_flatmap_from_mapper(
            transformed[0],
            mapper_file,
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / f"{subject}_{feature_space}_first_component_flatmap.png", dpi=150)
        plt.close(ax.figure)

    if plot_3d_flatmap_from_mapper is not None and transformed.shape[0] >= 4:
        rgb = transformed[1:4].T
        rgb = np.clip(rgb, -3, 3)
        rgb = (rgb + 3) / 6
        rgb = rgb.T
        plot_3d_flatmap_from_mapper(
            rgb[0],
            rgb[1],
            rgb[2],
            mapper_file=mapper_file,
            vmin=0,
            vmax=1,
            vmin2=0,
            vmax2=1,
            vmin3=0,
            vmax3=1,
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"{subject}_{feature_space}_rgb_components_flatmap.png", dpi=150)
        plt.close()
