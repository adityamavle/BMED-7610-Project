import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from himalaya.backend import set_backend
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.viz import plot_alphas_diagnostic
from sklearn.decomposition import PCA
from sklearn.model_selection import check_cv
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from voxelwise_tutorials.io import get_data_home, load_hdf5_array
from voxelwise_tutorials.utils import generate_leave_one_run_out
from voxelwise_tutorials.wordnet import (
    apply_cmap,
    correct_coefficients,
    load_wordnet,
    plot_wordnet_graph,
    scale_to_rgb_cube,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shortclips" / "wordnet"


def main() -> None:
    directory = get_data_home(dataset="shortclips")
    print(directory)

    subject = "S01"

    file_name = os.path.join(directory, "responses", f"{subject}_responses.hdf")
    Y_train = load_hdf5_array(file_name, key="Y_train")
    Y_test = load_hdf5_array(file_name, key="Y_test")

    print("(n_samples_train, n_voxels) =", Y_train.shape)
    print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)

    Y_test = Y_test.mean(0)
    print("(n_samples_test, n_voxels) =", Y_test.shape)

    Y_train = np.nan_to_num(Y_train)
    Y_test = np.nan_to_num(Y_test)

    feature_space = "wordnet"
    file_name = os.path.join(directory, "features", f"{feature_space}.hdf")
    X_train = load_hdf5_array(file_name, key="X_train")
    X_test = load_hdf5_array(file_name, key="X_test")

    print("(n_samples_train, n_features) =", X_train.shape)
    print("(n_samples_test, n_features) =", X_test.shape)

    run_onsets = load_hdf5_array(file_name, key="run_onsets")
    print(run_onsets)

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

    _ = pipeline.fit(X_train, Y_train)
    scores = pipeline.score(X_test, Y_test)
    print("(n_voxels,) =", scores.shape)
    scores = backend.to_numpy(scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{subject}_wordnet_scores.npy", scores)

    mapper_file = os.path.join(directory, "mappers", f"{subject}_mappers.hdf")
    try:
        from voxelwise_tutorials.viz import plot_3d_flatmap_from_mapper, plot_flatmap_from_mapper
    except ModuleNotFoundError as exc:
        plot_flatmap_from_mapper = None
        plot_3d_flatmap_from_mapper = None
        print(f"Skipping flatmap plotting because a visualization dependency is missing: {exc}")

    if plot_flatmap_from_mapper is not None:
        plot_flatmap_from_mapper(scores, mapper_file, vmin=0, vmax=0.4)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{subject}_wordnet_scores_flatmap.png", dpi=150)
        plt.close()

    best_alphas = backend.to_numpy(pipeline[-1].best_alphas_)
    np.save(OUTPUT_DIR / f"{subject}_wordnet_best_alphas.npy", best_alphas)
    plot_alphas_diagnostic(best_alphas=best_alphas, alphas=alphas)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_wordnet_alpha_diagnostic.png", dpi=150)
    plt.close()

    primal_coef = pipeline[-1].get_primal_coef()
    primal_coef = backend.to_numpy(primal_coef)
    print("(n_delays * n_features, n_voxels) =", primal_coef.shape)

    primal_coef /= np.linalg.norm(primal_coef, axis=0)[None]
    primal_coef *= np.sqrt(np.maximum(0, scores))[None]

    delayer = pipeline.named_steps["delayer"]
    primal_coef_per_delay = delayer.reshape_by_delays(primal_coef, axis=0)
    print("(n_delays, n_features, n_voxels) =", primal_coef_per_delay.shape)
    del primal_coef

    average_coef = np.mean(primal_coef_per_delay, axis=0)
    print("(n_features, n_voxels) =", average_coef.shape)
    del primal_coef_per_delay

    np.save(OUTPUT_DIR / f"{subject}_wordnet_average_coef.npy", average_coef)

    pca = PCA(n_components=4)
    pca.fit(average_coef.T)
    components = pca.components_
    print("(n_components, n_features) =", components.shape)
    print("PCA explained variance =", pca.explained_variance_ratio_)

    _, wordnet_categories = load_wordnet(directory=directory)
    components = correct_coefficients(components.T, wordnet_categories).T
    components -= components.mean(axis=1)[:, None]
    components /= components.std(axis=1)[:, None]
    np.save(OUTPUT_DIR / f"{subject}_wordnet_components.npy", components)

    first_component = components[0]
    node_sizes = np.abs(first_component)
    node_colors = apply_cmap(first_component, vmin=-2, vmax=2, cmap="coolwarm", n_colors=2)
    plot_wordnet_graph(node_colors=node_colors, node_sizes=node_sizes)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_wordnet_first_component_graph.png", dpi=150)
    plt.close()

    average_coef_transformed = pca.transform(average_coef.T).T
    print("(n_components, n_voxels) =", average_coef_transformed.shape)
    np.save(OUTPUT_DIR / f"{subject}_wordnet_average_coef_transformed.npy", average_coef_transformed)

    if plot_flatmap_from_mapper is not None:
        vmax = np.percentile(np.abs(average_coef_transformed), 99.9)
        plot_flatmap_from_mapper(
            average_coef_transformed[0],
            mapper_file,
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{subject}_wordnet_first_component_flatmap.png", dpi=150)
        plt.close()

    next_three_components = components[1:4].T
    node_sizes = np.linalg.norm(next_three_components, axis=1)
    node_colors = scale_to_rgb_cube(next_three_components)
    print("(n_nodes, n_channels) =", node_colors.shape)

    plot_wordnet_graph(node_colors=node_colors, node_sizes=node_sizes)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_wordnet_rgb_components_graph.png", dpi=150)
    plt.close()

    if plot_3d_flatmap_from_mapper is not None:
        voxel_colors = scale_to_rgb_cube(average_coef_transformed[1:4].T, clip=3).T
        print("(n_channels, n_voxels) =", voxel_colors.shape)
        plot_3d_flatmap_from_mapper(
            voxel_colors[0],
            voxel_colors[1],
            voxel_colors[2],
            mapper_file=mapper_file,
            vmin=0,
            vmax=1,
            vmin2=0,
            vmax2=1,
            vmin3=0,
            vmax3=1,
        )
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{subject}_wordnet_rgb_components_flatmap.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
