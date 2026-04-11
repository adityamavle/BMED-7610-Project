import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from moten.io import imagearray2luminance
from voxelwise_tutorials.io import get_data_home, load_hdf5_array


FRAMES_PER_TR = 30
EXPECTED_TRAIN_TRS = 300
EXPECTED_TEST_TRS = 270
TRAIN_RUN_NAMES = [f"train_{index:02d}.hdf" for index in range(12)]
TEST_RUN_NAME = "test.hdf"
REFERENCE_FEATURE_SPACES = ("wordnet", "motion_energy")


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV (`cv2`) is required for the optical-flow pipeline. "
            "Install it in `neuro_env`, e.g. `conda install -n neuro_env opencv`."
        ) from exc
    return cv2


def compute_luminance(stimuli_file: Path, size: tuple[int, int], batch_size: int) -> np.ndarray:
    with h5py.File(stimuli_file, "r") as handle:
        n_images = int(handle["stimuli"].shape[0])

    luminance = np.zeros((n_images, *size), dtype=np.float32)
    for start in range(0, n_images, batch_size):
        stop = min(start + batch_size, n_images)
        batch = slice(start, stop)
        images = load_hdf5_array(str(stimuli_file), key="stimuli", slice=batch)
        if images.dtype != np.uint8:
            images = np.int_(np.clip(images, 0, 1) * 255).astype(np.uint8)
        luminance[batch] = imagearray2luminance(images, size=size, dtype=np.float32)
    return luminance


def _validate_grid(size: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    height, width = size
    grid_h, grid_w = grid_shape
    if height % grid_h != 0 or width % grid_w != 0:
        raise ValueError(
            f"Image size {size} must be divisible by grid_shape={grid_shape}."
        )
    return height // grid_h, width // grid_w


def _patch_means(arr: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    n_frames, height, width = arr.shape
    grid_h, grid_w = grid_shape
    patch_h, patch_w = _validate_grid((height, width), grid_shape)
    patches = arr.reshape(
        n_frames,
        grid_h,
        patch_h,
        grid_w,
        patch_w,
    ).transpose(0, 1, 3, 2, 4)
    return patches.mean(axis=(-1, -2)).reshape(n_frames, grid_h * grid_w)


def compute_dense_optical_flow_features(
    luminance: np.ndarray,
    grid_shape: tuple[int, int] = (6, 6),
    frames_per_tr: int = FRAMES_PER_TR,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    cv2 = require_cv2()

    if luminance.ndim != 3:
        raise ValueError(f"Expected luminance array with 3 dimensions, got {luminance.shape}.")
    if luminance.shape[0] < 2:
        raise ValueError("Need at least two frames to compute optical flow.")
    if luminance.shape[0] % frames_per_tr != 0:
        raise ValueError(
            f"Frame count {luminance.shape[0]} is not divisible by {frames_per_tr}."
        )

    n_trs = luminance.shape[0] // frames_per_tr
    n_pairs_per_tr = frames_per_tr - 1
    n_features = 5 + 3 * (grid_shape[0] * grid_shape[1])
    flow_features = np.zeros((n_trs * n_pairs_per_tr, n_features), dtype=np.float32)

    pair_index = 0
    for tr_index in range(n_trs):
        tr_start = tr_index * frames_per_tr
        previous = luminance[tr_start]
        for offset in range(1, frames_per_tr):
            current = luminance[tr_start + offset]
            flow = cv2.calcOpticalFlowFarneback(
                prev=previous,
                next=current,
                flow=None,
                pyr_scale=pyr_scale,
                levels=levels,
                winsize=winsize,
                iterations=iterations,
                poly_n=poly_n,
                poly_sigma=poly_sigma,
                flags=flags,
            )
            vx = flow[..., 0]
            vy = flow[..., 1]
            magnitude, angle = cv2.cartToPolar(vx, vy, angleInDegrees=False)

            magnitude_sum = float(magnitude.sum())
            if magnitude_sum > 0:
                mean_sin_theta = float((np.sin(angle) * magnitude).sum() / magnitude_sum)
                mean_cos_theta = float((np.cos(angle) * magnitude).sum() / magnitude_sum)
            else:
                mean_sin_theta = 0.0
                mean_cos_theta = 0.0

            global_features = np.array(
                [
                    vx.mean(),
                    vy.mean(),
                    magnitude.mean(),
                    mean_sin_theta,
                    mean_cos_theta,
                ],
                dtype=np.float32,
            )
            patch_vx = _patch_means(vx[None, ...], grid_shape=grid_shape)[0]
            patch_vy = _patch_means(vy[None, ...], grid_shape=grid_shape)[0]
            patch_mag = _patch_means(magnitude[None, ...], grid_shape=grid_shape)[0]
            flow_features[pair_index] = np.concatenate([global_features, patch_vx, patch_vy, patch_mag])
            pair_index += 1
            previous = current

    return flow_features


def aggregate_to_tr_mean(
    feature_matrix: np.ndarray,
    frames_per_tr: int = FRAMES_PER_TR,
) -> np.ndarray:
    n_pairs_per_tr = frames_per_tr - 1
    if feature_matrix.shape[0] % n_pairs_per_tr != 0:
        raise ValueError(
            f"Pair count {feature_matrix.shape[0]} is not divisible by {n_pairs_per_tr}."
        )
    return feature_matrix.reshape(-1, n_pairs_per_tr, feature_matrix.shape[1]).mean(axis=1).astype(np.float32)


def extract_tr_optical_flow_features(
    stimuli_file: Path,
    size: tuple[int, int],
    batch_size: int,
    grid_shape: tuple[int, int],
    flow_params: dict,
) -> tuple[np.ndarray, dict]:
    with h5py.File(stimuli_file, "r") as handle:
        if "stimuli" not in handle:
            raise KeyError(f"{stimuli_file} does not contain a 'stimuli' dataset.")
        raw_shape = tuple(int(value) for value in handle["stimuli"].shape)

    t0 = time.perf_counter()
    luminance = compute_luminance(stimuli_file=stimuli_file, size=size, batch_size=batch_size)
    t1 = time.perf_counter()
    framewise_features = compute_dense_optical_flow_features(
        luminance=luminance,
        grid_shape=grid_shape,
        frames_per_tr=FRAMES_PER_TR,
        **flow_params,
    )
    t2 = time.perf_counter()
    tr_features = aggregate_to_tr_mean(framewise_features)
    t3 = time.perf_counter()

    summary = {
        "stimuli_file": str(stimuli_file),
        "raw_shape": raw_shape,
        "resized_luminance_shape": tuple(int(value) for value in luminance.shape),
        "optical_flow_pairwise_shape": tuple(int(value) for value in framewise_features.shape),
        "optical_flow_tr_shape": tuple(int(value) for value in tr_features.shape),
        "grid_shape": tuple(int(value) for value in grid_shape),
        "flow_params": flow_params,
        "luminance_seconds": t1 - t0,
        "optical_flow_seconds": t2 - t1,
        "aggregation_seconds": t3 - t2,
        "total_seconds": t3 - t0,
    }
    return tr_features, summary


def resolve_output_file(data_home: Path, output_file: Path | None) -> Path:
    if output_file is not None:
        return output_file
    return data_home / "features" / "optical_flow.hdf"


def validate_inputs(data_home: Path, run_names: list[str]) -> list[Path]:
    stimuli_dir = data_home / "stimuli"
    stimulus_paths = [stimuli_dir / run_name for run_name in run_names]
    missing_files = [path for path in stimulus_paths if not path.exists()]
    if missing_files:
        missing_lines = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing stimulus HDF files. Download them before building optical_flow.hdf:\n"
            f"{missing_lines}"
        )
    return stimulus_paths


def write_feature_hdf(
    output_file: Path,
    x_train: np.ndarray,
    x_test: np.ndarray,
    run_onsets: np.ndarray,
    train_run_names: list[str],
    test_run_name: str,
    size: tuple[int, int],
    grid_shape: tuple[int, int],
    flow_params: dict,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_file = output_file.with_suffix(f"{output_file.suffix}.tmp")
    with h5py.File(tmp_output_file, "w") as handle:
        handle.create_dataset("X_train", data=x_train, compression="gzip")
        handle.create_dataset("X_test", data=x_test, compression="gzip")
        handle.create_dataset("run_onsets", data=run_onsets)
        handle.attrs["feature_name"] = "optical_flow"
        handle.attrs["frames_per_tr"] = FRAMES_PER_TR
        handle.attrs["image_size"] = size
        handle.attrs["grid_shape"] = grid_shape
        handle.attrs["train_run_names"] = np.array(train_run_names, dtype=h5py.string_dtype())
        handle.attrs["test_run_name"] = test_run_name
        handle.attrs["flow_params_json"] = json.dumps(flow_params)
    os.replace(tmp_output_file, output_file)


def validate_against_reference_features(
    data_home: Path,
    x_train: np.ndarray,
    x_test: np.ndarray,
    run_onsets: np.ndarray,
) -> None:
    features_dir = data_home / "features"
    for feature_space in REFERENCE_FEATURE_SPACES:
        feature_file = features_dir / f"{feature_space}.hdf"
        if not feature_file.exists():
            raise FileNotFoundError(f"Reference feature file is missing: {feature_file}.")
        with h5py.File(feature_file, "r") as handle:
            if x_train.shape[0] != handle["X_train"].shape[0]:
                raise ValueError(
                    f"Train sample mismatch vs {feature_space}: "
                    f"{x_train.shape[0]} != {handle['X_train'].shape[0]}."
                )
            if x_test.shape[0] != handle["X_test"].shape[0]:
                raise ValueError(
                    f"Test sample mismatch vs {feature_space}: "
                    f"{x_test.shape[0]} != {handle['X_test'].shape[0]}."
                )
            run_onsets_ref = np.asarray(handle["run_onsets"])
            if not np.array_equal(run_onsets, run_onsets_ref):
                raise ValueError(
                    f"run_onsets mismatch vs {feature_space}: "
                    f"{run_onsets.tolist()} != {run_onsets_ref.tolist()}."
                )


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-name", default="train_00.hdf")
    parser.add_argument("--size", nargs=2, type=int, default=(96, 96), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--grid-shape", nargs=2, type=int, default=(6, 6), metavar=("ROWS", "COLS"))
    parser.add_argument("--pyr-scale", type=float, default=0.5)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--winsize", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--poly-n", type=int, default=5)
    parser.add_argument("--poly-sigma", type=float, default=1.2)
    parser.add_argument("--flags", type=int, default=0)
    return parser


def flow_params_from_args(args: argparse.Namespace) -> dict:
    return {
        "pyr_scale": float(args.pyr_scale),
        "levels": int(args.levels),
        "winsize": int(args.winsize),
        "iterations": int(args.iterations),
        "poly_n": int(args.poly_n),
        "poly_sigma": float(args.poly_sigma),
        "flags": int(args.flags),
    }
