import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from moten.io import imagearray2luminance
from scipy.signal import decimate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voxelwise_tutorials.io import get_data_home, load_hdf5_array


FRAMES_PER_TR = 30
EXPECTED_TRAIN_TRS = 300
EXPECTED_TEST_TRS = 270
TRAIN_RUN_NAMES = [f"train_{index:02d}.hdf" for index in range(12)]
TEST_RUN_NAME = "test.hdf"
REFERENCE_FEATURE_SPACES = ("wordnet", "motion_energy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tutorial-format contrast feature space for Shortclips from "
            "all train runs and the held-out test run."
        )
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=(96, 96),
        metavar=("HEIGHT", "WIDTH"),
        help="Spatial size used for luminance conversion before feature extraction.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Patch size for patchwise RMS contrast on the resized luminance frames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of stimulus frames to convert to luminance per batch.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Output HDF path. Defaults to <shortclips_data_home>/features/contrast.hdf "
            "so the modeling scripts can load it like the tutorial features."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing contrast feature HDF.",
    )
    return parser.parse_args()


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


def compute_patchwise_rms_contrast(
    luminance: np.ndarray,
    patch_size: int,
    eps: float = 1e-6,
) -> np.ndarray:
    n_frames, height, width = luminance.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image size {(height, width)} must be divisible by patch_size={patch_size}."
        )

    grid_h = height // patch_size
    grid_w = width // patch_size

    patches = luminance.reshape(
        n_frames,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    ).transpose(0, 1, 3, 2, 4)

    patch_means = patches.mean(axis=(-1, -2), keepdims=True)
    patch_stds = patches.std(axis=(-1, -2))
    rms_contrast = patch_stds / (patch_means[..., 0, 0] + eps)

    global_mean = luminance.mean(axis=(1, 2), keepdims=True)
    global_std = luminance.std(axis=(1, 2), keepdims=True)
    global_rms = global_std / (global_mean + eps)

    return np.concatenate(
        [
            global_rms.reshape(n_frames, 1),
            rms_contrast.reshape(n_frames, grid_h * grid_w),
        ],
        axis=1,
    ).astype(np.float32)


def decimate_to_tr(feature_matrix: np.ndarray, frames_per_tr: int = FRAMES_PER_TR) -> np.ndarray:
    if feature_matrix.shape[0] % frames_per_tr != 0:
        raise ValueError(
            f"Frame count {feature_matrix.shape[0]} is not divisible by {frames_per_tr}."
        )
    return decimate(feature_matrix, frames_per_tr, ftype="fir", axis=0).astype(np.float32)


def extract_tr_contrast_features(
    stimuli_file: Path,
    size: tuple[int, int],
    patch_size: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    with h5py.File(stimuli_file, "r") as handle:
        if "stimuli" not in handle:
            raise KeyError(f"{stimuli_file} does not contain a 'stimuli' dataset.")
        raw_shape = tuple(int(value) for value in handle["stimuli"].shape)

    t0 = time.perf_counter()
    luminance = compute_luminance(stimuli_file=stimuli_file, size=size, batch_size=batch_size)
    t1 = time.perf_counter()
    contrast_framewise = compute_patchwise_rms_contrast(
        luminance=luminance,
        patch_size=patch_size,
    )
    t2 = time.perf_counter()
    contrast_tr = decimate_to_tr(contrast_framewise)
    t3 = time.perf_counter()

    summary = {
        "stimuli_file": str(stimuli_file),
        "raw_shape": raw_shape,
        "resized_luminance_shape": tuple(int(value) for value in luminance.shape),
        "contrast_framewise_shape": tuple(int(value) for value in contrast_framewise.shape),
        "contrast_tr_shape": tuple(int(value) for value in contrast_tr.shape),
        "luminance_seconds": t1 - t0,
        "contrast_seconds": t2 - t1,
        "decimation_seconds": t3 - t2,
        "total_seconds": t3 - t0,
    }
    return contrast_tr, summary


def resolve_output_file(data_home: Path, output_file: Path | None) -> Path:
    if output_file is not None:
        return output_file
    return data_home / "features" / "contrast.hdf"


def validate_inputs(data_home: Path, run_names: list[str]) -> list[Path]:
    stimuli_dir = data_home / "stimuli"
    stimulus_paths = [stimuli_dir / run_name for run_name in run_names]
    missing_files = [path for path in stimulus_paths if not path.exists()]
    if missing_files:
        missing_lines = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing stimulus HDF files. Download them before building contrast.hdf:\n"
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
    patch_size: int,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_file = output_file.with_suffix(f"{output_file.suffix}.tmp")

    with h5py.File(tmp_output_file, "w") as handle:
        handle.create_dataset("X_train", data=x_train, compression="gzip")
        handle.create_dataset("X_test", data=x_test, compression="gzip")
        handle.create_dataset("run_onsets", data=run_onsets)

        handle.attrs["feature_name"] = "contrast"
        handle.attrs["frames_per_tr"] = FRAMES_PER_TR
        handle.attrs["image_size"] = size
        handle.attrs["patch_size"] = patch_size
        handle.attrs["train_run_names"] = np.array(train_run_names, dtype=h5py.string_dtype())
        handle.attrs["test_run_name"] = test_run_name

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
            x_train_ref = handle["X_train"]
            x_test_ref = handle["X_test"]
            run_onsets_ref = np.asarray(handle["run_onsets"])

            if x_train.shape[0] != x_train_ref.shape[0]:
                raise ValueError(
                    f"Train sample mismatch vs {feature_space}: "
                    f"{x_train.shape[0]} != {x_train_ref.shape[0]}."
                )

            if x_test.shape[0] != x_test_ref.shape[0]:
                raise ValueError(
                    f"Test sample mismatch vs {feature_space}: "
                    f"{x_test.shape[0]} != {x_test_ref.shape[0]}."
                )

            if not np.array_equal(run_onsets, run_onsets_ref):
                raise ValueError(
                    f"run_onsets mismatch vs {feature_space}: "
                    f"{run_onsets.tolist()} != {run_onsets_ref.tolist()}."
                )


def main() -> None:
    args = parse_args()

    data_home = Path(get_data_home(dataset="shortclips"))
    output_file = resolve_output_file(data_home=data_home, output_file=args.output_file)

    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. Use --overwrite to replace it."
        )

    run_names = TRAIN_RUN_NAMES + [TEST_RUN_NAME]
    stimulus_paths = validate_inputs(data_home=data_home, run_names=run_names)

    train_feature_blocks = []
    train_run_summaries = []

    for run_name, stimulus_path in zip(TRAIN_RUN_NAMES, stimulus_paths[: len(TRAIN_RUN_NAMES)]):
        print(f"Processing {run_name} ...")
        contrast_tr, summary = extract_tr_contrast_features(
            stimuli_file=stimulus_path,
            size=tuple(args.size),
            patch_size=args.patch_size,
            batch_size=args.batch_size,
        )
        if contrast_tr.shape[0] != EXPECTED_TRAIN_TRS:
            raise ValueError(
                f"{run_name} produced {contrast_tr.shape[0]} TRs, expected {EXPECTED_TRAIN_TRS}."
            )
        train_feature_blocks.append(contrast_tr)
        train_run_summaries.append(summary)
        print(
            f"  TR shape={contrast_tr.shape}, total_seconds={summary['total_seconds']:.2f}"
        )

    print(f"Processing {TEST_RUN_NAME} ...")
    x_test, test_summary = extract_tr_contrast_features(
        stimuli_file=stimulus_paths[-1],
        size=tuple(args.size),
        patch_size=args.patch_size,
        batch_size=args.batch_size,
    )
    if x_test.shape[0] != EXPECTED_TEST_TRS:
        raise ValueError(
            f"{TEST_RUN_NAME} produced {x_test.shape[0]} TRs, expected {EXPECTED_TEST_TRS}."
        )
    print(f"  TR shape={x_test.shape}, total_seconds={test_summary['total_seconds']:.2f}")

    x_train = np.concatenate(train_feature_blocks, axis=0).astype(np.float32)
    run_lengths = [block.shape[0] for block in train_feature_blocks]
    run_onsets = np.cumsum([0] + run_lengths[:-1]).astype(np.int64)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Expected 2D TR-aligned feature matrices for both train and test.")

    if any(block.shape[1] != x_train.shape[1] for block in train_feature_blocks):
        raise ValueError("Not all train runs produced the same contrast feature dimension.")

    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError(
            f"Train/test feature dimensions differ: {x_train.shape[1]} vs {x_test.shape[1]}."
        )

    validate_against_reference_features(
        data_home=data_home,
        x_train=x_train,
        x_test=x_test,
        run_onsets=run_onsets,
    )

    write_feature_hdf(
        output_file=output_file,
        x_train=x_train,
        x_test=x_test,
        run_onsets=run_onsets,
        train_run_names=TRAIN_RUN_NAMES,
        test_run_name=TEST_RUN_NAME,
        size=tuple(args.size),
        patch_size=args.patch_size,
    )

    total_seconds = sum(summary["total_seconds"] for summary in train_run_summaries) + test_summary[
        "total_seconds"
    ]

    print()
    print(f"Saved contrast feature space to: {output_file}")
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"run_onsets: {run_onsets.tolist()}")
    print(f"Total extraction time (all runs): {total_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
