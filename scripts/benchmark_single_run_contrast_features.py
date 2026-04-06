import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
from moten.io import imagearray2luminance
from scipy.signal import decimate
from voxelwise_tutorials.io import get_data_home, load_hdf5_array


N_FRAMES_PER_TR = 30


def compute_luminance(stimuli_file, size=(96, 96), batch_size=100):
    with h5py.File(stimuli_file, "r") as f:
        n_images = f["stimuli"].shape[0]

    luminance = np.zeros((n_images, *size), dtype=np.float32)
    for start in range(0, n_images, batch_size):
        batch = slice(start, start + batch_size)
        images = load_hdf5_array(stimuli_file, key="stimuli", slice=batch)
        if images.dtype != "uint8":
            images = np.int_(np.clip(images, 0, 1) * 255).astype(np.uint8)
        luminance[batch] = imagearray2luminance(images, size=size, dtype=np.float32)
    return luminance


def compute_patchwise_rms_contrast(luminance, patch_size=16, eps=1e-6):
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

    contrast_features = np.concatenate(
        [
            global_rms.reshape(n_frames, 1),
            rms_contrast.reshape(n_frames, grid_h * grid_w),
        ],
        axis=1,
    ).astype(np.float32)
    return contrast_features


def decimate_to_tr(feature_matrix, frames_per_tr=N_FRAMES_PER_TR):
    return decimate(feature_matrix, frames_per_tr, ftype="fir", axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark contrast-feature extraction on one Shortclips run."
    )
    parser.add_argument("--run-name", default="train_00.hdf")
    parser.add_argument("--size", nargs=2, type=int, default=(96, 96))
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--save-dir",
        default=str(Path("outputs") / "benchmarks"),
        help="Directory where timing summary and sample outputs are saved.",
    )
    args = parser.parse_args()

    directory = get_data_home(dataset="shortclips")
    stimuli_file = os.path.join(directory, "stimuli", args.run_name)
    if not os.path.exists(stimuli_file):
        raise FileNotFoundError(
            f"Stimulus file not found: {stimuli_file}. Download it first with git-annex or DataLad."
        )

    with h5py.File(stimuli_file, "r") as f:
        if "stimuli" not in f:
            raise KeyError(f"{stimuli_file} does not contain a 'stimuli' dataset.")
        n_images = int(f["stimuli"].shape[0])
        raw_shape = tuple(int(x) for x in f["stimuli"].shape)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    luminance = compute_luminance(
        stimuli_file=stimuli_file,
        size=tuple(args.size),
        batch_size=args.batch_size,
    )
    t1 = time.perf_counter()

    contrast_framewise = compute_patchwise_rms_contrast(
        luminance=luminance,
        patch_size=args.patch_size,
    )
    t2 = time.perf_counter()

    contrast_tr = decimate_to_tr(contrast_framewise)
    t3 = time.perf_counter()

    summary = {
        "stimuli_file": stimuli_file,
        "raw_shape": raw_shape,
        "resized_luminance_shape": tuple(int(x) for x in luminance.shape),
        "contrast_framewise_shape": tuple(int(x) for x in contrast_framewise.shape),
        "contrast_tr_shape": tuple(int(x) for x in contrast_tr.shape),
        "patch_size": int(args.patch_size),
        "n_images": n_images,
        "luminance_seconds": t1 - t0,
        "contrast_seconds": t2 - t1,
        "decimation_seconds": t3 - t2,
        "total_seconds": t3 - t0,
        "seconds_per_frame_total": (t3 - t0) / n_images,
        "estimated_seconds_for_12_train_runs": (t3 - t0) * 12,
        "estimated_hours_for_12_train_runs": ((t3 - t0) * 12) / 3600,
        "gpu_toggle_note": (
            "This contrast benchmark uses NumPy/PIL-based luminance conversion and "
            "NumPy feature extraction. There is no built-in GPU toggle in this script."
        ),
    }

    stem = Path(args.run_name).stem
    np.save(save_dir / f"{stem}_contrast_tr.npy", contrast_tr)
    with open(save_dir / f"{stem}_contrast_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
