import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voxelwise_tutorials.io import get_data_home


FRAMES_PER_TR = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype a contrast-based feature from one Shortclips stimulus "
            "run by aggregating framewise contrast into TR-level samples."
        )
    )
    parser.add_argument(
        "--stimulus-file",
        default="stimuli/train_01.hdf",
        help="Relative path under the shortclips data directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="Number of frames to process at once.",
    )
    return parser.parse_args()


def compute_frame_contrast(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float32) / 255.0
    gray = (
        0.299 * frames[..., 0]
        + 0.587 * frames[..., 1]
        + 0.114 * frames[..., 2]
    )
    return gray.std(axis=(1, 2))


def main() -> None:
    args = parse_args()
    data_home = Path(get_data_home(dataset="shortclips"))
    stimulus_path = data_home / args.stimulus_file

    if not stimulus_path.exists():
        raise FileNotFoundError(
            f"Stimulus file not found: {stimulus_path}. "
            "Download it first with scripts/download_shortclips_stimuli.py."
        )

    output_dir = PROJECT_ROOT / "outputs" / "shortclips" / "contrast"
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(stimulus_path, "r") as handle:
        stimuli = handle["stimuli"]
        n_frames = stimuli.shape[0]
        if n_frames % FRAMES_PER_TR != 0:
            raise ValueError(
                f"Expected frame count divisible by {FRAMES_PER_TR}, got {n_frames}."
            )

        frame_contrast = np.empty(n_frames, dtype=np.float32)
        for start in range(0, n_frames, args.batch_size):
            stop = min(start + args.batch_size, n_frames)
            frame_contrast[start:stop] = compute_frame_contrast(stimuli[start:stop])

    tr_contrast = frame_contrast.reshape(-1, FRAMES_PER_TR).mean(axis=1, keepdims=True)

    stem = Path(args.stimulus_file).stem
    np.save(output_dir / f"{stem}_contrast_framewise.npy", frame_contrast)
    np.save(output_dir / f"{stem}_contrast_tr.npy", tr_contrast)
    np.savetxt(
        output_dir / f"{stem}_contrast_tr.csv",
        tr_contrast,
        delimiter=",",
        fmt="%.8f",
        header="contrast",
        comments="",
    )

    print(f"Stimulus file: {stimulus_path}")
    print(f"Frames: {n_frames}")
    print(f"Framewise contrast shape: {frame_contrast.shape}")
    print(f"TR contrast shape: {tr_contrast.shape}")
    print(f"TR contrast mean: {float(tr_contrast.mean()):.6f}")
    print(f"TR contrast std: {float(tr_contrast.std()):.6f}")
    print(f"Saved: {output_dir / f'{stem}_contrast_framewise.npy'}")
    print(f"Saved: {output_dir / f'{stem}_contrast_tr.npy'}")
    print(f"Saved: {output_dir / f'{stem}_contrast_tr.csv'}")


if __name__ == "__main__":
    main()
