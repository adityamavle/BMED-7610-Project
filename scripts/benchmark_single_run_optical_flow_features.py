import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortclips_baseline.optical_flow import (
    build_parser,
    extract_tr_optical_flow_features,
    flow_params_from_args,
)
from voxelwise_tutorials.io import get_data_home


def main() -> None:
    parser = build_parser("Benchmark optical-flow feature extraction on one Shortclips run.")
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

    optical_flow_tr, summary = extract_tr_optical_flow_features(
        stimuli_file=Path(stimuli_file),
        size=tuple(args.size),
        batch_size=args.batch_size,
        grid_shape=tuple(args.grid_shape),
        flow_params=flow_params_from_args(args),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.run_name).stem
    summary["estimated_seconds_for_12_train_runs"] = summary["total_seconds"] * 12
    summary["estimated_hours_for_12_train_runs"] = (summary["total_seconds"] * 12) / 3600
    summary["gpu_toggle_note"] = (
        "This optical-flow benchmark uses OpenCV Farneback flow on the CPU. "
        "There is no built-in GPU toggle in this script."
    )

    npy_path = save_dir / f"{stem}_optical_flow_tr.npy"
    summary_path = save_dir / f"{stem}_optical_flow_benchmark_summary.json"

    import numpy as np

    np.save(npy_path, optical_flow_tr)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
