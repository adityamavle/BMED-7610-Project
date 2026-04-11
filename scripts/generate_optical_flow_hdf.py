import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortclips_baseline.optical_flow import (
    EXPECTED_TEST_TRS,
    EXPECTED_TRAIN_TRS,
    TEST_RUN_NAME,
    TRAIN_RUN_NAMES,
    build_parser,
    extract_tr_optical_flow_features,
    flow_params_from_args,
    resolve_output_file,
    validate_against_reference_features,
    validate_inputs,
    write_feature_hdf,
)
from voxelwise_tutorials.io import get_data_home


def main() -> None:
    parser = build_parser(
        "Build a tutorial-format optical-flow feature space for Shortclips from all train runs and the held-out test run."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "Output HDF path. Defaults to <shortclips_data_home>/features/optical_flow.hdf "
            "so the modeling scripts can load it like the tutorial features."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_home = Path(get_data_home(dataset="shortclips"))
    output_file = resolve_output_file(data_home=data_home, output_file=args.output_file)
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_file}. Use --overwrite to replace it."
        )

    flow_params = flow_params_from_args(args)
    run_names = TRAIN_RUN_NAMES + [TEST_RUN_NAME]
    stimulus_paths = validate_inputs(data_home=data_home, run_names=run_names)

    train_blocks = []
    train_summaries = []
    for run_name, stimulus_path in zip(TRAIN_RUN_NAMES, stimulus_paths[: len(TRAIN_RUN_NAMES)]):
        print(f"Processing {run_name} ...")
        tr_features, summary = extract_tr_optical_flow_features(
            stimuli_file=stimulus_path,
            size=tuple(args.size),
            batch_size=args.batch_size,
            grid_shape=tuple(args.grid_shape),
            flow_params=flow_params,
        )
        if tr_features.shape[0] != EXPECTED_TRAIN_TRS:
            raise ValueError(
                f"{run_name} produced {tr_features.shape[0]} TRs, expected {EXPECTED_TRAIN_TRS}."
            )
        train_blocks.append(tr_features)
        train_summaries.append(summary)
        print(f"  TR shape={tr_features.shape}, total_seconds={summary['total_seconds']:.2f}")

    print(f"Processing {TEST_RUN_NAME} ...")
    x_test, test_summary = extract_tr_optical_flow_features(
        stimuli_file=stimulus_paths[-1],
        size=tuple(args.size),
        batch_size=args.batch_size,
        grid_shape=tuple(args.grid_shape),
        flow_params=flow_params,
    )
    if x_test.shape[0] != EXPECTED_TEST_TRS:
        raise ValueError(
            f"{TEST_RUN_NAME} produced {x_test.shape[0]} TRs, expected {EXPECTED_TEST_TRS}."
        )
    print(f"  TR shape={x_test.shape}, total_seconds={test_summary['total_seconds']:.2f}")

    x_train = np.concatenate(train_blocks, axis=0).astype(np.float32)
    run_lengths = [block.shape[0] for block in train_blocks]
    run_onsets = np.cumsum([0] + run_lengths[:-1]).astype(np.int64)

    if any(block.shape[1] != x_train.shape[1] for block in train_blocks):
        raise ValueError("Not all train runs produced the same optical-flow feature dimension.")
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
        grid_shape=tuple(args.grid_shape),
        flow_params=flow_params,
    )

    total_seconds = sum(summary["total_seconds"] for summary in train_summaries) + test_summary[
        "total_seconds"
    ]

    print()
    print(f"Saved optical-flow feature space to: {output_file}")
    print(f"X_train shape: {x_train.shape}")
    print(f"X_test shape: {x_test.shape}")
    print(f"run_onsets: {run_onsets.tolist()}")
    print(f"Total extraction time (all runs): {total_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
