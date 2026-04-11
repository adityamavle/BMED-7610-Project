import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shortclips_baseline.feature_model import fit_single_feature_space_model


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shortclips" / "contrast"


def main() -> None:
    fit_single_feature_space_model(
        feature_space="contrast",
        output_dir=OUTPUT_DIR,
        response_preprocessing="visual_zscore",
        score_vmax=0.4,
    )


if __name__ == "__main__":
    main()
