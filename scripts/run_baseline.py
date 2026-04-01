import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    "download": PROJECT_ROOT / "scripts" / "download_shortclips.py",
    "ev": PROJECT_ROOT / "scripts" / "compute_explainable_variance.py",
    "wordnet": PROJECT_ROOT / "scripts" / "fit_wordnet_model.py",
    "motion": PROJECT_ROOT / "scripts" / "fit_motion_energy_model.py",
    "banded": PROJECT_ROOT / "scripts" / "fit_banded_ridge_model.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(STEPS),
        default=list(STEPS),
    )
    args = parser.parse_args()

    for step in args.steps:
        script = STEPS[step]
        print(f"=== Running {step}: {script.name} ===")
        subprocess.run([sys.executable, str(script)], check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
