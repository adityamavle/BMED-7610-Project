from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "shortclips"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "shortclips"

SHORTCLIPS_SOURCE = "https://gin.g-node.org/gallantlab/shortclips"
SHORTCLIPS_FILES = [
    "features/motion_energy.hdf",
    "features/wordnet.hdf",
    "mappers/S01_mappers.hdf",
    "responses/S01_responses.hdf",
]
