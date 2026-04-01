import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voxelwise_tutorials.io import download_datalad, get_data_home


DATAFILES = [
    "features/motion_energy.hdf",
    "features/wordnet.hdf",
    "mappers/S01_mappers.hdf",
    "responses/S01_responses.hdf",
]
SOURCE = "https://gin.g-node.org/gallantlab/shortclips"


def main() -> None:
    directory = get_data_home(dataset="shortclips")
    print(directory)

    for datafile in DATAFILES:
        local_filename = download_datalad(
            datafile,
            destination=directory,
            source=SOURCE,
        )
        print(local_filename)


if __name__ == "__main__":
    main()
