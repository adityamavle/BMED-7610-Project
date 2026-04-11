import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voxelwise_tutorials.io import download_datalad, get_data_home


DATAFILES = [
    "stimuli/train_00.hdf",
    "stimuli/train_01.hdf",
    "stimuli/train_02.hdf",
    "stimuli/train_03.hdf",
    "stimuli/train_04.hdf",
    "stimuli/train_05.hdf",
    "stimuli/train_06.hdf",
    "stimuli/train_07.hdf",
    "stimuli/train_08.hdf",
    "stimuli/train_09.hdf",
    "stimuli/train_10.hdf",
    "stimuli/train_11.hdf",
    "stimuli/test.hdf",
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
