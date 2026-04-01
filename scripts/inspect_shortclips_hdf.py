import h5py


FILES = [
    "data/shortclips/responses/S01_responses.hdf",
    "data/shortclips/mappers/S01_mappers.hdf",
    "data/shortclips/features/wordnet.hdf",
    "data/shortclips/features/motion_energy.hdf",
]


def main() -> None:
    for file_name in FILES:
        print(f"FILE {file_name}")
        with h5py.File(file_name, "r") as handle:
            def walk(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(name, obj.shape, obj.dtype)

            handle.visititems(walk)
        print("---")


if __name__ == "__main__":
    main()
