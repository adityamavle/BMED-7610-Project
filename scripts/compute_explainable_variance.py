import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from voxelwise_tutorials.io import get_data_home, load_hdf5_array
from voxelwise_tutorials.utils import explainable_variance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shortclips" / "explainable_variance"


def main() -> None:
    directory = get_data_home(dataset="shortclips")
    print(directory)

    subject = "S01"

    file_name = os.path.join(directory, "responses", f"{subject}_responses.hdf")
    Y_test = load_hdf5_array(file_name, key="Y_test")
    print("(n_repeats, n_samples_test, n_voxels) =", Y_test.shape)

    ev = explainable_variance(Y_test)
    print("(n_voxels,) =", ev.shape)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / f"{subject}_ev.npy", ev)

    voxel_1 = np.argmax(ev)
    time = np.arange(Y_test.shape[1]) * 2
    plt.figure(figsize=(10, 3))
    plt.plot(time, Y_test[:, :, voxel_1].T, color="C0", alpha=0.5)
    plt.plot(time, Y_test[:, :, voxel_1].mean(0), color="C1", label="average")
    plt.xlabel("Time (sec)")
    plt.title("Voxel with large explainable variance (%.2f)" % ev[voxel_1])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_high_ev_voxel.png", dpi=150)
    plt.close()

    voxel_2 = np.argmin(ev)
    plt.figure(figsize=(10, 3))
    plt.plot(time, Y_test[:, :, voxel_2].T, color="C0", alpha=0.5)
    plt.plot(time, Y_test[:, :, voxel_2].mean(0), color="C1", label="average")
    plt.xlabel("Time (sec)")
    plt.title("Voxel with low explainable variance (%.2f)" % ev[voxel_2])
    plt.yticks([])
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_low_ev_voxel.png", dpi=150)
    plt.close()

    plt.figure()
    plt.hist(ev, bins=np.linspace(0, 1, 100), log=True, histtype="step")
    plt.xlabel("Explainable variance")
    plt.ylabel("Number of voxels")
    plt.title("Histogram of explainable variance")
    plt.grid("on")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{subject}_ev_histogram.png", dpi=150)
    plt.close()

    summary = {
        "subject": subject,
        "shape": tuple(int(x) for x in Y_test.shape),
        "ev_shape": tuple(int(x) for x in ev.shape),
        "ev_min": float(ev.min()),
        "ev_max": float(ev.max()),
        "ev_mean": float(ev.mean()),
        "ev_median": float(np.median(ev)),
        "argmax_voxel": int(voxel_1),
        "argmin_voxel": int(voxel_2),
    }
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
