#!/bin/bash
#SBATCH --job-name=bench_flow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/hice1/amavle3/BMED-7610-Project/logs/slurm/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/hice1/amavle3/BMED-7610-Project"
CONDA_SH="/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"
SCRATCH_DATA_ROOT="/home/hice1/amavle3/scratch/voxelwise_tutorials_data"

cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate neuro_env

export VOXELWISE_TUTORIALS_DATA="${SCRATCH_DATA_ROOT}"
python scripts/benchmark_single_run_optical_flow_features.py --run-name train_00.hdf
