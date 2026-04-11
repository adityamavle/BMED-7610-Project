#!/bin/bash
#SBATCH --job-name=flow_hdf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hice1/amavle3/BMED-7610-Project/logs/slurm/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/hice1/amavle3/BMED-7610-Project"
CONDA_SH="/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"
SCRATCH_DATA_ROOT="/home/hice1/amavle3/scratch/voxelwise_tutorials_data"

cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate neuro_env

export VOXELWISE_TUTORIALS_DATA="${SCRATCH_DATA_ROOT}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python scripts/generate_optical_flow_hdf.py --overwrite
