#!/bin/bash
#SBATCH --job-name=shortclips_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hice1/amavle3/BMED-7610-Project/logs/slurm/%x-%j.out

set -euo pipefail

PROJECT_ROOT="/home/hice1/amavle3/BMED-7610-Project"
CONDA_SH="/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh"
SCRATCH_DATA_ROOT="/home/hice1/amavle3/scratch/voxelwise_tutorials_data"
CA_BUNDLE="/etc/ssl/cert.pem"

cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate neuro_env

mkdir -p "${SCRATCH_DATA_ROOT}"
export VOXELWISE_TUTORIALS_DATA="${SCRATCH_DATA_ROOT}"
export GIT_SSL_CAINFO="${CA_BUNDLE}"
export SSL_CERT_FILE="${CA_BUNDLE}"
export REQUESTS_CA_BUNDLE="${CA_BUNDLE}"

python scripts/download_shortclips.py
python scripts/download_shortclips_stimuli.py
