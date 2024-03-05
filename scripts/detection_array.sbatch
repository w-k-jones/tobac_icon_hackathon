#!/usr/bin/bash
#SBATCH --partition=shared
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH -n 1
#SBATCH --job-name=tobac_detection
#SBATCH --mail-type=FAIL
#SBATCH --account=bb1153
#SBATCH --output=./logs/%A_%a.out
#SBATCH --array=0-365

set -e
ulimit -s 204800

./process_tobac_features_icon.py 2021-01-01-00:00:00 24 -offset=${SLURM_ARRAY_TASK_ID} -s=/scratch/b/b382728/tobac_features