#!/usr/bin/bash
#SBATCH --partition=shared
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --job-name=tobac_detection
#SBATCH --mail-type=FAIL
#SBATCH --account=bb1153
#SBATCH --output=%A_%a.out
#SBATCH --array=0-2

set -e
ulimit -s 204800

./process_tobac_features_icon.py 2020-01-01-00:00:00 24 -offset=${SLURM_ARRAY_TASK_ID} -s=/scratch/b/b382728/tobac_features