#!/bin/bash
#SBATCH --job-name=tobac_tracking
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=bb1376
#SBATCH --output=tobac.%j.out

# limit stacksize ... adjust to your programs need
# and core file size
ulimit -s 204800
ulimit -c 0

export PATH=/work/bb1153/b380352/tools:$PATH
cmd=./process_tobac_tracking_icon.py
lfile=$*
start_proc_from_list -n 64 -p $cmd $lfile