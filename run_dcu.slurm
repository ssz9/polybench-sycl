#!/bin/bash

#SBATCH -c 7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:4
#SBATCH --time=10:00:00

ulimit -c 0

source ${REPODIR}/env_dcu.sh

export ONEAPI_DEVICE_SELECTOR=hip:0

echo ${TIMESTAMP}
echo BIN=${BIN}
echo SLURM_JOB_ID: ${SLURM_JOB_ID}

$BIN
