#!/bin/bash
set -e

export PARTITION
export INTERACTIVE
export BIN
export REPODIR
export LOGDIR
export LOGFILE
export TEST_NAME
export TIMESTAMP="${TIMESTAMP:-$(date +'%Y%m%d_%H%M%S')}"

REPODIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
BIN=${1}
LOGDIR=${REPODIR}/logs
TEST_NAME=$(basename "${BIN}")
LOGFILE=${LOGDIR}/${TEST_NAME}_${TIMESTAMP}.log
mkdir -p ${LOGDIR}

if [[ ${INTERACTIVE:-1} == 1 ]]; then
  source /public/home/pacman/repos/polybench-sycl/env_dcu.sh

  export ONEAPI_DEVICE_SELECTOR=hip:*
  export OPAL_OUTPUT_STDERR_FD=/dev/null

  echo ${TIMESTAMP} >> ${LOGFILE}
  echo BIN=${BIN} >> ${LOGFILE}

  srun --mpi=pmix -p ${PARTITION:-debug} --gres=dcu:4 --ntasks-per-node=1 --cpus-per-task=7 -n 1 \
    bash -c " \
    ONEAPI_DEVICE_SELECTOR=hip:0 \
    ${BIN} \
    " | tee -a ${LOGFILE}
else
  sbatch --partition=${PARTITION:-debug} --nodes=1 --job-name=${TEST_NAME} -o ${LOGFILE} -e ${LOGFILE}.err ${REPODIR}/run_dcu.slurm
fi
