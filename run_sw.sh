#!/bin/bash
set -e

PARTITION=${PARTITION:-q_share}
INTERACTIVE=${INTERACTIVE:-1}
NP=${NP:-1}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPODIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

source ${REPODIR}/env_sw.sh

LOGDIR=${REPODIR}/logs
mkdir -p "${LOGDIR}"

BIN=$1
TEST_NAME=$(basename "${BIN}")
LOGFILE=${LOGDIR}/${TEST_NAME}_${TIMESTAMP}.log

echo "Running ${BIN}"
echo "Log: ${LOGFILE}"

if [[ "${INTERACTIVE}" == "1" ]]; then
  bsub -I -b -q "${PARTITION}" -n "${NP}" -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128 -o "${LOGFILE}" "${BIN}"
else
  bsub -b -q "${PARTITION}" -n "${NP}" -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128 -o "${LOGFILE}" "${BIN}"
fi
