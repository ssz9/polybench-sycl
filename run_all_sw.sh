#!/bin/bash
set -e

REPODIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

TESTS=$(find ${REPODIR}/bin -maxdepth 1 -type f -name 'test_*' | sort)

for BIN in ${TESTS}; do
  INTERACTIVE=0 ./run_sw.sh "${BIN}"
  sleep 0.1
done
