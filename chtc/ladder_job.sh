#!/bin/bash
# One CHTC job = one ladder point on one network.
# Arguments: $1 = session, $2 = point index, $3 = ladder config path
# Outputs a single small JSON returned by HTCondor file transfer.
set -euo pipefail

SESSION="$1"
POINT_IDX="$2"
LADDER="${3:-chtc/ladder_mechanisms.json}"

echo "job start: ${SESSION} point ${POINT_IDX} host=$(hostname) $(date -Is)"

tar xzf repo.tar.gz
cd repo
sed -i 's/\r$//' neuron_simulation/mechanisms/*.mod
( cd neuron_simulation && nrnivmodl mechanisms ) > nrnivmodl.log 2>&1 \
    || { cat nrnivmodl.log; exit 1; }

python3 chtc/ladder_one.py \
    --ladder "${LADDER}" \
    --session "${SESSION}" \
    --point-idx "${POINT_IDX}" \
    --save-spikes \
    --out out

# return one tar with the phenotype JSON + the spike npz (spike-only, ~MBs)
cd ..
tar cf "ladder_${SESSION}_p${POINT_IDX}.tar" -C repo/out .
echo "job done: $(date -Is)"
