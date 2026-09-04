#!/bin/bash
# One CHTC job = one (session, state, method) baseline run, both arms.
# Arguments: $1 = session, $2 = state, $3 = method, $4 = n_recordings
# Spike-only inputs; no NEURON. Output: baselines_<session>_<state>_<method>.tar
set -euo pipefail

SESSION="$1"
STATE="$2"
METHOD="$3"
NREC="${4:-200}"

echo "job start: ${SESSION}/${STATE} ${METHOD} n=${NREC} host=$(hostname) $(date -Is)"

tar xzf repo_baselines.tar.gz
tar xzf "spikes_${SESSION}_${STATE}.tar.gz"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

python3 repo/chtc/baselines_one.py \
    --session "${SESSION}" --state "${STATE}" --method "${METHOD}" \
    --n-recordings "${NREC}" --data data --out out

tar cf "baselines_${SESSION}_${STATE}_${METHOD}.tar" -C out .
echo "job done: $(date -Is)"
