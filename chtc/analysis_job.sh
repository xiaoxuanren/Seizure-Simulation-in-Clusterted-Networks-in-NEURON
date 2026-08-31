#!/bin/bash
# One CHTC job = one duration-grid analysis point (session, n_recordings).
# Arguments: $1 = session, $2 = n_recordings
# Spike-only inputs; no NEURON, no mod compilation. Output: one small JSON.
set -euo pipefail

SESSION="$1"
NREC="$2"

echo "job start: ${SESSION} n=${NREC} host=$(hostname) $(date -Is)"

tar xzf repo_analysis.tar.gz
tar xzf "spikes_${SESSION}.tar.gz"

python3 repo/chtc/analysis_one.py \
    --session "${SESSION}" \
    --n-recordings "${NREC}" \
    --data data \
    --out .

echo "job done: $(date -Is)"
