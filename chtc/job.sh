#!/bin/bash
# One CHTC job = one recording. Runs inside the neuron-sim container.
# Arguments: $1 = session name (e.g. sweep_c50_seed01), $2 = recording index.
# EDIT STAGING_DIR to your /staging allocation before submitting.
set -euo pipefail

SESSION="$1"
REC_IDX="$2"
STAGING_DIR="/staging/YOUR_NETID/neuron_sweeps"

echo "job start: session=${SESSION} rec=${REC_IDX} host=$(hostname) $(date -Is)"

# --- unpack the code (transferred as repo.tar.gz) -------------------------
tar xzf repo.tar.gz
cd repo

# --- normalize line endings: a tarball built on (or copied via) Windows may
#     carry CRLF .mod files, which nocmodl rejects ------------------------
sed -i 's/\r$//' neuron_simulation/mechanisms/*.mod

# --- compile the NEURON mechanisms (per-job; ~seconds) --------------------
( cd neuron_simulation && nrnivmodl mechanisms ) > nrnivmodl.log 2>&1 \
    || { cat nrnivmodl.log; exit 1; }

# --- run the one recording ------------------------------------------------
python3 chtc/generate_one.py \
    --sweep chtc/sweep_config.json \
    --session "${SESSION}" \
    --rec-idx "${REC_IDX}" \
    --out out

# --- ship outputs to staging, atomically: cp to a temp name, then mv (mv is
#     atomic within a filesystem), so an evicted job can never leave a
#     truncated file under the final name. NOT via HTCondor transfer:
#     ~77 MB/recording, ~308 GB for the whole sweep. ----------------------
DEST="${STAGING_DIR}/${SESSION}"
mkdir -p "${DEST}"
put() {  # put <local-file>
    local base tmp
    base=$(basename "$1")
    tmp="${DEST}/.tmp.$$.${base}"
    cp "$1" "${tmp}"
    mv -f "${tmp}" "${DEST}/${base}"
    echo "staged ${base}"
}
REC=$(printf "recording%03d" "${REC_IDX}")
for f in "out/${SESSION}/${REC}"*; do
    put "${f}"
done
if [ "${REC_IDX}" -eq 0 ]; then
    put "out/${SESSION}/network_${SESSION}.npz"
    put "out/${SESSION}/session_provenance.json"
fi

echo "job done: $(date -Is)"
