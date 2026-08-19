#!/bin/bash
# One CHTC job = one recording. Runs inside the neuron-sim container.
# Arguments: $1 = session name (e.g. sweep_c50_seed01), $2 = recording index.
#
# Outputs return via HTCondor file transfer as ONE uniquely-named tar per job
# (CHTC guidance for sub-GB outputs: use file transfer to /home, not /staging).
# The rec-0 job's tar additionally carries network_<session>.npz and
# session_provenance.json. chtc/collect.py extracts the tars automatically.
set -euo pipefail

SESSION="$1"
REC_IDX="$2"

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

# --- package for transfer back (sandbox root; unique name per job) --------
cd ..
TARBALL="${SESSION}_r${REC_IDX}.tar"
REC=$(printf "recording%03d" "${REC_IDX}")
if [ "${REC_IDX}" -eq 0 ]; then
    tar cf "${TARBALL}" -C repo/out "${SESSION}"
else
    tar cf "${TARBALL}" -C repo/out \
        "${SESSION}/${REC}.npz" "${SESSION}/${REC}_summary.json"
fi
echo "packaged $(du -h "${TARBALL}" | cut -f1) -> ${TARBALL}"

echo "job done: $(date -Is)"
