#!/bin/bash
# One-shot CHTC setup + submit. Run ON a CHTC access point, from the repo root:
#
#     bash chtc/submit_all.sh                                  # FULL sweep, NetID = whoami
#     bash chtc/submit_all.sh mynetid                          # full sweep, explicit NetID
#     bash chtc/submit_all.sh mynetid sweep_c50_seed01 ...     # only these sessions
#
# Session filtering exists for staged waves: the DEFAULT /staging quota
# (100 GB / 1,000 files) fits at most TWO full-voltage sessions at a time
# (~15.4 GB + 402 files each). Request >=400 GB and >=10,000 files for the
# full 20-session sweep.
#
# Does, in order: check /staging allocation -> patch the YOUR_NETID
# placeholders -> build + stage the container (if not already staged) ->
# write jobs.txt + repo.tar.gz -> condor_submit. Idempotent: safe to re-run.
set -euo pipefail

NETID="${1:-$(whoami)}"
shift $(( $# > 0 ? 1 : 0 ))
SESSIONS=("$@")                 # optional session-name filter
STAGING="/staging/${NETID}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "${STAGING}" ]; then
    echo "ERROR: ${STAGING} does not exist."
    echo "Request a /staging allocation (>= 400 GB for this sweep) first:"
    echo "  https://chtc.cs.wisc.edu/uw-research-computing/file-avail-largedata"
    exit 1
fi

echo "== patching placeholders for ${NETID}"
sed -i "s|/staging/YOUR_NETID|/staging/${NETID}|g" "${HERE}/job.sh" "${HERE}/generate.sub"

if [ ! -f "${STAGING}/neuron-sim.sif" ]; then
    echo "== building container (once)"
    # CHTC profiles point the apptainer cache/tmp at dirs that may not exist
    # yet ("failed to create build parent dir: ... .apptainer_cache")
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${HOME}/.apptainer_cache}"
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${HOME}/.apptainer_tmp}"
    mkdir -p "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"
    if ! apptainer build "${HERE}/neuron-sim.sif" "${HERE}/neuron.def"; then
        echo "ERROR: apptainer build failed on this access point."
        echo "Build it via CHTC's interactive build job instead:"
        echo "  https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc"
        echo "then: cp neuron-sim.sif ${STAGING}/  and re-run this script."
        exit 1
    fi
    cp "${HERE}/neuron-sim.sif" "${STAGING}/"
    echo "== container staged at ${STAGING}/neuron-sim.sif"
else
    echo "== container already staged at ${STAGING}/neuron-sim.sif"
fi

mkdir -p "${STAGING}/neuron_sweeps" "${HERE}/logs"

echo "== writing jobs.txt + repo.tar.gz"
if [ "${#SESSIONS[@]}" -gt 0 ]; then
    echo "   (session filter: ${SESSIONS[*]})"
    ( cd "${HERE}/.." && python3 chtc/make_manifest.py --sessions "${SESSIONS[@]}" )
else
    ( cd "${HERE}/.." && python3 chtc/make_manifest.py )
fi

echo "== submitting"
cd "${HERE}"
condor_submit generate.sub
echo
condor_q
echo
echo "Watch with: condor_q   |   outputs land in ${STAGING}/neuron_sweeps/<session>/"
