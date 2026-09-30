#!/usr/bin/env bash
# Export the models the ONIOM jobs use into $ONIOM_MODELS (TorchScript, from lab-cosmo/upet
# on Hugging Face): PET-MAD xs v1.5.0 (both jobs) and PET-OMol s v1.0.0 (hemoglobin, per-site
# charge and spin). Needs the upet venv (UPET_PY, default $METAWORK/.venv-upet/bin/python)
# and network access; run it once per cluster, or copy $ONIOM_MODELS over.
set -euo pipefail
. "$(dirname "$0")/env.sh"
: "${UPET_PY:=$METAWORK/.venv-upet/bin/python}"
mkdir -p "$ONIOM_MODELS"
for spec in "pet-mad xs 1.5.0" "pet-omol s 1.0.0"; do
    set -- $spec
    out=$ONIOM_MODELS/$1-$2-v$3.pt
    [[ -f $out ]] && { echo "have $out"; continue; }
    "$UPET_PY" -c "import upet; upet.save_upet(model='$1', size='$2', version='$3', output='$out')"
    echo "wrote $out"
done
