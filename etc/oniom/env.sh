# Shared environment for the GROMACS ONIOM jobs (gromacs-oniom-cyclic, gromacs-oniom-hemoglobin).
# Source it: `. etc/oniom/env.sh`. Every value can be set beforehand to override it.
#
#   METAWORK      metawork checkout (default: two levels above this file)
#   ONIOM_CLUSTER kuma | lyra | daint | clariden | local (default: guessed from the host)
#   ONIOM_PY      python with torch + metatomic (the shared .venv, .venv-blackwell on lyra,
#                 the uenv's python on Alps)
#   ONIOM_BUILDS  where build-gromacs.sh installs GROMACS (default: $METAWORK/builds)
#   GMX           single-precision GROMACS with CUDA non-bondeds and metatomic (Torch)
#   GMX_D         double-precision GROMACS with metatomic, MM on the CPU
#   ONIOM_MODELS  exported models (default: $METAWORK/models/oniom), from export-models.sh
#   NT            OpenMP threads per run (default: CPUs given to this task, else 16)

: "${METAWORK:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -z ${ONIOM_CLUSTER:-} ]]; then
    case "${SLURM_CLUSTER_NAME:-$(hostname -f 2>/dev/null || hostname)}" in
        *kuma*) ONIOM_CLUSTER=kuma ;;
        *lyra*) ONIOM_CLUSTER=lyra ;;
        *daint*) ONIOM_CLUSTER=daint ;;
        *clariden*) ONIOM_CLUSTER=clariden ;;
        *) ONIOM_CLUSTER=local ;;
    esac
fi

case $ONIOM_CLUSTER in
    lyra) : "${ONIOM_PY:=$METAWORK/.venv-blackwell/bin/python}" ;;
    daint | clariden) : "${ONIOM_PY:=$(command -v python3)}" ;;  # the uenv's python
    *) : "${ONIOM_PY:=$METAWORK/.venv/bin/python}" ;;
esac

: "${ONIOM_BUILDS:=$METAWORK/builds}"
: "${GMX:=$ONIOM_BUILDS/gromacs-oniom-$ONIOM_CLUSTER-$(uname -m)/bin/gmx}"
: "${GMX_D:=$ONIOM_BUILDS/gromacs-oniom-$ONIOM_CLUSTER-$(uname -m)-double/bin/gmx_d}"
: "${ONIOM_MODELS:=$METAWORK/models/oniom}"
: "${NT:=${SLURM_CPUS_PER_TASK:-16}}"
export METAWORK ONIOM_CLUSTER ONIOM_PY ONIOM_BUILDS GMX GMX_D ONIOM_MODELS NT

# Writes mdp file $2 from template $1 with @MODELS@ replaced by the model directory
oniom_mdp() { sed "s|@MODELS@|$ONIOM_MODELS|g" "$1" > "$2"; }
