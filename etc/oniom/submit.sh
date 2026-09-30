#!/usr/bin/env bash
# Submit an ONIOM task list on this cluster: etc/oniom/submit.sh TASKS [sbatch options]
#
#   kuma, lyra        a job array of single-GPU jobs, one per pending task
#                     (kuma defaults to --partition=h100; on lyra pass e.g. --partition=b200)
#   daint, clariden   one allocation of 8 GPUs (2 nodes) with 8 workers; submit again if
#                     tasks remain when it ends. Needs ONIOM_ACCOUNT, and ONIOM_UENV (default
#                     pytorch/v2.6.0:v1@clariden, view default) for torch and CUDA.
# Extra arguments go to sbatch. Logs: TASKS.d/NAME/log per task, slurm-*.out next to TASKS.
set -euo pipefail
. "$(dirname "$0")/env.sh"
tasks=$(realpath "$1"); shift
here=$(dirname "$tasks")
pending=0
while read -r name _; do
    [[ -z $name || $name == \#* ]] && continue
    [[ -f $tasks.d/$name/done ]] || pending=$((pending + 1))
done < "$tasks"
(( pending > 0 )) || { echo "nothing to do in $tasks"; exit 0; }

case $ONIOM_CLUSTER in
    kuma | lyra)
        # kuma: h100 unless a partition is given (l40s is often full; the MIG slices mig12gb and
        # mig24gb suit the small ML regions, see README.md)
        if [[ $ONIOM_CLUSTER == kuma && " $* " != *" --partition="* && " $* " != *" -p "* ]]; then
            set -- --partition="${ONIOM_PARTITION:-h100}" "$@"
        fi
        sbatch --array=1-"$pending" --chdir="$here" --output="$here/slurm-%A_%a.out" \
            --export=ALL,METAWORK="$METAWORK" "$@" "$METAWORK/etc/oniom/single-gpu.sbatch" "$tasks"
        ;;
    daint | clariden)
        : "${ONIOM_ACCOUNT:?set ONIOM_ACCOUNT to the project to charge}"
        sbatch --account="$ONIOM_ACCOUNT" --uenv="${ONIOM_UENV:-pytorch/v2.6.0:v1@clariden}" \
            --view="${ONIOM_UENV_VIEW:-default}" --chdir="$here" --output="$here/slurm-%j.out" \
            --export=ALL,METAWORK="$METAWORK" "$@" "$METAWORK/etc/oniom/alps-8gpu.sbatch" "$tasks"
        ;;
    *)
        echo "not on a known cluster; running the tasks here, one after another"
        bash "$METAWORK/etc/oniom/worker.sh" "$tasks"
        ;;
esac
echo "$pending pending task(s) submitted on $ONIOM_CLUSTER"
