#!/usr/bin/env bash
# Work through a task list: worker.sh TASKS [MAX]
#
# TASKS has one task per line, "NAME command ...", run with bash from the directory of TASKS
# ('#' starts a comment). Workers claim tasks through TASKS.d/NAME/ on the shared file
# system, so any number of them (one per GPU in an allocation, or one per array task) can
# share a list. A claim held by a SLURM job that is no longer running is taken over, so a
# task cut off by the time limit is picked up by the next allocation; the ONIOM run scripts
# resume from their checkpoints. A task that fails MAX_TRIES times (default 3) is skipped.
# MAX: stop after this many tasks (default: run until none are left).
set -uo pipefail
tasks=$(realpath "$1")
max=${2:-0}
state=$tasks.d
here=$(dirname "$tasks")
me="${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-0} $(hostname) $$"
mkdir -p "$state"
log() { echo "$(date -Is) [$me] $*"; }

alive() {  # alive CLAIM_DIR: is the owner of the claim still running?
    local jobid host pid
    read -r jobid host pid < "$1/owner" 2>/dev/null || return 1
    if [[ $jobid != local.* ]] && command -v squeue > /dev/null; then
        squeue -h -j "${jobid%%.*}" -t R,CG 2> /dev/null | grep -q .
    else
        [[ $host == "$(hostname)" ]] && kill -0 "$pid" 2> /dev/null
    fi
}

claim() {  # claim NAME: take the task, or fail if someone else runs it
    local d=$state/$1
    mkdir -p "$d"
    if ! mkdir "$d/claim" 2> /dev/null; then
        alive "$d/claim" && return 1
        mv "$d/claim" "$d/stale.$$" 2> /dev/null || return 1  # only one worker wins the rename
        rm -rf "$d/stale.$$"
        mkdir "$d/claim" 2> /dev/null || return 1
    fi
    echo "$me" > "$d/claim/owner"
}

done_count=0
while true; do
    ran=0
    while read -r name cmd; do
        [[ -z $name || $name == \#* ]] && continue
        d=$state/$name
        [[ -f $d/done ]] && continue
        (( $(cat "$d/tries" 2> /dev/null || echo 0) >= ${MAX_TRIES:-3} )) && continue
        claim "$name" || continue
        echo $(( $(cat "$d/tries" 2> /dev/null || echo 0) + 1 )) > "$d/tries"
        log "start $name: $cmd"
        if (cd "$here" && bash -c "$cmd") >> "$d/log" 2>&1; then
            touch "$d/done"
            log "done $name"
        else
            log "FAILED $name (see $d/log)"
        fi
        rm -rf "$d/claim"
        ran=1
        done_count=$((done_count + 1))
        (( max > 0 && done_count >= max )) && exit 0
        break  # re-read the list: another worker may have finished tasks meanwhile
    done < <(grep -v '^[[:space:]]*#' "$tasks")
    (( ran == 0 )) && { log "no tasks left for me"; exit 0; }
done
