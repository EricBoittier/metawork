#!/usr/bin/env bash
# Pair lists for GMX_METATOMIC_DISTRIBUTED: the built-in cell list against vesin, rebuilt every step
# or every 10 steps with a 0.1 nm buffer. Pure-ML water, PET-MAD xs, 2 H100s. PHASE:
#   speed  build capi-halo-split with vesin; 98k-atom box, 100 MD steps per variant: ns/day and
#          the per-phase timers (GMX_METATOMIC_TIMER)
#   equil  4 nm box (about 6.4k atoms): 2 ps NVT at 300 K on 1 GPU, then 2 ps NVE on 1 GPU (the
#          reference, GROMACS's own pair list)
#   nve    2 ps NVE from the equilibrated state for each 2-GPU variant; drift against the reference
#   single the regular (non-distributed) path on 1 GPU with threaded pair packing, against the
#          build without it: 98k frame compared, 98k MD, hemoglobin s1 (4 heme sites) PET-MAD xs and
#          PET-OMol s for 2000 steps
#   seeds  buffer repeats: 3 seeds of fresh 300 K velocities on the equilibrated box, 4 ps NVE each
#          for the 1-GPU reference and vesin every 10 steps with 0.05/0.1/0.2 nm buffers (2 ranks
#          on one GPU); four runs at a time, one per GPU. Submit with -q normal --gpus=4 -c 64
#          --mem=360G -t 02:00:00
#   speed2 more 98k variants: vesin every 5, 20, 40 steps, cells every 5, vesin every 10 with 0.2 nm
#   nve2   drift matrix, two 2-rank runs at a time (each on one GPU and 16 cores): vesin every
#          10/20/40 steps with 0.05/0.1/0.2 nm buffers, and cells every 10
# Free debug QOS.
#SBATCH --job-name=mlwater-nl
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/mlwater-nl-%j.out
set -uo pipefail
M=/home/boittier/metawork
W=$M/gromacs-mlwater-bench
V=$M/builds/torch-cu126/lib/python3.12/site-packages
PHASE=${PHASE:?speed, equil or nve}
echo "== $SLURM_JOB_ID $PHASE $(hostname) $(date -Is) halo $(git -C $M/gromacs-capi-halo rev-parse --short HEAD)"; nvidia-smi -L
PREFIX=gromacs-capi-halo; [[ $PHASE == speed3 || $PHASE == single ]] && PREFIX=gromacs-capi-pack   # not under running jobs
if [[ $PHASE == speed || $PHASE == speed3 || $PHASE == single ]]; then
    module purge
    module load gcc/13.2.0 cuda/12.4.1
    export PATH=$M/builds/tools/bin:$PATH CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    export LD_LIBRARY_PATH=$V/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
    ONIOM_PY=$M/builds/torch-cu126/bin/python JOBS=32 GMX_SRC=$M/gromacs-capi-halo GMX=$M/builds/$PREFIX-kuma-x86_64/bin/gmx \
        GMX_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$M/builds/capi/core-prefix;$V/metatensor/lib/cmake -DGMX_METATOMIC_VESIN_DIR=$V/vesin -DPython_ROOT_DIR=/home/boittier/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu -DPython_FIND_VIRTUALENV=STANDARD -DCUDA_cudart_LIBRARY=$V/nvidia/cuda_runtime/lib/libcudart.so.12" \
        $M/etc/oniom/build-gromacs.sh > $M/builds/logs/mlwater-nl-build-$SLURM_JOB_ID.out 2>&1 || { echo "build failed"; tail -30 $M/builds/logs/mlwater-nl-build-$SLURM_JOB_ID.out; exit 1; }
    grep -h 'Found vesin' $M/builds/logs/mlwater-nl-build-$SLURM_JOB_ID.out || { echo "vesin not found"; exit 1; }
    echo "== built $(date -Is)"
fi
. $M/builds/kuma-env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GMX=$M/builds/$PREFIX-kuma-x86_64/bin/gmx ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
. $M/etc/oniom/env.sh
R=$W/runs/nl
mkdir -p $R && cd $R

box() {  # box NAME L: water box, topology and the bench mdp
    $GMX solvate -cs spc216.gro -box $2 $2 $2 -o $1.gro -quiet > /dev/null 2>&1
    printf '#include "%s/mlwater.itp"\n\n[ system ]\nPET-MAD xs water\n\n[ molecules ]\nSOL %d\n' $W $(( $(sed -n 2p $1.gro) / 3 )) > $1.top
}
run() {  # run TAG TPR NTMPI [VAR=VALUE...] -- [mdrun args]
    local tag=$1 tpr=$2 n=$3; shift 3
    local envs=(); while [[ $# -gt 0 && $1 != -- ]]; do envs+=("$1"); shift; done; [[ ${1:-} == -- ]] && shift
    local gpus=${RUN_GPUS:-01}; [[ $n == 1 && -z ${RUN_GPUS:-} ]] && gpus=0
    local gpuargs="-gpu_id $gpus"; [[ -n ${RUN_GPUTASKS:-} ]] && gpuargs="-gputasks $RUN_GPUTASKS"   # not both
    mkdir -p $tag && cd $tag
    local smi=()
    for g in 0 1; do nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 > gpu$g.mem & smi+=($!); done
    env GMX_METATOMIC_DEVICE="${RUN_DEVICE:-cuda:{rank\}}" "${envs[@]}" $GMX mdrun -s ../$tpr -deffnm run -ntmpi $n -ntomp 8 -npme 0 -nb gpu \
        -pme cpu -bonded cpu -update cpu $gpuargs -dlb no -pin on -pinoffset ${RUN_PINOFF:-0} -pinstride 1 "$@" > run.out 2>&1 && st=ok || st=FAILED
    kill "${smi[@]}"; wait "${smi[@]}" 2>/dev/null
    local phases=""
    if [[ -f metatomic_timer_rank_0.log ]]; then
        phases=$(awk -F': ' '/ ms$/{v=$2; sub(/ ms/,"",v); gsub(/^ +/,"",$1); s[$1]+=v; n[$1]++} END{for(k in s) if (k ~ /^(dist|execute_model)/) printf "%s %.1f  ", k, s[k]/n[k]}' metatomic_timer_rank_0.log)
    fi
    [[ -f run.out ]] && phases="$phases $(grep -h '^\[metatomic-torch\]' run.out | tail -3 | sed 's/\[metatomic-torch\] *//' | tr -s ' ' | tr '\n' ';')"
    printf '%-22s %-6s %s  GPU %s/%s MiB  %s\n' $tag $st "$(grep -h '^Performance:' run.log 2>/dev/null | awk '{print $2" ns/day, "$4" ms/step"}')" \
        "$(sort -n gpu0.mem | tail -1)" "$(sort -n gpu1.mem | tail -1)" "$phases"
    cd ..
}
VARIANTS=("cells-1 GMX_METATOMIC_DIST_NL=cells GMX_METATOMIC_DIST_NSTLIST=1"
          "vesin-1 GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=1"
          "cells-10 GMX_METATOMIC_DIST_NL=cells GMX_METATOMIC_DIST_NSTLIST=10 GMX_METATOMIC_DIST_BUFFER=0.1"
          "vesin-10 GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=10 GMX_METATOMIC_DIST_BUFFER=0.1")

drift() {  # drift of every NVE run in this directory, against nve-1gpu
    $ONIOM_PY -I - "$GMX" <<'EOF'
import glob, subprocess, sys
import numpy as np
gmx = sys.argv[1]
def series(d):
    subprocess.run([gmx, "energy", "-f", f"{d}/run.edr", "-o", f"{d}/e.xvg", "-quiet"], input="Total-Energy\nTemperature\n0\n",
                   text=True, capture_output=True, check=True)
    L = open(f"{d}/e.xvg").read().splitlines()
    a = np.loadtxt([l for l in L if l and l[0] not in "#@"], ndmin=2)
    return a[:, 0], a[:, 1], a[:, 2]
ref = None
for d in ["nve-1gpu"] + sorted(glob.glob("nve-*-*"), key=lambda d: (d.split("-")[1], [int(x) if x.isdigit() else x for x in d.split("-")[2:]])):
    try:
        t, E, T = series(d)
    except (subprocess.CalledProcessError, OSError, ValueError):
        print(f"   {d:12s} no energies (run failed)")
        continue
    if ref is None:
        ref = E
    n = min(len(E), len(ref))
    print(f"   {d:12s} drift {np.polyfit(t, E, 1)[0]:+8.3f} kJ/mol/ps  std {E.std():6.2f}  T {T.mean():6.1f} K  "
          f"|E - E_1gpu| max over the first 200 steps {np.abs(E[:min(n, 200)] - ref[:min(n, 200)]).max():.2f} kJ/mol")
EOF
}

case $PHASE in
speed)
    box w98k 10.0
    oniom_mdp $W/bench.mdp w98k.mdp
    $GMX grompp -f w98k.mdp -c w98k.gro -p w98k.top -o w98k.tpr -maxwarn 3 -quiet > w98k.grompp.log 2>&1 || { tail -12 w98k.grompp.log; exit 1; }
    for v in "${VARIANTS[@]}"; do
        set -- $v
        run speed-$1 w98k.tpr 2 GMX_METATOMIC_DISTRIBUTED=1 GMX_METATOMIC_TIMER=1 "${@:2}"
    done ;;
equil)
    box w4 4.0
    oniom_mdp $W/bench.mdp w4.mdp
    sed -e 's/^nsteps .*/nsteps                  = 4000/' -e 's/^nstlog .*/nstlog                  = 500/' \
        -e 's/^tcoupl .*/tcoupl                  = v-rescale/' w4.mdp > w4-nvt.mdp
    printf 'tc-grps                 = System\ntau-t                   = 0.1\nref-t                   = 300\n' >> w4-nvt.mdp
    $GMX grompp -f w4-nvt.mdp -c w4.gro -p w4.top -o w4-nvt.tpr -maxwarn 3 -quiet > w4-nvt.grompp.log 2>&1 || { tail -12 w4-nvt.grompp.log; exit 1; }
    run equil-nvt w4-nvt.tpr 1
    sed -e 's/^nsteps .*/nsteps                  = 4000/' -e 's/^nstlog .*/nstlog                  = 500/' \
        -e 's/^gen-vel .*/gen-vel                 = no/' w4.mdp > w4-nve.mdp
    echo 'continuation            = yes' >> w4-nve.mdp
    $GMX grompp -f w4-nve.mdp -c equil-nvt/run.gro -t equil-nvt/run.cpt -p w4.top -o w4-nve.tpr -maxwarn 3 -quiet > w4-nve.grompp.log 2>&1 || { tail -12 w4-nve.grompp.log; exit 1; }
    run nve-1gpu w4-nve.tpr 1 ;;
nve)
    for v in "cells-1 GMX_METATOMIC_DIST_NL=cells GMX_METATOMIC_DIST_NSTLIST=1" "${VARIANTS[1]}" "${VARIANTS[3]}"; do
        set -- $v
        run nve-$1 w4-nve.tpr 2 GMX_METATOMIC_DISTRIBUTED=1 "${@:2}"
    done
    drift
    ;;
single)
    OLD=$M/builds/gromacs-capi-halo-kuma-x86_64/bin/gmx
    printf 'nstfout                 = 1\n' | cat w98k.mdp - | sed 's/^nsteps .*/nsteps                  = 0/' > w98k.rerun.mdp
    $GMX grompp -f w98k.rerun.mdp -c w98k.gro -p w98k.top -o w98k.rerun.tpr -maxwarn 3 -quiet > /dev/null 2>&1
    GMXSAVE=$GMX; GMX=$OLD run single-old-r1 w98k.rerun.tpr 1 -- -rerun ../w98k.gro; GMX=$GMXSAVE
    run single-new-r1 w98k.rerun.tpr 1 -- -rerun ../w98k.gro
    for t in old new; do ln -sf single-$t-r1/run.edr single-$t-r1.edr; ln -sf single-$t-r1/run.trr single-$t-r1.trr; done
    ln -sf w98k.rerun.tpr single-old.tpr
    $ONIOM_PY -I $M/builds/halo_compare.py $GMX single-old-r1 single-new-r1 2>&1 | grep -v virial | sed 's/^/   /'
    run single-new-md1 w98k.tpr 1
    HB=$M/gromacs-oniom-hemoglobin/runs/s1-fix-capi
    for stage in onve omve; do
        ln -sf $HB/$stage.tpr hb-$stage.tpr
        run single-new-hb-$stage hb-$stage.tpr 1 -- -nsteps 2000
    done ;;
seeds)
    runs=()
    for seed in 1 2 3; do
        sed -e 's/^nsteps .*/nsteps                  = 8000/' -e 's/^nstlog .*/nstlog                  = 1000/' \
            -e "s/^gen-seed .*/gen-seed                = $seed/" w4.mdp > w4-seed$seed.mdp
        $GMX grompp -f w4-seed$seed.mdp -c equil-nvt/run.gro -p w4.top -o w4-seed$seed.tpr -maxwarn 3 -quiet > /dev/null 2>&1 \
            || { echo "grompp seed $seed failed"; exit 1; }
        runs+=("s$seed-1gpu w4-seed$seed.tpr 1")
        for b in 0.05 0.1 0.2; do
            runs+=("s$seed-vesin-10-b$b w4-seed$seed.tpr 2 GMX_METATOMIC_DISTRIBUTED=1 GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=10 GMX_METATOMIC_DIST_BUFFER=$b")
        done
    done
    for ((k = 0; k < ${#runs[@]}; k += 4)); do
        for g in 0 1 2 3; do
            [[ $((k + g)) -lt ${#runs[@]} ]] || continue
            set -- ${runs[$((k + g))]}
            if [[ $3 == 1 ]]; then
                RUN_GPUS=$g RUN_DEVICE=cuda:$g RUN_PINOFF=$((16 * g)) run seeds-$1 $2 1 &
            else
                RUN_GPUS=$g RUN_GPUTASKS=$g$g RUN_DEVICE=cuda:$g RUN_PINOFF=$((16 * g)) run seeds-$1 $2 2 "${@:4}" &
            fi
        done
        wait
    done
    $ONIOM_PY -I - "$GMX" <<'EOF2'
import subprocess, sys
import numpy as np
gmx = sys.argv[1]
def series(d):
    subprocess.run([gmx, "energy", "-f", f"{d}/run.edr", "-o", f"{d}/e.xvg", "-quiet"], input="Total-Energy\n0\n",
                   text=True, capture_output=True, check=True)
    a = np.loadtxt([l for l in open(f"{d}/e.xvg") if l[0] not in "#@"], ndmin=2)
    return a[:, 0], a[:, 1]
rows = {}
for variant in ("1gpu", "vesin-10-b0.05", "vesin-10-b0.1", "vesin-10-b0.2"):
    for seed in (1, 2, 3):
        try:
            t, E = series(f"seeds-s{seed}-{variant}")
        except Exception:
            print(f"   s{seed} {variant}: no energies"); continue
        rows.setdefault(variant, []).append(np.polyfit(t, E, 1)[0])
        print(f"   s{seed} {variant:16s} drift {rows[variant][-1]:+8.3f} kJ/mol/ps over {t[-1] - t[0]:.1f} ps")
for variant, d in rows.items():
    print(f"   {variant:16s} mean {np.mean(d):+7.2f}  sd {np.std(d, ddof=1) if len(d) > 1 else float('nan'):5.2f}  (n={len(d)})")
EOF2
    ;;
speed3)
    for v in "${VARIANTS[2]}" "${VARIANTS[3]}" "${VARIANTS[0]}"; do
        set -- $v
        run speed3-$1 w98k.tpr 2 GMX_METATOMIC_DISTRIBUTED=1 GMX_METATOMIC_TIMER=1 "${@:2}"
    done
    set -- ${VARIANTS[2]}
    run speed3-$1-plugin w98k.tpr 2 GMX_METATOMIC_DISTRIBUTED=1 GMX_METATOMIC_TIMER=1 METATOMIC_TORCH_TIMER=1 "${@:2}" ;;
speed2)
    for v in "vesin-5 GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=5 GMX_METATOMIC_DIST_BUFFER=0.1" \
             "cells-5 GMX_METATOMIC_DIST_NL=cells GMX_METATOMIC_DIST_NSTLIST=5 GMX_METATOMIC_DIST_BUFFER=0.1" \
             "vesin-20 GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=20 GMX_METATOMIC_DIST_BUFFER=0.1" \
             "vesin-40 GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=40 GMX_METATOMIC_DIST_BUFFER=0.1" \
             "vesin-10-b0.2 GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=10 GMX_METATOMIC_DIST_BUFFER=0.2"; do
        set -- $v
        run speed-$1 w98k.tpr 2 GMX_METATOMIC_DISTRIBUTED=1 GMX_METATOMIC_TIMER=1 "${@:2}"
    done ;;
nve2)
    combos=()
    for nst in 10 20 40; do for b in 0.05 0.1 0.2; do
        [[ $nst == 10 && $b == 0.1 ]] && continue   # done in phase nve
        combos+=("vesin-$nst-b$b GMX_METATOMIC_DIST_NL=vesin GMX_METATOMIC_DIST_NSTLIST=$nst GMX_METATOMIC_DIST_BUFFER=$b")
    done; done
    combos+=("cells-10-b0.1 GMX_METATOMIC_DIST_NL=cells GMX_METATOMIC_DIST_NSTLIST=10 GMX_METATOMIC_DIST_BUFFER=0.1")
    for ((k = 0; k < ${#combos[@]}; k += 2)); do
        for g in 0 1; do
            [[ $((k + g)) -lt ${#combos[@]} ]] || continue
            set -- ${combos[$((k + g))]}
            RUN_GPUS=$g RUN_GPUTASKS=$g$g RUN_DEVICE=cuda:$g RUN_PINOFF=$((16 * g)) run nve-$1 w4-nve.tpr 2 GMX_METATOMIC_DISTRIBUTED=1 "${@:2}" &
        done
        wait
    done
    drift ;;
esac
echo "== done $(date -Is)"
