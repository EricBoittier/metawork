#!/usr/bin/env bash
# MODE=split (default): the halo split as below; MODE=dist: GMX_METATOMIC_DISTRIBUTED on 2 GPUs
# against 1 GPU (100k: one frame compared, then MD; 150k: MD on 2 GPUs).
# Pure-ML water with PET-MAD xs on one and two H100s (GROMACS capi-halo-split, C API + torch
# plugin). Boxes of BOX_100K and BOX_150K nm (about 100k and 150k atoms):
#   100k: one frame (-rerun) on 1 GPU and on 2 GPUs with GMX_METATOMIC_HALO_SPLIT=1, energies
#         and all forces compared; then MD on 1 and 2 GPUs for ns/day
#   150k: MD on 1 GPU, on 2 GPUs as before, and on 2 GPUs with the split; the first two are
#         expected to run out of memory
# Each GPU's peak memory from nvidia-smi. Free debug QOS, 2 GPUs.
#SBATCH --job-name=mlwater-bench
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/mlwater-bench-%j.out
set -uo pipefail
M=/home/boittier/metawork
W=$M/gromacs-mlwater-bench
V=$M/builds/torch-cu126/lib/python3.12/site-packages
module purge
module load gcc/13.2.0 cuda/12.4.1
export PATH=$M/builds/tools/bin:$PATH CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export LD_LIBRARY_PATH=$V/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
export ONIOM_PY=$M/builds/torch-cu126/bin/python JOBS=32
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) halo $(git -C $M/gromacs-capi-halo rev-parse --short HEAD)"; nvidia-smi -L
GMX_SRC=$M/gromacs-capi-halo GMX=$M/builds/gromacs-capi-halo-kuma-x86_64/bin/gmx \
    GMX_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$M/builds/capi/core-prefix;$V/metatensor/lib/cmake -DPython_ROOT_DIR=/home/boittier/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu -DPython_FIND_VIRTUALENV=STANDARD -DCUDA_cudart_LIBRARY=$V/nvidia/cuda_runtime/lib/libcudart.so.12" \
    $M/etc/oniom/build-gromacs.sh > $M/builds/logs/mlwater-build-$SLURM_JOB_ID.out 2>&1 || { echo "build failed"; tail -30 $M/builds/logs/mlwater-build-$SLURM_JOB_ID.out; exit 1; }
echo "== built $(date -Is)"
. $M/builds/kuma-env.sh
# the number of pairs changes every step; without expandable segments the caching allocator
# fragments and an MD run needs far more memory than one evaluation
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export GMX=$M/builds/gromacs-capi-halo-kuma-x86_64/bin/gmx ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
. $M/etc/oniom/env.sh
R=$W/runs/$SLURM_JOB_ID
mkdir -p $R && cd $R

run() {  # run TAG TPR NTMPI MODE [mdrun args]; MODE 0 (as before), 1 (halo split), dist (distributed)
    local tag=$1 tpr=$2 n=$3 mode=$4; shift 4
    local halo=0 dist=0; [[ $mode == 1 ]] && halo=1; [[ $mode == dist ]] && dist=1
    local smi=()
    for g in 0 1; do nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 > $tag.gpu$g.mem & smi+=($!); done
    local gpus=01; [[ $n == 1 ]] && gpus=0
    GMX_METATOMIC_HALO_SPLIT=$halo GMX_METATOMIC_DISTRIBUTED=$dist GMX_METATOMIC_DEVICE='cuda:{rank}' $GMX mdrun -s $tpr -deffnm $tag \
        -ntmpi $n -ntomp 8 -npme 0 -nb gpu -pme cpu -bonded cpu -update cpu -gpu_id $gpus -dlb no -pin on "$@" \
        > $tag.out 2>&1 && st=ok || st=FAILED
    grep -qiE 'out of memory' $tag.out $tag.log 2>/dev/null && st=OOM
    kill "${smi[@]}"; wait "${smi[@]}" 2>/dev/null
    printf '%-18s %-6s peak GPU0 %6s MiB  GPU1 %6s MiB  %s %s\n' $tag $st \
        "$(sort -n $tag.gpu0.mem | tail -1)" "$(sort -n $tag.gpu1.mem | tail -1)" \
        "$(grep -h '^Performance:' $tag.log 2>/dev/null | awk '{print $2" ns/day"}')" \
        "$(grep -h 'halo split, first step\|distributed, first step' $tag.log 2>/dev/null | sed 's/.*rows: //')"
}

for size in ${SIZES:-100k 150k}; do
    L=$([[ $size == 100k ]] && echo ${BOX_100K:-10.0} || echo ${BOX_150K:-11.45})
    echo "== $size: box $L nm"
    $GMX solvate -cs spc216.gro -box $L $L $L -o $size.gro -quiet > /dev/null 2>&1
    nw=$(( ($(sed -n 2p $size.gro) ) / 3 ))
    printf '#include "%s/mlwater.itp"\n\n[ system ]\nPET-MAD xs water %s\n\n[ molecules ]\nSOL %d\n' $W $size $nw > $size.top
    oniom_mdp $W/bench.mdp $size.mdp
    sed -e 's/^nsteps .*/nsteps                  = 0/' $size.mdp > $size.rerun.mdp; echo 'nstfout                 = 1' >> $size.rerun.mdp
    for t in "" .rerun; do
        $GMX grompp -f $size$t.mdp -c $size.gro -p $size.top -o $size$t.tpr -po $size$t.out.mdp -maxwarn 3 -quiet > $size$t.grompp.log 2>&1 \
            || { echo "grompp $size$t failed"; tail -12 $size$t.grompp.log; continue 2; }
    done
    echo "   $(sed -n 2p $size.gro) atoms ($nw waters)"
    case ${MODE:-split} in
    split)
        if [[ $size == 100k ]]; then
            run ${size}-r1 $size.rerun.tpr 1 0 -rerun $size.gro
            run ${size}-r2halo $size.rerun.tpr 2 1 -rerun $size.gro
            $ONIOM_PY -I $M/builds/halo_compare.py $GMX ${size}-r1 ${size}-r2halo 2>&1 | sed 's/^/   /'
            run ${size}-md1 $size.tpr 1 0
            run ${size}-md2 $size.tpr 2 0
            run ${size}-md2halo $size.tpr 2 1
        else
            run ${size}-md1 $size.tpr 1 0
            run ${size}-md2 $size.tpr 2 0
            run ${size}-md2halo $size.tpr 2 1
        fi ;;
    dist)
        if [[ $size == 100k ]]; then
            run ${size}-r1 $size.rerun.tpr 1 0 -rerun $size.gro
            run ${size}-r2dist $size.rerun.tpr 2 dist -rerun $size.gro
            $ONIOM_PY -I $M/builds/halo_compare.py $GMX ${size}-r1 ${size}-r2dist 2>&1 | sed 's/^/   /'
            run ${size}-md1 $size.tpr 1 0
        fi
        run ${size}-md2dist $size.tpr 2 dist ;;
    esac
done
echo "== done $(date -Is)"
