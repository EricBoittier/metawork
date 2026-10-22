#!/usr/bin/env bash
# Halo split across two GPUs (local helper, not tracked): build capi-halo-split, then the whole
# solute (9,026 atoms) of s1 as ML, one frame (npt.gro) evaluated with -rerun on 1 rank, on 2
# ranks as before (every rank runs the whole ML region), and on 2 ranks with
# GMX_METATOMIC_HALO_SPLIT=1; energies, all forces, and each GPU's peak memory compared.
# Then 40 MD steps of each for ns/day. Free debug QOS, 2 GPUs.
#SBATCH --job-name=halo-test
#SBATCH --partition=h100
#SBATCH --qos=debug
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/halo-test-%j.out
set -uo pipefail
M=/home/boittier/metawork
HB=$M/gromacs-oniom-hemoglobin
V=$M/builds/torch-cu126/lib/python3.12/site-packages
module purge
module load gcc/13.2.0 cuda/12.4.1
export PATH=$M/builds/tools/bin:$PATH CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export LD_LIBRARY_PATH=$V/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
export ONIOM_PY=$M/builds/torch-cu126/bin/python JOBS=32
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) halo $(git -C $M/gromacs-capi-halo rev-parse --short HEAD)"; nvidia-smi -L
GMX_SRC=$M/gromacs-capi-halo GMX=$M/builds/gromacs-capi-halo-kuma-x86_64/bin/gmx \
    GMX_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$M/builds/capi/core-prefix;$V/metatensor/lib/cmake -DPython_ROOT_DIR=/home/boittier/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu -DPython_FIND_VIRTUALENV=STANDARD -DCUDA_cudart_LIBRARY=$V/nvidia/cuda_runtime/lib/libcudart.so.12" \
    $M/etc/oniom/build-gromacs.sh > $M/builds/logs/halo-build-$SLURM_JOB_ID.out 2>&1 || { echo "build failed"; tail -30 $M/builds/logs/halo-build-$SLURM_JOB_ID.out; exit 1; }
echo "== built $(date -Is)"
. $M/builds/kuma-env.sh
export GMX=$M/builds/gromacs-capi-halo-kuma-x86_64/bin/gmx ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
. $M/etc/oniom/env.sh
R=$HB/runs/halotest
mkdir -p $R
for f in $HB/runs/s1/*; do n=$(basename $f); [[ $n == o[mn]v[te].* || $n == allml-* || -e $R/$n ]] || cp -a $f $R/; done
cd $R
for model in ${MODELS:-pet-mad-xs-v1.5.0 pet-mad-m-v1.6.0}; do
    echo "== $model"
    # allml.sh's whole-solute mdp, with forces written every step for the rerun
    sed -e "s|pet-[a-z]*-[a-z]*-v[0-9.]*\.pt|$model.pt|" -e "s|^metatomic-input-group .*|metatomic-input-group   = SOLUTE|" \
        -e "/^metatomic-link-atoms/d" -e "/^metatomic-site-/d" $HB/mdp/onvt.mdp > $model.raw.mdp
    oniom_mdp $model.raw.mdp $model.mdp
    sed -i -e '/^nstfout/d' -e 's/^nstcalcenergy .*/nstcalcenergy           = 1/' -e 's/^nstenergy .*/nstenergy               = 1/' $model.mdp
    echo 'nstfout                 = 1' >> $model.mdp
    $GMX grompp -f $model.mdp -c npt.gro -t npt.cpt -p topol.top -n index.ndx -o $model.tpr -po $model.out.mdp -quiet > $model.grompp.log 2>&1 || { tail -5 $model.grompp.log; continue; }
    run() {  # run TAG NTMPI HALO MODE(rerun|md)
        local smi=()
        for g in 0 1; do nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 > $1.gpu$g.mem & smi+=($!); done
        local extra=(-rerun npt.gro); [[ $4 == md ]] && extra=(-nsteps 40)
        GMX_METATOMIC_HALO_SPLIT=$3 GMX_METATOMIC_DEVICE='cuda:{rank}' $GMX mdrun -s $model.tpr -deffnm $1 \
            -ntmpi $2 -ntomp 8 -npme 0 -nb gpu -pme cpu -bonded cpu -update cpu -gpu_id 01 -pin on "${extra[@]}" > $1.out 2>&1 \
            && st=ok || st=FAILED
        kill "${smi[@]}"; wait "${smi[@]}" 2>/dev/null
        printf '%-28s %-6s peak GPU0 %6s MiB  GPU1 %6s MiB  %s\n' $1 $st "$(sort -n $1.gpu0.mem | tail -1)" "$(sort -n $1.gpu1.mem | tail -1)" \
            "$(grep -h 'halo split, rank\|Performance:' $1.log 2>/dev/null | tr '\n' ' ')"
        [[ $st == ok ]] || tail -15 $1.out
    }
    run $model-r1 1 0 rerun
    run $model-r2 2 0 rerun
    run $model-r2halo 2 1 rerun
    $ONIOM_PY -I $M/builds/halo_compare.py $GMX $model-r1 $model-r2 $model-r2halo
    run $model-md1 1 0 md
    run $model-md2 2 0 md
    run $model-md2halo 2 1 md
done
echo "== done $(date -Is)"
