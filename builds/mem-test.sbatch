#!/usr/bin/env bash
# Does the whole solute (protein + hemes, 9,026 atoms) fit on the card as ML, on each build?
# (local helper, not tracked). allml.sh with every mdrun capped at CAP steps, for every model and
# both GROMACS builds (libtorch: oniom-stress-fixes; C API: capi-binding + torch plugin), with
# nvidia-smi sampling the card's used memory. Free debug QOS; submit with -p h100 or -p l40s.
#SBATCH --job-name=hb-memtest
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/metawork/builds/logs/memtest-%j.out
set -uo pipefail
M=/home/boittier/metawork
HB=$M/gromacs-oniom-hemoglobin
. $M/builds/kuma-env.sh
export CAP=${CAP:-30}
OUT=$HB/runs/memtest/$SLURM_JOB_PARTITION${ALLML_GROUP:+-$ALLML_GROUP}
D=allml${ALLML_GROUP:+-$ALLML_GROUP}
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "== $SLURM_JOB_ID $(hostname) $(date -Is) $GPU CAP=$CAP"
declare -A REAL=([torch]=$M/builds/gromacs-oniom-kuma-x86_64/bin/gmx [capi]=$M/builds/gromacs-capi-kuma-x86_64/bin/gmx)
MODELS=${MODELS:-"pet-mad-xs-v1.5.0 pet-mad-s-v1.5.0 pet-omol-s-v1.0.0 pet-omol-m-v1.0.0 pet-mad-m-v1.6.0 pet-omol-l-v1.0.0"}
mkdir -p $OUT
printf "%-6s %-20s %-8s %10s %10s\n" build model status peak_MiB ms/step | tee $OUT.tsv
for build in capi torch; do
    R=$OUT/$build
    mkdir -p $R
    for f in $HB/runs/s1/*; do   # MM stages and index groups of replica s1
        n=$(basename $f)
        [[ $n == o[mn]v[te].* || $n == allml-* || -e $R/$n ]] || cp -a $f $R/
    done
    for model in $MODELS; do
        rm -rf $R/$D-$model
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 > $R/$model.mem &
        smi=$!
        ( export GMX=$M/builds/gmx-capped GMX_REAL=${REAL[$build]}
          [[ $build == capi ]] && export ONIOM_EXTENSIONS=$M/builds/capi/build-plugin/libmetatomic_torch_plugin.so
          $HB/allml.sh $R $model ) > $R/$model.log 2>&1 && st=ok || st=FAILED
        kill $smi; wait $smi 2>/dev/null
        grep -qiE 'out of memory|CUDA error: out' $R/$model.log $R/$D-$model/*.log 2>/dev/null && st=OOM
        peak=$(sort -n $R/$model.mem | tail -1)
        ms=$(awk '/^Performance:/{print 86400e3/($2*1e6/0.5)}' $R/$D-$model/nvt.log 2>/dev/null | tail -1)
        printf '%-6s %-20s %-8s %10s %10s\n' $build $model $st "$peak" "${ms:--}" | tee -a $OUT.tsv
    done
done
echo "== done $(date -Is)"
