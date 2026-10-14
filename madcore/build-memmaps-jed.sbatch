#!/usr/bin/env bash
# MAD-CORE memmaps (metatrain MemmapDataset format) for every coreset size
# 2^N (first 2^N rows in FPS order, split seed 0), built on jed (CPU).
# Outputs go to /work (home is near its 100 GiB quota) and are symlinked
# as ~/data/madcore/memmap-2powN. Submit from lyra: ssh jed sbatch <this>.
#SBATCH --job-name=madcore-memmaps
#SBATCH --account=cosmo
#SBATCH --partition=standard
#SBATCH --qos=serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

SITE=/home/boittier/metawork/.venv-blackwell/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SITE/metatensor/lib:$SITE/metatomic/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CONV=/home/boittier/metawork/metatomic-core-examples/build/madcore-to-diskdataset
SPLIT=/home/boittier/metawork/madcore/splits/madcore-split-seed0.npz
SRC=/home/boittier/data/madcore
DST=/work/cosmo/boittier/lyra/madcore-data
mkdir -p $DST
LEVELS=${LEVELS:-"15 17 19 21 22 23 24"}

echo "host $(hostname); levels $LEVELS"
build() {
    local n=$1 out=$DST/memmap-2pow$1
    if [ -d $out ]; then echo "2^$n: exists, skipped"; return; fi
    rm -rf $out.tmp
    local t=$(date +%s)
    # shards in FPS order (glob sorts 0-2pow18 < 2pow18-2pow19 < ... < 2pow23-2pow24)
    if $CONV --split $SPLIT -o $out.tmp --max-rows $((1 << n)) $SRC/madcore_*.extxyz.gz > $DST/memmap-2pow$n.log 2>&1; then
        mv $out.tmp $out
        ln -sfn $out $SRC/memmap-2pow$n
        echo "2^$n: done in $(( $(date +%s) - t )) s, $(du -sh $out | cut -f1)"
    else
        rm -rf $out.tmp # never leave partial output on the shared fileset
        echo "2^$n: FAILED (see $DST/memmap-2pow$n.log)"; tail -5 $DST/memmap-2pow$n.log
    fi
}
for n in $LEVELS; do build $n & done
wait
ls -la $SRC | grep memmap
