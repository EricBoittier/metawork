#!/bin/bash
source $HOME/petlr/work/tiled_vs_mixed/env.sh
C="--cache $SCRATCH/tvm_pool.pkl --config mad --steps 8 --warmup 3 --compilation-cache $SCRATCH/tvm_jax_cache"

run () { echo "### $*"; python bench.py trial $C "$@" 2>&1 | grep -E "^RESULT|Error|error" | head -3; }

echo "===== A: LR share (MAD, tiled) ====="
run --backend tiled --batch-size 256
run --backend tiled --batch-size 256 --sr-only
run --backend tiled --batch-size 1024
run --backend tiled --batch-size 1024 --sr-only

echo "===== B: LR share (MAD, mixed) ====="
run --backend mixed --batch-size 256
run --backend mixed --batch-size 256 --sr-only

echo "===== C: BM tuning (MAD, tiled, S=256) ====="
run --backend tiled --batch-size 256 --bm 16
run --backend tiled --batch-size 256 --bm 32

echo "===== D: reproducibility (repeat of the headline point) ====="
run --backend tiled --batch-size 1024
