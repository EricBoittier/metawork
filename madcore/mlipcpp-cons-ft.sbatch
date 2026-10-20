#!/usr/bin/env bash
# Conservative fine-tune of an mlip.cpp-trained PET in metatrain
# (options-pet-big6-adapt16-cons-ft.yaml).
#   STAGE=smoke (default; 1 GPU, debug): convert the GGUF (gguf_to_checkpoint.py),
#     round-trip export -> the same tensors, one epoch at lr 1e-12 on DATA (the
#     validation errors are the converted model's: compare with mlipcpp-md check
#     on the same split), then one real epoch for throughput and memory.
#   STAGE=train (sbatch --qos=normal --gpus-per-node=2 --ntasks=2 --ntasks-per-node=2
#     --time=...): the fine-tune proper, then export to GGUF (--nc-forces).
#SBATCH --job-name=cons-ft
#SBATCH --account=cosmo
#SBATCH --partition=rtx6000
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/boittier/data/madcore/sweep/logs/%x-%j.out
set -euo pipefail

STAGE=${STAGE:-smoke}
VPY=/home/boittier/metawork/.venv-blackwell/bin/python
MTT=$(dirname $VPY)/mtt
MTT_SRC=/home/boittier/metawork/metatrain/main-baseline/src
MLIPCPP=/home/boittier/metawork/mlip.cpp
PROJ=/work/cosmo/boittier/lyra/mlipcpp
RUN=${RUN:-$PROJ/runs/mlipcpp-step-2pow24-big6adapt16-muon10-145880}
INIT=${INIT:-$PROJ/runs/mlipcpp-bench-adapt16-145278/init.pt}
DATA=${DATA:-/home/boittier/data/madcore/memmap-2pow15}
MD=${MD:-} # mlipcpp-md binary for the comparison check (optional)
OPTIONS=/home/boittier/metawork/madcore/options-pet-big6-adapt16-cons-ft.yaml
OUT=$PROJ/finetune/cons-ft-$STAGE-$SLURM_JOB_ID
export PYTHONPATH=$MTT_SRC
C=/work/cosmo/boittier/cache
export XDG_CACHE_HOME=$C/xdg TORCH_HOME=$C/torch TRITON_CACHE_DIR=$C/triton PIP_CACHE_DIR=$C/pip
mkdir -p $OUT && cd $OUT
echo "host $(hostname)  stage $STAGE  run $RUN  data $DATA  metatrain $(git -C $MTT_SRC/.. rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. GGUF -> checkpoint + baselines, and the round trip back to GGUF
$VPY -W ignore $MLIPCPP/scripts/train/gguf_to_checkpoint.py --init $INIT --gguf $RUN/model.gguf -o $OUT/start
$VPY -W ignore $MLIPCPP/scripts/export_pytorch/export_pet_gguf.py --model $OUT/start.ckpt --nc-forces \
    -o $OUT/roundtrip.gguf > roundtrip.log 2>&1 || { tail -20 roundtrip.log; exit 1; }
$VPY - $RUN/model.gguf $OUT/roundtrip.gguf <<'EOF'
import json, sys
import numpy as np
sys.path.insert(0, "/home/boittier/metawork/mlip.cpp/scripts/train")
from gguf_to_checkpoint import read_gguf
(ma, ta), (mb, tb) = read_gguf(sys.argv[1]), read_gguf(sys.argv[2])
names = json.loads(ma["train.param_names"])
bad = [n for n in names if n not in tb or not np.array_equal(ta[n], tb[n])]
print(f"round trip: {len(names) - len(bad)}/{len(names)} trained tensors identical" + (f"; differ: {bad[:5]}" if bad else ""))
sys.exit(1 if bad else 0)
EOF

# 2. options: the template + read_from + the GGUF's baselines
stage_data() {
    LOCAL=${TMPDIR:-/tmp}/$(basename $DATA)
    cp -rL $DATA $LOCAL
    for d in train val test; do ln -sfn $LOCAL/$d $d; done
}
make_options() { # out lr epochs; EWEIGHT (default 1) weights the energy term against the
    # position-gradient term (FWEIGHT, default 1) and the direct-force term (NCWEIGHT, default 1); FLOSS=huber
    # (default mse) uses a Huber loss with FDELTA (default 0.5, scaled force units) on both force terms
    $VPY - $OPTIONS $OUT/start.ckpt $OUT/start.baselines.yaml "$1" "$2" "$3" "${EWEIGHT:-1}" \
        "${NCWEIGHT:-1}" "${FLOSS:-mse}" "${FDELTA:-0.5}" "${FWEIGHT:-1}" <<'EOF'
import sys, yaml
opts, ckpt, base, out, lr, epochs, ew, ncw, floss, fdelta, fw = sys.argv[1:]
o = yaml.safe_load(open(opts))
b = yaml.safe_load(open(base))
t = o["architecture"]["training"]
t["finetune"]["read_from"] = ckpt
t["atomic_baseline"] = b["atomic_baseline"]
t["fixed_scaling_weights"] = b["fixed_scaling_weights"]
t["learning_rate"] = float(lr)
t["num_epochs"] = int(epochs)
if (float(ew), float(ncw), floss, float(fw)) != (1.0, 1.0, "mse", 1.0):
    f = {"type": floss, "reduction": "mean"} | ({"delta": float(fdelta)} if floss == "huber" else {})
    t["loss"] = {
        "energy": {"type": "mse", "weight": float(ew), "reduction": "mean",
                   "gradients": {"positions": f | {"weight": float(fw)}}},
        "non_conservative_force": f | {"weight": float(ncw)},
    }
yaml.safe_dump(o, open(out, "w"), sort_keys=False)
EOF
}
stage_data
( while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; done ) > gpumem.log &
sampler=$!
trap "kill $sampler 2>/dev/null || true" EXIT

if [ $STAGE = smoke ]; then
    # 3. lr 1e-12: the validation errors after "training" are the converted model's
    make_options gate.yaml 1e-12 1
    start=$(date +%s)
    $MTT train gate.yaml -o gate.pt -e extensions > gate.log 2>&1 || { tail -40 gate.log; exit 1; }
    echo "gate epoch: $(( $(date +%s) - start )) s"
    grep -E "Epoch|validation|RMSE|MAE" gate.log | tail -12
    if [ -n "$MD" ]; then
        echo "--- mlipcpp-md check on the same validation split (FP32)"
        GGML_CUDA_NO_TF32=1 $MD check $RUN/model.gguf val --n 100000 --fd 0 | grep -E "structures|vs labels"
    fi
    # 4. one real epoch: throughput, memory, loss going down
    make_options real.yaml 2e-5 1
    start=$(date +%s)
    $MTT train real.yaml -o real.pt -e extensions > real.log 2>&1 || { tail -40 real.log; exit 1; }
    echo "real epoch: $(( $(date +%s) - start )) s for $(( $(stat -c %s train/energy.bin) / 4 )) structures, peak GPU MiB $(sort -n gpumem.log | tail -1)"
    grep -E "Epoch|validation|RMSE|MAE" real.log | tail -12
else
    make_options options.yaml ${LR:-2e-5} ${EPOCHS:-4}
    start=$(date +%s)
    # 147619 died with "can't start new thread" in the DataLoader workers:
    # fixed worker count, small per-process thread pools
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    srun --ntasks=$SLURM_NTASKS $MTT train options.yaml \
        -r "architecture.training.distributed=true" \
        -r "architecture.training.num_workers=${WORKERS:-4}" \
        -r "architecture.training.distributed_port=$((20000 + SLURM_JOB_ID % 20000))" \
        -o model.pt -e extensions
    echo "elapsed $(( $(date +%s) - start )) s, peak GPU MiB $(sort -n gpumem.log | tail -1)"
    $VPY -W ignore $MLIPCPP/scripts/export_pytorch/export_pet_gguf.py --model model.ckpt --nc-forces -o model.gguf
fi
