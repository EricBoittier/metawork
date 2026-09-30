#!/usr/bin/env bash
# Build and install GROMACS for the ONIOM jobs: the oniom-stress-fixes branch of
# EricBoittier/gromacs_metatensor with metatomic (Torch).
#
#   etc/oniom/build-gromacs.sh            single precision, non-bondeds and PME on the GPU -> $GMX
#   etc/oniom/build-gromacs.sh --double   double precision, MM on the CPU (GROMACS has no GPU
#                                         support in double), model on the GPU  -> $GMX_D
#
# Run it on a compute node of the target cluster (the CUDA architectures and the CPU SIMD
# are chosen for it), with ONIOM_PY's torch. Paths and the cluster come from env.sh; set
# GMX_SRC to build an existing checkout, JOBS for the build parallelism.
set -euo pipefail
. "$(dirname "$0")/env.sh"

double=OFF suffix=
if [[ ${1:-} == --double ]]; then double=ON suffix=-double; fi

: "${GMX_SRC:=$METAWORK/gromacs-oniom-src}"
if [[ ! -d $GMX_SRC ]]; then
    git clone --branch oniom-stress-fixes https://github.com/EricBoittier/gromacs_metatensor.git "$GMX_SRC"
fi

# CUDA architectures of each cluster's GPUs
case $ONIOM_CLUSTER in
    kuma) arch="89;90" ;;                 # L40S, H100
    lyra) arch="100;120" ;;               # B200, RTX PRO 6000 (Blackwell)
    daint | clariden) arch="90" ;;        # GH200
    *) arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .) ;;
esac
torch_arch=$(echo "$arch" | sed 's/\([0-9]\)\([0-9]\)\(;\|$\)/\1.\2\3/g')  # 89;90 -> 8.9;9.0

torch_prefix=$("$ONIOM_PY" -c "import torch; print(torch.utils.cmake_prefix_path)")
nvcc=${CUDACXX:-$(command -v nvcc || echo "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc")}

if [[ $double == ON ]]; then
    prefix=$(dirname "$(dirname "$GMX_D")") gpu=OFF
else
    prefix=$(dirname "$(dirname "$GMX")") gpu=CUDA
fi
build=$GMX_SRC/build-$ONIOM_CLUSTER-$(uname -m)$suffix

cmake -S "$GMX_SRC" -B "$build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DGMX_DOUBLE=$double \
    -DGMX_GPU=$gpu \
    -DGMX_MPI=OFF \
    -DGMX_METATOMIC=TORCH -DDOWNLOAD_METATOMIC=ON \
    -DGMX_NNPOT=OFF \
    -DCMAKE_PREFIX_PATH="$torch_prefix" \
    -DPython_EXECUTABLE="$ONIOM_PY" \
    -DCMAKE_CUDA_COMPILER="$nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" \
    -DTORCH_CUDA_ARCH_LIST="$torch_arch" \
    -DGMX_BUILD_OWN_FFTW="${GMX_BUILD_OWN_FFTW:-ON}" \
    -DBUILD_TESTING=OFF -DGMX_BUILD_MANUAL=OFF
cmake --build "$build" -j "${JOBS:-16}" --target install
echo "installed $prefix (double precision: $double, GPU: $gpu, CUDA arch $arch)"
