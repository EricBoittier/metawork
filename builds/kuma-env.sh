# Runtime environment for the ONIOM jobs on kuma (local helper, not tracked). Source it before
# etc/oniom/submit.sh or sbatch: the jobs inherit it through --export=ALL.
module purge
module load gcc/13.2.0 cuda/12.4.1
export ONIOM_PY=/home/boittier/metawork/builds/torch-cu126/bin/python
export PATH=/home/boittier/metawork/builds/torch-cu126/bin:$PATH
# torch cu126 needs the CUDA 12.6 runtime it ships; the 12.4 module's libcudart is too old
export LD_LIBRARY_PATH=/home/boittier/metawork/builds/torch-cu126/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}
