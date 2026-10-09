---
tags: [gpu, cuda]
---
# CUDA

## Our GPUs
| Cluster | GPU | Compute cap. (`sm_`) | Notes |
|---|---|---|---|
| kuma | L40S | 89 | Ada, low FP64, 48 GB |
| kuma | H100 | 90 | Hopper, 80 GB HBM |
| kuma | `mig24gb` | 90 | H100 MIG slice |
| lyra | B200 | 100 | Blackwell — needs CUDA ≥ 12.8 torch, separate `.venv-blackwell` |
| lyra | RTX 6000 Pro | 120 | Blackwell workstation |

Shared metawork venv = torch 2.5.1+cu121 → max `sm_90`. "no kernel image is available" = arch not compiled in.

## Checks
```bash
nvidia-smi                        # driver version = max CUDA runtime supported
nvidia-smi -q -d CLOCK,POWER
nvidia-smi topo -m                # NVLink / PCIe topology between GPUs
nvcc --version                    # toolkit (login node has none → srun --qos=debug)
python -c "import torch;print(torch.version.cuda, torch.cuda.get_device_capability())"
```
Driver ≥ toolkit: a cu121 wheel runs on any driver supporting 12.1+. See [[Conda and uv envs]] → `fix-torch-cuda.sh`.

## Compile
```bash
nvcc -O3 -arch=sm_90 -lineinfo kernel.cu -o k
nvcc -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 ...
export TORCH_CUDA_ARCH_LIST="8.9;9.0"     # for torch C++/CUDA extensions
```
`-lineinfo` keeps source mapping for Nsight without slowing code.

## Kernel basics
```cuda
__global__ void axpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
axpy<<<(n + 255) / 256, 256>>>(n, a, x, y);
```
- Coalesce: consecutive threads → consecutive addresses.
- Block size multiple of 32 (warp); 128–256 is a good start.
- Shared memory for reuse within a block; watch bank conflicts.
- Atomics (e.g. scatter-add of forces) serialise on collisions — consider sorting/segmented reduction.
- Always check errors: `cudaGetLastError()` after launch; `CUDA_LAUNCH_BLOCKING=1` to make errors appear at the right line.

## Debug & profile
```bash
compute-sanitizer ./k                      # out-of-bounds, races (--tool racecheck)
nsys profile -o rep --trace=cuda,nvtx,osrt python train.py   # timeline
nsys stats rep.nsys-rep
ncu --set full -k regex:my_kernel -o kprof ./k                # per-kernel deep dive
```
Roofline thinking: is the kernel **memory-bound** (bytes/FLOP high, compare to HBM BW) or **compute-bound**? `ncu` "Speed of Light" section answers this.

Related: [[Triton]], [[Benchmarking and profiling]], [[PyTorch extensions and TorchScript]]
