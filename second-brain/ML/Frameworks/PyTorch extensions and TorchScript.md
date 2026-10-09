---
tags: [ml, pytorch, cpp, torchscript]
---
# PyTorch extensions and TorchScript

Why it matters: metatomic models are exported as **TorchScript** so engines (LAMMPS, i-PI, OpenMM, ASE) can run them from C++ without Python; metatensor-torch/featomic-torch register **custom ops / classes** in C++.

## TorchScript
```python
scripted = torch.jit.script(model)     # compiles Python source; needs type annotations
scripted.save("model.pt")
m = torch.jit.load("model.pt", map_location="cuda")
print(scripted.code)                   # see what got compiled
```
Common script failures:
- missing type hints on non-Tensor args → annotate (`Optional[Tensor]`, `Dict[str, Tensor]`)
- Python features: `**kwargs`, closures, comprehension over non-lists, most third-party calls
- `@torch.jit.unused` / `@torch.jit.ignore` to skip methods
- Custom classes need `@torch.jit.script` or a C++ `torch::CustomClassHolder`
C++ side:
```cpp
auto m = torch::jit::load("model.pt");
auto out = m.forward({inputs}).toTensor();
```

## Custom C++ op (TORCH_LIBRARY)
```cpp
#include <torch/library.h>
torch::Tensor my_op(torch::Tensor x) { return x * 2; }
TORCH_LIBRARY(myns, m) { m.def("my_op(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(myns, CPU, m)  { m.impl("my_op", my_op); }
TORCH_LIBRARY_IMPL(myns, CUDA, m) { m.impl("my_op", my_op_cuda); }
```
Python: `torch.ops.load_library("libmy.so"); torch.ops.myns.my_op(x)`.
Autograd: implement a `torch::autograd::Function<...>` with `forward`/`backward`, or register under `Autograd` dispatch key.

## Building
```bash
# against the installed torch
cmake -S . -B build -DCMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)")
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
```
Quick prototyping: `torch.utils.cpp_extension.load(name, sources=[...], verbose=True)` (JIT build).

## Gotchas
- An extension `.so` is tied to the exact torch version + CXX11 ABI it was built against → rebuild after any torch change (`fix-torch-cuda.sh` does this for metatensor/metatomic/featomic-torch).
- `undefined symbol: _ZN...c10...` = torch version mismatch.
- CUDA kernels can't be built on kuma login node (no `nvcc`) → `srun --qos=debug` or `--qos=build` + a CUDA module.
- torch.compile can't trace into TorchScript / opaque custom ops unless they're registered with fake/meta kernels (`torch.library.register_fake`).

Related: [[PyTorch]], [[C++ and CMake]], [[CUDA]], [[Atomistic ML stack]]
