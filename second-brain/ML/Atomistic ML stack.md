---
tags: [ml, atomistic, metatensor]
---
# Atomistic ML stack

How the repos in metawork fit together:

```
          data (extxyz, MAD-CORE, QM7-X …)
                    │
   featomic ──► descriptors (SOAP, …)        torch-pme ──► long-range
                    │                                    │
              metatensor (TensorMap: sparse blocks, labels, gradients)
                    │
              metatrain  ──  `mtt train options.yaml` ──► model.pt
                    │
              metatomic (model interface + TorchScript export)
                    │
   ┌──────────┬─────┴──────┬──────────┬──────────┐
  ASE       i-PI       LAMMPS     OpenMM    torch-sim
                (engines load the TorchScript model from C++/Python)
```

| Repo | Language | Role |
|---|---|---|
| metatensor | Rust core + C API, C++/Python/torch bindings | sparse labelled tensor format |
| metatomic | C++/Python (torch) | model API, `System`, neighbor lists, export |
| featomic | Rust + C/C++/Python | descriptor calculators |
| metatrain | Python | training CLI/architectures (SOAP-BPNN, PET, LOREM…) |
| lorem-jax | JAX | reference LOREM implementation |
| openmm-metatomic / openmm-ml | C++/Python | OpenMM plugin |
| lammps / plumed feedstocks | conda | packaged engines → [[Conda feedstock protocol]] |

## Common commands
```bash
mtt train options.yaml -o model.pt
mtt eval model.pt eval.yaml
mtt export model.ckpt -o model.pt     # checkpoint → TorchScript
```

## Where bugs usually hide
- Neighbor lists: cutoff, `full` vs half list, periodic images — engine and model must agree.
- Units and length/energy conventions at the engine boundary.
- dtype (fp32 vs fp64) between model and engine.
- TorchScript compatibility of new architecture code → [[PyTorch extensions and TorchScript]].

Related: [[metawork setup]], [[Training practice]]
