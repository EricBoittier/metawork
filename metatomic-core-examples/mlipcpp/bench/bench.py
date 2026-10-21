"""Time a model from Python, on the same silicon supercells as bench.cpp.

  bench.py plugin <plugin.so> <model.gguf> [cells...]   # metatomic-core Python + plugin
  bench.py torch <model.pt> [cells...]                  # metatomic-torch, TorchScript

Prints CSV lines: wrapper,backend,threads,n_atoms,mode,ms_per_call.
MLIPCPP_BACKEND (plugin) and BENCH_DEVICE (torch) select cpu or cuda.
"""
import os
import random
import sys
import time

import numpy as np


BASIS = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
         (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]


def silicon(n, a=5.43):
    """Same supercells as bench.cpp (rattle differs, irrelevant for timings)."""
    rng = random.Random(7)
    positions = [[a * (shift + b) + 0.05 * (rng.random() - 0.5) for shift, b in zip((i, j, k), basis)]
                 for i in range(n) for j in range(n) for k in range(n) for basis in BASIS]
    return np.full(len(positions), 14, dtype=np.int32), np.array(positions), np.eye(3) * a * n


def time_ms(run):
    for _ in range(3):  # warmup, TorchScript optimizes after a few calls
        run()
    calls, start = 0, time.perf_counter()
    while calls < 3 or ((time.perf_counter() - start) < 2.0 and calls < 50):
        run()
        calls += 1
    return 1000 * (time.perf_counter() - start) / calls


def bench_plugin(plugin, gguf, cells):
    import metatensor
    import metatomic as mta

    mta.load_plugin(plugin)
    metatensor.register_external_data_wrapper("metatensor::SimpleDataArray", metatensor.ExternalCpuArray)
    backend = os.environ.get("MLIPCPP_BACKEND", "cpu")
    model = mta.load_model(gguf, {"backend": backend})
    energy = lambda gradients: mta.Quantity(name="energy", unit="eV", sample_kind="system", gradients=gradients)
    nc = mta.Quantity(name="non_conservative_force", unit="eV/Angstrom", sample_kind="atom")
    requests = {"energy": [energy([])], "forces": [energy(["positions"])], "nc": [energy([]), nc]}

    for n in cells:
        types, positions, cell = silicon(n)
        for mode, request in requests.items():
            def run():
                system = mta.System("angstrom", types.copy(), positions.copy(), cell.copy(), np.ones(3, dtype=bool))
                outputs = mta.execute_model(model, [system], request)
                if mode == "forces":
                    outputs[0].block().gradient("positions").values
            yield "plugin-python", f"ggml-{backend}", len(types), mode, time_ms(run)


def bench_torch(pt, cells):
    # the model is called directly: metatomic-torch's ASE calculator (0.1.x)
    # computes energy, forces and stress whatever properties are requested
    import torch
    from metatomic.torch import ModelEvaluationOptions, ModelOutput, System, load_atomistic_model
    from vesin.metatomic import compute_requested_neighbors

    device = os.environ.get("BENCH_DEVICE", "cpu")
    model = load_atomistic_model(pt).to(device)
    dtype = getattr(torch, model.capabilities().dtype)
    energy = {"energy": ModelOutput(per_atom=False)}
    outputs = {"energy": energy, "forces": energy,
               "nc": {**energy, "non_conservative_force": ModelOutput(per_atom=True)}}
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None

    for n in cells:
        types, positions, cell = silicon(n)
        for mode, requested in outputs.items():
            options = ModelEvaluationOptions(length_unit="angstrom", outputs=requested)

            def run():
                pos = torch.tensor(positions, dtype=dtype, device=device, requires_grad=mode == "forces")
                system = System(torch.tensor(types, device=device), pos,
                                torch.tensor(cell, dtype=dtype, device=device),
                                torch.ones(3, dtype=torch.bool, device=device))
                compute_requested_neighbors(system, "angstrom", model)
                result = model([system], options, check_consistency=False)
                if mode == "forces":
                    forces = -torch.autograd.grad(result["energy"].block().values.sum(), pos)[0]
                    forces.cpu()
                else:
                    result["energy"].block().values.cpu()
                sync()

            yield "metatomic-torch", f"torch-{device}", len(types), mode, time_ms(run)


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "plugin":
        plugin, gguf, *cells = sys.argv[2:]
        rows = bench_plugin(plugin, gguf, [int(c) for c in cells] or [1, 2, 3, 4])
        threads = os.environ.get("MLIPCPP_NUM_THREADS", "4")
    else:
        import torch

        pt, *cells = sys.argv[2:]
        rows = bench_torch(pt, [int(c) for c in cells] or [1, 2, 3, 4])
        threads = str(torch.get_num_threads())
    for wrapper, backend, n, mode, ms in rows:
        print(f"{wrapper},{backend},{threads},{n},{mode},{ms:.2f}", flush=True)
