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
    import ase
    from metatomic.torch.ase_calculator import MetatomicCalculator

    device = os.environ.get("BENCH_DEVICE", "cpu")
    conservative = MetatomicCalculator(pt, device=device)
    direct = MetatomicCalculator(pt, device=device, non_conservative="forces")
    modes = {"energy": (conservative, ["energy"]), "forces": (conservative, ["energy", "forces"]),
             "nc": (direct, ["energy", "forces"])}
    for n in cells:
        types, positions, cell = silicon(n)
        atoms = ase.Atoms(numbers=types, positions=positions, cell=cell, pbc=True)
        for mode, (calc, properties) in modes.items():
            # calling calculate directly skips the ASE results cache
            run = lambda: calc.calculate(atoms, properties, ["positions"])
            yield "metatomic-torch-ase", f"torch-{device}", len(types), mode, time_ms(run)


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
