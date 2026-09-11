"""Run in the metatrain venv. Produces the two-comparison parity report
(torch-vs-JAX checkpoint fidelity, torch-vs-DFT model accuracy) described in
RESULTS_bio_dimers.md, for any dumped checkpoint + dataset.

Usage::

    .venv/bin/python etc/lorem-parity/jax_checkpoint_parity/plot_parity.py \\
        lorem-tmlr-archive/evals/bio_dimers/lorem/run/checkpoints/R2_E+F/model/model.yaml \\
        lorem-tmlr-archive/datasets/bio_dimers_test.xyz \\
        bio_dimers_reference_200.npz \\
        --out bio_dimers_parity_report.pdf
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from compare import build_model, load_model_yaml, predict
from metatomic.torch import NeighborListOptions
from metatrain.experimental.lorem.modules.jax_parity_checkpoint import load_checkpoint

matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_yaml", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("reference_npz", type=Path)
    parser.add_argument("--out", type=Path, default=Path("parity_report.pdf"))
    args = parser.parse_args()

    hypers, baseline = load_model_yaml(args.model_yaml)
    reference = np.load(args.reference_npz)
    flax_params = {
        k[len("param__"):].replace("__SLASH__", "/"): v
        for k, v in reference.items()
        if k.startswith("param__")
    }
    jax_energies = reference["jax_energies"]
    jax_forces = reference["jax_forces"]
    jax_force_sizes = reference["jax_force_sizes"]
    n_frames = len(jax_energies)

    nlo = NeighborListOptions(cutoff=hypers["cutoff"], full_list=True, strict=True)
    atomic_types = sorted(int(z) for z in baseline)
    backbone, long_range = build_model(hypers, atomic_types, nlo)
    load_checkpoint(backbone, long_range, flax_params)
    backbone.eval()
    long_range.eval()

    atoms_list = read(args.dataset, index=f"0:{n_frames}")
    torch_energies, dft_energies = [], []
    force_start = 0
    f_vs_jax, f_vs_dft = [], []
    for atoms, n_atoms in zip(atoms_list, jax_force_sizes):
        energy, forces = predict(backbone, long_range, atoms, baseline, want_forces=True)
        torch_energies.append(energy)
        dft_energies.append(atoms.get_potential_energy())
        f_vs_jax.append(forces - jax_forces[force_start : force_start + n_atoms])
        f_vs_dft.append(forces - atoms.get_forces())
        force_start += n_atoms

    torch_energies = np.array(torch_energies)
    dft_energies = np.array(dft_energies)
    sizes = np.array([len(a) for a in atoms_list])
    f_vs_jax = np.concatenate(f_vs_jax, axis=0) * 1000
    f_vs_dft = np.concatenate(f_vs_dft, axis=0) * 1000
    e_vs_jax = (torch_energies - jax_energies) / sizes * 1000
    e_vs_dft = (torch_energies - dft_energies) / sizes * 1000

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    ax.scatter(jax_energies / sizes, torch_energies / sizes, s=14, color="#2980b9", alpha=0.7)
    lo, hi = (jax_energies / sizes).min(), (jax_energies / sizes).max()
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlabel("JAX reference energy/atom [eV]")
    ax.set_ylabel("torch port energy/atom [eV]")
    ax.set_title(
        f"Checkpoint-loading fidelity: energy\nRMSE={np.sqrt(np.mean(e_vs_jax**2)):.6f} "
        f"meV/atom (n={n_frames} frames)"
    )

    ax = axes[0, 1]
    ax.bar(np.arange(n_frames), e_vs_jax, color="#2980b9", width=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("frame index")
    ax.set_ylabel("torch - JAX [meV/atom]")
    ax.set_title("Checkpoint-loading residual, per frame\n(float32 noise floor)")

    ax = axes[0, 2]
    ax.hist(f_vs_jax.flatten(), bins=80, color="#2980b9")
    ax.set_xlabel("torch - JAX force component [meV/A]")
    ax.set_ylabel("count")
    ax.set_title(f"Checkpoint-loading fidelity: forces\nRMSE={np.sqrt(np.mean(f_vs_jax**2)):.6f} meV/A")
    ax.set_yscale("log")

    ax = axes[1, 0]
    ax.scatter(dft_energies / sizes, torch_energies / sizes, s=14, color="#c0392b", alpha=0.7)
    lo, hi = (dft_energies / sizes).min(), (dft_energies / sizes).max()
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlabel("DFT energy/atom [eV]")
    ax.set_ylabel("torch port energy/atom [eV]")
    ax.set_title(f"Model accuracy vs. DFT: energy\nRMSE={np.sqrt(np.mean(e_vs_dft**2)):.4f} meV/atom")

    ax = axes[1, 1]
    ax.bar(np.arange(n_frames), e_vs_dft, color="#c0392b", width=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("frame index")
    ax.set_ylabel("torch - DFT [meV/atom]")
    ax.set_title("Model-accuracy residual, per frame")

    ax = axes[1, 2]
    ax.hist(f_vs_dft.flatten(), bins=80, color="#c0392b")
    ax.set_xlabel("torch - DFT force component [meV/A]")
    ax.set_ylabel("count")
    ax.set_title(f"Model accuracy vs. DFT: forces\nRMSE={np.sqrt(np.mean(f_vs_dft**2)):.4f} meV/A")
    ax.set_yscale("log")

    fig.suptitle(
        f"{args.dataset.stem} checkpoint parity: torch port vs. lorem-jax, {n_frames} frames",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.text(
        0.5, 0.005,
        "Top row = does the torch port reproduce the SAME checkpoint's predictions as JAX "
        "(weight-loading fidelity)?  Bottom row = how accurate is that checkpoint against "
        "real DFT labels (model quality, comparable to the paper's own reported numbers)?",
        ha="center", fontsize=9, style="italic",
    )
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    print(f"energy vs JAX:  RMSE={np.sqrt(np.mean(e_vs_jax**2)):.6f} meV/atom")
    print(f"energy vs DFT:  RMSE={np.sqrt(np.mean(e_vs_dft**2)):.4f} meV/atom")
    print(f"forces vs JAX:  RMSE={np.sqrt(np.mean(f_vs_jax**2)):.6f} meV/A")
    print(f"forces vs DFT:  RMSE={np.sqrt(np.mean(f_vs_dft**2)):.4f} meV/A")


if __name__ == "__main__":
    main()
