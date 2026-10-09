"""Copy geometry/results to ASE and chemiscope outside autodiff/compiled code.

These are tutorial display adapters, not part of the metatomic public API.
Energy/stress remain structure properties: no fictitious atomic energy or stress
partition is introduced. Arrow lengths are visual aids with an explicit scale.
"""

import chemiscope
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.visualize.plot import plot_atoms


def as_numpy(value):
    """Copy NumPy/JAX arrays or detached CPU Torch tensors for visualization."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def system_to_atoms(system):
    """Convert these tutorials' atomic-number types and angstrom geometry to ASE."""
    if system.length_unit.lower() != "angstrom":
        raise ValueError("The tutorial viewer requires geometry in angstrom")
    return Atoms(
        numbers=as_numpy(system.types),
        positions=as_numpy(system.positions),
        cell=as_numpy(system.cell),
        pbc=as_numpy(system.pbc),
    )


def make_dataset(
    systems, *, energies=None, forces=None, stresses=None, force_scale=0.25
):
    """Build an explicit, unit-labeled chemiscope dataset (one sample per system)."""
    frames = [system_to_atoms(system) for system in systems]
    if not frames:
        raise ValueError("At least one structure is required")
    if not np.isfinite(force_scale) or force_scale <= 0:
        raise ValueError("force_scale must be a finite positive number")
    properties = {
        "frame": {"target": "structure", "values": list(range(len(frames)))},
        "volume": {
            "target": "structure",
            "values": [abs(np.linalg.det(f.cell)) for f in frames],
            "units": "angstrom^3",
        },
    }
    for label, values in (
        ("energies", energies),
        ("forces", forces),
        ("stresses", stresses),
    ):
        if values is not None and len(values) != len(frames):
            raise ValueError(f"{label} must have one entry per structure")
    if energies is not None:
        properties["energy"] = {
            "target": "structure",
            "values": [float(as_numpy(e)) for e in energies],
            "units": "eV",
        }
    shapes = {}
    if forces is not None:
        arrays = [as_numpy(f) for f in forces]
        for frame, array in zip(frames, arrays, strict=True):
            if array.shape != (len(frame), 3):
                raise ValueError("Forces must have shape (n_atoms, 3)")
            frame.new_array("forces", array)
        properties["force_magnitude"] = {
            "target": "atom",
            "values": np.concatenate([np.linalg.norm(f, axis=1) for f in arrays]),
            "units": "eV/angstrom",
        }
        properties["max_force"] = {
            "target": "structure",
            "values": [float(np.linalg.norm(f, axis=1).max()) for f in arrays],
            "units": "eV/angstrom",
        }
        shapes["forces"] = chemiscope.ase_vectors_to_arrows(
            frames,
            "forces",
            target="atom",
            scale=force_scale,
            radius=0.05,
        )
        shapes["forces"]["parameters"]["global"]["color"] = "#d62728"
    if stresses is not None:
        tensors = np.stack([as_numpy(s) for s in stresses])
        if tensors.shape != (len(frames), 3, 3):
            raise ValueError("Stress must have shape (3, 3) per structure")
        for i, a in enumerate("xyz"):
            for j, b in enumerate("xyz"):
                properties[f"stress_{a}{b}"] = {
                    "target": "structure",
                    "values": tensors[:, i, j],
                    "units": "eV/angstrom^3",
                }
        properties["pressure"] = {
            "target": "structure",
            "values": -np.trace(tensors, axis1=1, axis2=2) / 3,
            "units": "eV/angstrom^3",
        }
    return {"structures": frames, "properties": properties, "shapes": shapes}


def _viewer(dataset, *, trajectory=False):
    structure_settings = {
        "unitCell": True,
        "axes": "xyz",
        "keepOrientation": True,
        "bonds": False,
    }
    if dataset["shapes"]:
        structure_settings["shape"] = "forces"
    settings = {"structure": [structure_settings]}
    if trajectory:
        settings["map"] = {
            "x": {"property": "frame"},
            "y": {"property": "energy"},
            "color": {"property": "max_force"},
            "joinPoints": True,
        }
    return mo.ui.anywidget(
        chemiscope.show(
            **dataset,
            settings=settings,
            mode="default" if trajectory else "structure",
            metadata={
                "name": "metatomic tutorial",
                "description": "Positions/cell in angstrom; total energy and stress are per structure.",
            },
        )
    )


def ase_plot(system, forces=None, *, force_scale=0.25):
    """Static xy projection: ASE atoms/cell plus projected force arrows."""
    atoms = system_to_atoms(system)
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_atoms(atoms, ax=ax, rotation="0x,0y,0z", radii=0.25, show_unit_cell=2)
    if forces is not None:
        vectors = as_numpy(forces)
        # ASE translates its rendered scene. Align arrows with the Circle patches
        # rather than assuming that plotted x/y coordinates equal world x/y.
        from matplotlib.patches import Circle

        centers = np.array(
            [patch.center for patch in ax.patches if isinstance(patch, Circle)]
        )
        # ASE draws atoms in ascending projected z order.
        order = np.argsort(atoms.positions[:, 2], kind="stable")
        if len(centers) != len(atoms):
            raise RuntimeError("ASE rendered an unexpected number of atom markers")
        ax.quiver(
            centers[:, 0],
            centers[:, 1],
            vectors[order, 0],
            vectors[order, 1],
            angles="xy",
            scale_units="xy",
            scale=1 / force_scale,
            color="tab:red",
            width=0.008,
        )
    ax.set_title("ASE xy projection (z components are not visible)")
    ax.set_axis_off()
    fig.tight_layout()
    rendered = mo.as_html(fig)
    plt.close(fig)
    return rendered


def structure_panel(system, *, energy=None, forces=None, stress=None, force_scale=0.25):
    """Interactive 3D view, numerical tables, and a static ASE fallback."""
    dataset = make_dataset(
        [system],
        energies=None if energy is None else [energy],
        forces=None if forces is None else [forces],
        stresses=None if stress is None else [stress],
        force_scale=force_scale,
    )
    atoms = dataset["structures"][0]
    rows = []
    for i, (symbol, position) in enumerate(
        zip(atoms.get_chemical_symbols(), atoms.positions, strict=True)
    ):
        row = {
            "atom": i,
            "element": symbol,
            **{f"{a} (angstrom)": float(position[j]) for j, a in enumerate("xyz")},
        }
        if forces is not None:
            row.update(
                {
                    f"F{a} (eV/angstrom)": float(as_numpy(forces)[i, j])
                    for j, a in enumerate("xyz")
                }
            )
        rows.append(row)
    tables = [
        mo.md("**Positions and forces**"),
        mo.ui.table(rows, selection=None),
        mo.md("**Cell vectors (rows, angstrom)**"),
        mo.ui.table(
            [
                {
                    "vector": a,
                    **{b: float(atoms.cell[i, j]) for j, b in enumerate("xyz")},
                }
                for i, a in enumerate("abc")
            ],
            selection=None,
        ),
    ]
    if stress is not None:
        tensor = as_numpy(stress)
        tables.extend(
            [
                mo.md("**Stress (eV/angstrom³; positive tension)**"),
                mo.ui.table(
                    [
                        {
                            "row": a,
                            **{b: float(tensor[i, j]) for j, b in enumerate("xyz")},
                        }
                        for i, a in enumerate("xyz")
                    ],
                    selection=None,
                ),
                mo.md(f"Pressure = **{-np.trace(tensor) / 3:.6g} eV/angstrom³**"),
            ]
        )
    summary = (
        ""
        if energy is None
        else f"Total energy: **{float(as_numpy(energy)):.6f} eV**. "
    )
    summary += f"Cell volume: **{abs(np.linalg.det(atoms.cell)):.3f} angstrom³**."
    instructions = "Drag to rotate; scroll to zoom. The wireframe shows the unit cell."
    table_label = "Coordinates and cell"
    if forces is not None:
        instructions += (
            f" Force arrows: **{force_scale:g} angstrom per (eV/angstrom)**."
        )
        table_label = "Coordinates, forces, and cell"
    if stress is not None:
        instructions += " Stress is a cell property, shown numerically rather than assigned to atoms."
        table_label = (
            "Coordinates, forces, cell, and stress"
            if forces is not None
            else "Coordinates, cell, and stress"
        )
    return mo.vstack(
        [
            mo.md("### Structure and properties\n" + summary),
            _viewer(dataset),
            mo.md(instructions),
            mo.accordion(
                {
                    table_label: mo.vstack(tables),
                    "ASE static plot (also works without WebGL)": ase_plot(
                        system, forces, force_scale=force_scale
                    ),
                }
            ),
        ]
    )


def trajectory_panel(systems, energies, forces, *, stresses=None, force_scale=0.25):
    """Link simulation energy/force history to geometry, arrows and optional stress."""
    dataset = make_dataset(
        systems,
        energies=energies,
        forces=forces,
        stresses=stresses,
        force_scale=force_scale,
    )
    return mo.vstack(
        [
            mo.md(
                "### Simulation trajectory\nSelect a point in the energy plot to inspect that step's atoms, cell, and forces. "
                f"Force-arrow scale: {force_scale:g} angstrom per (eV/angstrom)."
            ),
            _viewer(dataset, trajectory=True),
        ]
    )
