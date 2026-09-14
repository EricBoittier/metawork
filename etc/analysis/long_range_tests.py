"""
Sanity checks for whether an interatomic potential has any dependence beyond
its stated short-range cutoff.

Usage (from a notebook in this directory):

    from long_range_tests import displace_atom_test, fragment_separation_test

    locality = displace_atom_test(atoms, calc, probe_index=-1,
                                   distances=np.linspace(0, 15, 16))

    fragmentation = fragment_separation_test(atoms, calc, fragment_a, fragment_b,
                                              separations=np.linspace(0, 15, 16))
"""

import numpy as np


def displace_atom_test(atoms, calc, probe_index, distances, direction=None,
                        reference_indices=None):
    """
    Locality / receptive-field test.

    Move `probe_index` away along `direction` by each amount in `distances` (in
    Angstrom, measured from its original position) and recompute the energy and
    the forces on `reference_indices`. A strictly local model (interaction_range
    <= some cutoff) should give energy/forces on the reference atoms that stop
    changing once the probe atom is displaced past that cutoff; a model with a
    genuine long-range term (electrostatics, charge transfer, ...) will keep
    responding as the probe moves further away.

    `atoms` is not modified; a fresh copy is displaced and given `calc` at each
    step.

    Returns a dict with "distance", "energy", "forces" (shape
    (len(distances), len(reference_indices), 3)) and "max_force_change" (the
    largest force-vector-norm change on any reference atom, relative to the
    first entry in `distances`).
    """
    distances = np.asarray(distances, dtype=float)

    if reference_indices is None:
        reference_indices = [i for i in range(len(atoms)) if i != probe_index]

    if direction is None:
        centroid = atoms.positions[reference_indices].mean(axis=0)
        direction = atoms.positions[probe_index] - centroid
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError(
                "probe atom sits at the reference centroid; pass `direction` "
                "explicitly"
            )
        direction = direction / norm
    else:
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)

    original_position = atoms.positions[probe_index].copy()

    energies = np.empty(len(distances))
    forces = np.empty((len(distances), len(reference_indices), 3))

    for i, d in enumerate(distances):
        probe_atoms = atoms.copy()
        probe_atoms.positions[probe_index] = original_position + direction * d
        probe_atoms.calc = calc

        energies[i] = probe_atoms.get_potential_energy()
        forces[i] = probe_atoms.get_forces()[reference_indices]

    force_change = np.linalg.norm(forces - forces[0], axis=-1)  # (n_dist, n_ref)
    max_force_change = force_change.max(axis=-1)

    return {
        "distance": distances,
        "energy": energies,
        "forces": forces,
        "max_force_change": max_force_change,
    }


def fragment_separation_test(atoms, calc, fragment_a, fragment_b, separations,
                              axis=None):
    """
    Size-consistency / interaction-decay test.

    Rigidly translate `fragment_b` away from `fragment_a` along `axis` by each
    amount in `separations` (in Angstrom, added on top of the fragments'
    original separation) and compute the "interaction energy"

        E_int(s) = E_total(s) - (E_A_isolated + E_B_isolated)

    where E_A_isolated / E_B_isolated are the energies of each fragment computed
    alone (same internal geometry, no relaxation). For a model with a finite
    interaction range, E_int should decay to ~0 once `s` exceeds that range; a
    persistent non-zero tail indicates a long-range term (or a discontinuity /
    lack of size-consistency, if this wasn't expected).

    `fragment_a` and `fragment_b` are lists of atom indices into `atoms` (they
    need not cover every atom in `atoms` - only the two fragments being pulled
    apart are moved; anything else is left untouched). `atoms` is not modified.

    Returns a dict with "separation", "energy_total", "energy_fragment_a",
    "energy_fragment_b" (each a scalar) and "interaction_energy".
    """
    separations = np.asarray(separations, dtype=float)

    centroid_a = atoms.positions[fragment_a].mean(axis=0)
    centroid_b = atoms.positions[fragment_b].mean(axis=0)

    if axis is None:
        axis = centroid_b - centroid_a
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError("fragments share a centroid; pass `axis` explicitly")
        axis = axis / norm
    else:
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

    fragment_a_atoms = atoms[fragment_a]
    fragment_a_atoms.calc = calc
    energy_a = fragment_a_atoms.get_potential_energy()

    fragment_b_atoms = atoms[fragment_b]
    fragment_b_atoms.calc = calc
    energy_b = fragment_b_atoms.get_potential_energy()

    energy_total = np.empty(len(separations))
    for i, s in enumerate(separations):
        pulled_atoms = atoms.copy()
        pulled_atoms.positions[fragment_b] += axis * s
        pulled_atoms.calc = calc
        energy_total[i] = pulled_atoms.get_potential_energy()

    return {
        "separation": separations,
        "energy_total": energy_total,
        "energy_fragment_a": energy_a,
        "energy_fragment_b": energy_b,
        "interaction_energy": energy_total - (energy_a + energy_b),
    }


def estimate_cutoff(distances, values, threshold):
    """
    Empirically estimate a model's interaction cutoff from a decay curve.

    Given `values` (e.g. `max_force_change` from `displace_atom_test`, or
    `abs(interaction_energy)` from `fragment_separation_test`) sampled at each
    of `distances`, return the smallest distance beyond which `values` stays
    below `threshold` for the remainder of the scan. Returns None if it never
    does (e.g. `distances` didn't go far enough, or the model has a genuine
    long-range tail).
    """
    distances = np.asarray(distances)
    values = np.asarray(values)
    below = values < threshold
    for i in range(len(below)):
        if below[i:].all():
            return distances[i]
    return None


def plot_locality_test(result, ax=None):
    """Quick diagnostic plot for `displace_atom_test`'s output."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(result["distance"], result["max_force_change"],
            color="#4C72B0", linewidth=2)
    ax.set_xlabel("probe atom displacement (Å)")
    ax.set_ylabel("max |Δforce| on reference atoms (eV/Å)")
    ax.set_title("Locality test: force response to a distant probe atom")
    ax.grid(True, alpha=0.3)
    return ax


def plot_fragment_separation_test(result, ax=None):
    """Quick diagnostic plot for `fragment_separation_test`'s output."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(result["separation"], result["interaction_energy"],
            color="#4C72B0", linewidth=2)
    ax.axhline(0, color="0.6", linewidth=1, linestyle="--")
    ax.set_xlabel("additional fragment separation (Å)")
    ax.set_ylabel("interaction energy, E_total - (E_A + E_B) (eV)")
    ax.set_title("Size-consistency test: interaction energy vs. fragment separation")
    ax.grid(True, alpha=0.3)
    return ax
